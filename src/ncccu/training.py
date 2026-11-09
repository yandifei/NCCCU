"""ConvNeXt 多标签训练、验证及按轮恢复。"""

import csv
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ColorJitter, Compose, ToTensor
from tqdm import tqdm

from ncccu.model import ARCHITECTURE, CHECKPOINT_FORMAT, build_model, read_checkpoint
from ncccu.multilabel import CLASS_NAMES, ROOT, ChestDataset, aggregate, device_for, evaluate, metrics, output_dir
from ncccu.runtime import configure_randomness, plot_history, seed_worker


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def write_results(run: Path, history: list[dict]) -> None:
    temporary = run / "results.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(run / "results.csv")


def train_multilabel(
    *, model: str | Path, data: str | Path = "datasets", name: str | None = None,
    epochs: int = 40, patience: int = 12, imgsz: int = 512,
    batch: int = 8, nbs: int = 128, workers: int = 6, device: str | int = 0,
    lr0: float = 1e-4, lrf: float = 0.01, warmup_epochs: int = 3,
    weight_decay: float = 5e-4, dropout: float = 0.2, brightness: float = 0.1,
    amp: bool = True, seed: int = 0, deterministic: bool = True,
    channels_last: bool = True, smoke_test: bool = False, resume: bool = False,
) -> Path:
    config = dict(locals())
    weights = (ROOT / model).resolve()
    checkpoint = read_checkpoint(weights) if resume else None
    if resume:
        if "training_state" not in checkpoint or weights.name != "last.pt":
            raise ValueError("续训须选择训练目录中的 weights/last.pt")
        config = checkpoint["training_state"]["config"].copy()
        config["device"] = device
    else:
        config.update(model=str(weights), data=str((ROOT / data).resolve()))

    batch, nbs, epochs = config["batch"], config["nbs"], config["epochs"]
    if min(batch, nbs, epochs) < 1 or config["workers"] < 0:
        raise ValueError("batch、nbs、epochs 须为正整数，workers 须为非负整数")
    if config["imgsz"] < 32 or config["imgsz"] % 32:
        raise ValueError("imgsz 须为 32 的正整数倍")
    if (not 0 <= config["brightness"] <= 1 or not 0 <= config["dropout"] < 1
            or not 0 < config["lr0"] < math.inf or not 0 < config["lrf"] <= 1
            or not 0 <= config["weight_decay"] < math.inf or config["patience"] < 0
            or not 0 <= config["warmup_epochs"] < epochs):
        raise ValueError("学习率、预热、早停或增强参数超出范围")
    if not resume and not weights.is_file():
        raise FileNotFoundError(f"找不到 ImageNet 初始化权重：{weights}")

    target = device_for(config["device"])
    configure_randomness(config["seed"], config["deterministic"])
    if resume and checkpoint["training_state"]["device_type"] != target.type:
        raise ValueError("续训须使用与原训练相同的设备类型")
    print(f"正在准备 {ARCHITECTURE}，输入 {config['imgsz']}，设备 {target}", flush=True)
    train = ChestDataset(Path(config["data"]), "train", config["imgsz"], config["smoke_test"])
    valid = ChestDataset(Path(config["data"]), "valid", config["imgsz"], config["smoke_test"])
    if config["brightness"]:
        train.transform = Compose([ColorJitter(brightness=config["brightness"]), ToTensor()])
    print(f"训练 {len(train)} 张，验证 {len(valid)} 张", flush=True)
    model = build_model(config["dropout"], None if resume else weights)
    if resume:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(target)
    if config["channels_last"]:
        model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr0"], weight_decay=config["weight_decay"])
    amp = config["amp"] and target.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history = []
    best, stale = -math.inf, 0
    if resume:
        run = weights.parent.parent
        if not run.is_relative_to((ROOT / "outputs/train").resolve()):
            raise ValueError("续训检查点须位于 outputs/train 内")
        state = checkpoint["training_state"]
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history, best, stale = state["history"], state["best"], state["stale"]
        if len(history) >= epochs or (config["patience"] and stale >= config["patience"]):
            raise ValueError("训练已经完成或触发早停")
    else:
        run = output_dir("train", name)
        (run / "weights").mkdir()
        (run / "args.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    loader_options = dict(batch_size=batch, num_workers=config["workers"],
                          pin_memory=target.type == "cuda", worker_init_fn=seed_worker)
    val_loader = DataLoader(valid, **loader_options)
    accumulation = math.ceil(nbs / batch)
    for epoch in range(len(history), epochs):
        # 每轮由固定种子派生采样、增强和 Dropout 状态，便于从轮次边界恢复。
        epoch_seed = config["seed"] + epoch
        configure_randomness(epoch_seed, config["deterministic"])
        train_loader = DataLoader(train, shuffle=True, generator=torch.Generator().manual_seed(epoch_seed),
                                  **loader_options)
        model.train()
        loss_sum = 0.0
        optimizer.zero_grad(set_to_none=True)
        with tqdm(train_loader, desc=f"Train {epoch + 1}/{epochs}") as progress:
            for step, (images, labels, _) in enumerate(progress):
                if step % accumulation == 0:
                    position = epoch + step / len(train_loader)
                    warmup = config["warmup_epochs"]
                    if position < warmup:
                        factor = max(0.01, position / warmup)
                    else:
                        phase = (position - warmup) / (epochs - warmup)
                        factor = config["lrf"] + (1 - config["lrf"]) * (1 + math.cos(math.pi * phase)) / 2
                    for group in optimizer.param_groups:
                        group["lr"] = config["lr0"] * factor
                images = images.to(target, non_blocking=True)
                labels = labels.to(target, non_blocking=True)
                if config["channels_last"]:
                    images = images.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=target.type, enabled=amp):
                    loss = F.binary_cross_entropy_with_logits(model(images), labels)
                value = loss.detach().item()
                if not math.isfinite(value):
                    raise ValueError("训练损失出现非有限数值")
                group_start = (step // accumulation) * accumulation * batch
                group_size = min(accumulation * batch, len(train) - group_start)
                scaler.scale(loss * (len(images) / group_size)).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                loss_sum += value * len(images)
                if step % 20 == 0:
                    progress.set_postfix(bce=f"{loss_sum / min((step + 1) * batch, len(train)):.4f}",
                                         lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        probabilities, val_loss = evaluate(model, val_loader, target,
                                           channels_last=config["channels_last"], verbose=True)
        _, scores, labels = aggregate(valid, probabilities)
        row = dict(epoch=epoch + 1, train_loss=loss_sum / len(train), val_loss=val_loss,
                   **metrics(labels, scores))
        if row["auc_classes"] != len(CLASS_NAMES) and not config["smoke_test"]:
            raise ValueError("正式验证集必须包含全部类别的正负样本")
        history.append(row)
        score = row["macro_auc"] if row["macro_auc"] is not None else -val_loss
        improved = score > best
        best, stale = (score, 0) if improved else (best, stale + 1)
        weights_only = dict(
            format=CHECKPOINT_FORMAT, architecture=ARCHITECTURE, classes=list(CLASS_NAMES),
            imgsz=config["imgsz"], epoch=epoch + 1, metrics=row, smoke_test=config["smoke_test"],
            state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
        )
        if improved:
            save_checkpoint(run / "weights/best.pt", weights_only)
        state = dict(config=config, optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                     device_type=target.type, history=history, best=best, stale=stale)
        save_checkpoint(run / "weights/last.pt", {**weights_only, "training_state": state})
        write_results(run, history)
        plot_history(history, run)
        print(f"Epoch {epoch + 1} | BCE {row['train_loss']:.5f} | Val {val_loss:.5f} | "
              f"AUC {row['macro_auc']} | Best {best:.5f} | Stale {stale}", flush=True)
        if config["patience"] and stale >= config["patience"]:
            break
    print(f"训练产出：{run}", flush=True)
    return run
