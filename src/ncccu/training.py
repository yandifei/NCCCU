"""ConvNeXt 多标签训练：组织单轮优化、Study 验证和轮次边界恢复。"""

import csv
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ColorJitter, Compose, ToTensor
from tqdm import tqdm

from ncccu.configuration import TrainingConfig
from ncccu.data import CLASS_NAMES, ChestDataset
from ncccu.evaluation import aggregate, evaluate, metrics
from ncccu.model import (
    ARCHITECTURE,
    CHECKPOINT_FORMAT,
    Checkpoint,
    HistoryRow,
    TrainingState,
    build_model,
    read_checkpoint,
)
from ncccu.runtime import (
    ROOT,
    configure_randomness,
    device_for,
    output_dir,
    plot_history,
    seed_worker,
)


@dataclass
class _TrainingRun:
    """当前运行的轮次进度；检查点仍写入原有普通字典格式。"""

    path: Path
    history: list[HistoryRow] = field(default_factory=list)
    best: float = -math.inf
    stale: int = 0


def save_checkpoint(path: Path, checkpoint: Checkpoint) -> None:
    """先写临时文件，再原子替换，保留上一次完整的检查点。"""
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def write_results(run: Path, history: list[HistoryRow]) -> None:
    """按历史字段顺序原子更新结果 CSV。"""
    temporary = run / "results.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(run / "results.csv")


def _resolve_configuration(
    config: TrainingConfig,
) -> tuple[TrainingConfig, Path, Checkpoint | None]:
    """解析初始化权重或恢复配置，恢复时仅允许覆盖运行设备。"""
    weights = (ROOT / config.model).resolve()
    checkpoint = read_checkpoint(weights) if config.resume else None
    if checkpoint is not None:
        if "training_state" not in checkpoint or weights.name != "last.pt":
            raise ValueError("续训须选择训练目录中的 weights/last.pt")
        saved = TrainingConfig.from_dict(checkpoint["training_state"]["config"])
        config = replace(saved, device=config.device)
    else:
        config = replace(config, model=str(weights), data=str((ROOT / config.data).resolve()))
    config.validate()
    if checkpoint is None and not weights.is_file():
        raise FileNotFoundError(f"找不到 ImageNet 初始化权重：{weights}")
    return config, weights, checkpoint


def _prepare_datasets(config: TrainingConfig) -> tuple[ChestDataset, ChestDataset]:
    """训练集使用亮度扰动，验证集保持固定图像预处理。"""
    train = ChestDataset(Path(config.data), "train", config.imgsz, config.smoke_test)
    valid = ChestDataset(Path(config.data), "valid", config.imgsz, config.smoke_test)
    if config.brightness:
        train.transform = Compose([ColorJitter(brightness=config.brightness), ToTensor()])
    return train, valid


def _initialize_run(
    config: TrainingConfig,
    weights: Path,
    checkpoint: Checkpoint | None,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> _TrainingRun:
    """恢复原运行进度，或为新训练创建独立产物目录。"""
    if checkpoint is not None:
        run = weights.parent.parent
        if not run.is_relative_to((ROOT / "outputs/train").resolve()):
            raise ValueError("续训检查点须位于 outputs/train 内")
        state = checkpoint["training_state"]
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        progress = _TrainingRun(run, state["history"], state["best"], state["stale"])
        if len(progress.history) >= config.epochs or (config.patience and progress.stale >= config.patience):
            raise ValueError("训练已经完成或触发早停")
        return progress

    run = output_dir("train", config.name)
    (run / "weights").mkdir()
    (run / "args.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return _TrainingRun(run)


def _learning_rate(config: TrainingConfig, position: float) -> float:
    """按累积组起点计算线性预热及余弦衰减后的学习率。"""
    if position < config.warmup_epochs:
        factor = max(0.01, position / config.warmup_epochs)
    else:
        phase = (position - config.warmup_epochs) / (config.epochs - config.warmup_epochs)
        factor = config.lrf + (1 - config.lrf) * (1 + math.cos(math.pi * phase)) / 2
    return config.lr0 * factor


def _train_epoch(
    network: torch.nn.Module,
    dataset: ChestDataset,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    target: torch.device,
    config: TrainingConfig,
    epoch: int,
) -> float:
    """完成一轮逐图 BCE 优化，返回按样本数加权的平均损失。"""
    # 每轮重新派生采样、增强及 Dropout 状态，恢复不依赖中断前的 RNG 快照。
    epoch_seed = config.seed + epoch
    configure_randomness(epoch_seed, config.deterministic)
    loader = DataLoader(
        dataset,
        batch_size=config.batch,
        num_workers=config.workers,
        pin_memory=target.type == "cuda",
        worker_init_fn=seed_worker,
        shuffle=True,
        generator=torch.Generator().manual_seed(epoch_seed),
    )
    accumulation = math.ceil(config.nbs / config.batch)
    amp = config.amp and target.type == "cuda"
    network.train()
    loss_sum = 0.0
    optimizer.zero_grad(set_to_none=True)
    with tqdm(loader, desc=f"Train {epoch + 1}/{config.epochs}") as progress:
        for step, (images, labels, _) in enumerate(progress):
            if step % accumulation == 0:
                learning_rate = _learning_rate(config, epoch + step / len(loader))
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
            images = images.to(target, non_blocking=True)
            labels = labels.to(target, non_blocking=True)
            if config.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=target.type, enabled=amp):
                loss = F.binary_cross_entropy_with_logits(network(images), labels)
            value = loss.detach().item()
            if not math.isfinite(value):
                raise ValueError("训练损失出现非有限数值")

            # 尾组可能不足 nbs，按实际样本数归一化，避免少量样本的梯度被缩小。
            group_start = (step // accumulation) * accumulation * config.batch
            group_size = min(accumulation * config.batch, len(dataset) - group_start)
            scaler.scale(loss * (len(images) / group_size)).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += value * len(images)
            if step % 20 == 0:
                progress.set_postfix(
                    bce=f"{loss_sum / min((step + 1) * config.batch, len(dataset)):.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )
    return loss_sum / len(dataset)


def _validate_epoch(
    network: torch.nn.Module,
    loader: DataLoader,
    dataset: ChestDataset,
    target: torch.device,
    config: TrainingConfig,
    epoch: int,
    train_loss: float,
) -> HistoryRow:
    """以 FP32 验证逐图损失，并用 Study 平均概率计算选模指标。"""
    probabilities, val_loss = evaluate(
        network, loader, target, channels_last=config.channels_last, verbose=True
    )
    _, scores, labels = aggregate(dataset, probabilities)
    row: HistoryRow = dict(
        epoch=epoch + 1,
        train_loss=train_loss,
        val_loss=val_loss,
        **metrics(labels, scores),
    )
    if row["auc_classes"] != len(CLASS_NAMES) and not config.smoke_test:
        raise ValueError("正式验证集必须包含全部类别的正负样本")
    return row


def _record_epoch(
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    target: torch.device,
    config: TrainingConfig,
    progress: _TrainingRun,
    row: HistoryRow,
) -> None:
    """更新选模进度，按原顺序保存最佳权重、恢复权重和训练记录。"""
    progress.history.append(row)
    val_loss = cast(float, row["val_loss"])
    score = cast(float, row["macro_auc"]) if row["macro_auc"] is not None else -val_loss
    improved = score > progress.best
    progress.best, progress.stale = (score, 0) if improved else (progress.best, progress.stale + 1)
    weights_only: Checkpoint = dict(
        format=CHECKPOINT_FORMAT,
        architecture=ARCHITECTURE,
        classes=list(CLASS_NAMES),
        imgsz=config.imgsz,
        epoch=cast(int, row["epoch"]),
        metrics=row,
        smoke_test=config.smoke_test,
        state_dict={key: value.detach().cpu() for key, value in network.state_dict().items()},
    )
    if improved:
        save_checkpoint(progress.path / "weights/best.pt", weights_only)
    state: TrainingState = dict(
        config=config.to_dict(),
        optimizer=optimizer.state_dict(),
        scaler=scaler.state_dict(),
        device_type=target.type,
        history=progress.history,
        best=progress.best,
        stale=progress.stale,
    )
    save_checkpoint(progress.path / "weights/last.pt", {**weights_only, "training_state": state})
    write_results(progress.path, progress.history)
    plot_history(progress.history, progress.path)


def train_multilabel(
    *, model: str | Path, data: str | Path = "datasets", name: str | None = None,
    epochs: int = 40, patience: int = 12, imgsz: int = 512,
    batch: int = 8, nbs: int = 128, workers: int = 6, device: str | int = 0,
    lr0: float = 1e-4, lrf: float = 0.01, warmup_epochs: int = 3,
    weight_decay: float = 5e-4, dropout: float = 0.2, brightness: float = 0.1,
    amp: bool = True, seed: int = 0, deterministic: bool = True,
    channels_last: bool = True, smoke_test: bool = False, resume: bool = False,
) -> Path:
    """从官方 CSV 训练十类模型，或从 last.pt 的下一轮恢复。"""
    config = TrainingConfig(
        model=model, data=data, name=name, epochs=epochs, patience=patience,
        imgsz=imgsz, batch=batch, nbs=nbs, workers=workers, device=device,
        lr0=lr0, lrf=lrf, warmup_epochs=warmup_epochs, weight_decay=weight_decay,
        dropout=dropout, brightness=brightness, amp=amp, seed=seed,
        deterministic=deterministic, channels_last=channels_last,
        smoke_test=smoke_test, resume=resume,
    )
    config, weights, checkpoint = _resolve_configuration(config)
    target = device_for(config.device)
    configure_randomness(config.seed, config.deterministic)
    if checkpoint is not None and checkpoint["training_state"]["device_type"] != target.type:
        raise ValueError("续训须使用与原训练相同的设备类型")
    print(f"正在准备 {ARCHITECTURE}，输入 {config.imgsz}，设备 {target}", flush=True)
    precision = "AMP FP16" if config.amp and target.type == "cuda" else "FP32"
    print(
        f"训练配置 name={config.name} | 精度={precision} | amp={config.amp} | "
        f"batch={config.batch} | nbs={config.nbs} | resume={checkpoint is not None}",
        flush=True,
    )
    print(
        f"学习率 lr0={config.lr0:.2e} | lrf={config.lrf} | "
        f"epochs={config.epochs} | warmup_epochs={config.warmup_epochs} | "
        f"patience={config.patience}",
        flush=True,
    )
    train, valid = _prepare_datasets(config)
    print(f"训练 {len(train)} 张，验证 {len(valid)} 张", flush=True)

    network = build_model(config.dropout, None if checkpoint is not None else weights)
    if checkpoint is not None:
        network.load_state_dict(checkpoint["state_dict"], strict=True)
    network.to(target)
    if config.channels_last:
        network.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.lr0, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and target.type == "cuda")
    progress = _initialize_run(config, weights, checkpoint, optimizer, scaler)
    print(
        f"产出目录 {progress.path} | 模型类型={next(network.parameters()).dtype} | "
        f"GradScaler={scaler.is_enabled()} | "
        f"TF32(matmul/cuDNN)={torch.backends.cuda.matmul.allow_tf32}/"
        f"{torch.backends.cudnn.allow_tf32}",
        flush=True,
    )
    val_loader = DataLoader(
        valid,
        batch_size=config.batch,
        num_workers=config.workers,
        pin_memory=target.type == "cuda",
        worker_init_fn=seed_worker,
    )
    for epoch in range(len(progress.history), config.epochs):
        train_loss = _train_epoch(network, train, optimizer, scaler, target, config, epoch)
        row = _validate_epoch(network, val_loader, valid, target, config, epoch, train_loss)
        _record_epoch(network, optimizer, scaler, target, config, progress, row)
        print(
            f"Epoch {epoch + 1} | BCE {row['train_loss']:.5f} | Val {row['val_loss']:.5f} | "
            f"AUC {row['macro_auc']} | Best {progress.best:.5f} | Stale {progress.stale}",
            flush=True,
        )
        if config.patience and progress.stale >= config.patience:
            break
    print(f"训练产出：{progress.path}", flush=True)
    return progress.path
