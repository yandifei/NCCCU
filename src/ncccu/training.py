"""多标签训练循环与轮次级断点恢复。"""
import csv
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from ncccu.multilabel import (
    ROOT, CLASS_NAMES, ChestDataset, aggregate, device_for, evaluate,
    load_model, logits, metrics, output_dir, pretrained_model,
)


def atomic_save(checkpoint: dict, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def random_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_random_state(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("续训的 CUDA 设备数量与检查点不同")
        torch.cuda.set_rng_state_all(state["cuda"])


def write_history(history: list[dict], run: Path) -> None:
    temporary = run / "results.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(run / "results.csv")


def train_multilabel(
    *, model: str | Path = ROOT / "pretrain_model/yolo26n-cls.pt",
    data: str | Path = ROOT / "datasets", resume: bool = False,
    epochs: int = 100, imgsz: int = 224, batch: int = 16, workers: int = 0,
    device: str | int = 0, lr0: float = 0.001, lrf: float = 0.01,
    weight_decay: float = 0.0005, dropout: float = 0.0, patience: int = 20,
    seed: int = 0, amp: bool = True, name: str | None = None,
    smoke_test: bool = False,
) -> Path:
    args = SimpleNamespace(**locals())
    model_path = (ROOT / model).resolve()
    args.data = str((ROOT / data).resolve())
    args.model = str(model_path)
    args.device = str(device)
    device = device_for(device)
    checkpoint = None
    if resume:
        model, checkpoint = load_model(model_path, device)
        if "training_state" not in checkpoint:
            raise ValueError("该权重仅支持推理，续训需要新版 last.pt")
        # 训练超参数沿用原任务，防止恢复时默认值意外改变学习率和数据处理。
        for key, value in checkpoint["training_state"]["config"].items():
            if key not in {"model", "data", "device", "workers", "name", "resume"}:
                setattr(args, key, value)
        state = checkpoint["training_state"]
        if checkpoint["epoch"] >= args.epochs or (args.patience and state["stale"] >= args.patience):
            raise ValueError("该任务已完成总轮数或已触发早停，无待恢复轮次")
    if args.epochs < 1 or args.batch < 2 or args.imgsz < 64 or args.workers < 0:
        raise ValueError("epochs>=1，batch>=2，imgsz>=64，workers>=0")
    if args.lr0 <= 0 or not 0 <= args.lrf <= 1 or args.weight_decay < 0 or not 0 <= args.dropout < 1 or args.patience < 0:
        raise ValueError("学习率、正则化或 patience 参数无效")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = ChestDataset(Path(args.data), "train", args.imgsz, args.smoke_test)
    valid = ChestDataset(Path(args.data), "valid", args.imgsz, args.smoke_test)
    if not resume:
        model = pretrained_model(model_path, args.dropout).to(device)
    model.model[-1].drop.p = args.dropout
    model.requires_grad_(True)
    training = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    validation = DataLoader(valid, batch_size=args.batch, num_workers=args.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr0, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr0 * args.lrf)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, best, stale = 0, float("-inf"), 0
    history, best_checkpoint = [], None
    if resume:
        state = checkpoint["training_state"]
        if state["device_type"] != device.type:
            raise ValueError("续训请使用与原任务相同类型的设备，保持 AMP 和优化器状态一致")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best, stale = checkpoint["epoch"], state["best"], state["stale"]
        history, best_checkpoint = state["history"], state["best_checkpoint"]
        restore_random_state(state["random"])
        print(f"恢复第 {start_epoch} 轮完成后的状态，从第 {start_epoch + 1} 轮继续", flush=True)
    run = output_dir("train", args.name)
    (run / "weights").mkdir()
    (run / "args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    if resume:
        atomic_save(checkpoint, run / "weights/last.pt")
        atomic_save(best_checkpoint, run / "weights/best.pt")
        write_history(history, run)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total = 0.0
        for step, (images, targets, _) in enumerate(training):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = logits(model, images.to(device))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(output, targets.to(device))
            if not torch.isfinite(loss):
                raise ValueError("训练损失非有限数值")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(images)
            if step % 50 == 0:
                print(f"epoch={epoch+1} batch={step+1}/{len(training)} loss={loss.item():.5f}", flush=True)
        scheduler.step()
        probabilities, val_loss = evaluate(model, validation, device)
        _, scores, labels = aggregate(valid, probabilities)
        result = metrics(labels, scores)
        if result["auc_classes"] != 10 and not args.smoke_test:
            raise ValueError("验证集须包含全部10类的正负样本，无法计算完整 Macro-AUC")
        row = dict(epoch=epoch+1, train_loss=total/len(train), val_loss=val_loss, **result)
        history.append(row)
        print(row, flush=True)
        score = result["macro_auc"] if result["macro_auc"] is not None else -val_loss
        improved = score > best
        best = max(best, score)
        stale = 0 if improved else stale + 1
        checkpoint = {
            "format": "ncccu-multilabel-v1", "classes": list(CLASS_NAMES),
            "architecture": model.yaml,
            "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            "imgsz": args.imgsz, "epoch": epoch+1, "metrics": row,
            "smoke_test": args.smoke_test,
            "preprocessing": "rgb-pad114-bilinear-div255", "aggregation": "mean",
        }
        if improved:
            best_checkpoint = dict(checkpoint)
        # last.pt 内含历史最佳权重和指标，单个文件即可完整恢复。
        checkpoint["training_state"] = {
            "config": vars(args), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "best": best, "stale": stale, "history": history,
            "best_checkpoint": best_checkpoint, "random": random_state(),
            "device_type": device.type,
        }
        atomic_save(checkpoint, run / "weights/last.pt")
        atomic_save(best_checkpoint, run / "weights/best.pt")
        write_history(history, run)
        if args.patience and stale >= args.patience:
            break
    print(f"训练产出 {run}")
    return run
