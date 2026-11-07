"""多标签训练循环与轮次级断点恢复。"""
import csv
import json
import random
import math
import time as clock
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics.nn.tasks import ClassificationModel
from ncccu.runtime import configure_randomness, seed_worker, validate_fixed, plot_history
from ncccu.train_options import TRAIN_FIXED, training_transform, make_optimizer, set_warmup

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
    *, model: str | Path = ROOT / "pretrain_model/yolo26x-cls.pt",
    data: str | Path = ROOT / "datasets", resume: bool = False,
    epochs: int = 100, imgsz: int = 224, batch: int = 16, workers: int = 0,
    device: str | int = 0, lr0: float = 0.001, lrf: float = 0.01,
    weight_decay: float = 0.0005, dropout: float = 0.0, patience: int = 20,
    seed: int = 0, amp: bool = True, name: str | None = None,
    smoke_test: bool = False,
    deterministic: bool = True, optimizer: str = "AdamW", momentum: float = 0.9,
    cos_lr: bool = True, warmup_epochs: float = 0.0,
    warmup_momentum: float = 0.8, warmup_bias_lr: float = 0.0,
    save: bool = True, save_period: int = -1, plots: bool = False,
    cache: bool | str = False, pretrained: bool = True, freeze: int = 0,
    time: float | None = None, nbs: int | None = None,
    cls: float = 1.0, cls_pw: float = 0.0, verbose: bool = True,
    channels_last: bool = False, project: str | Path | None = None,
    save_dir: str | Path | None = None, fliplr: float = 0.0,
    flipud: float = 0.0, hsv_h: float = 0.0, hsv_s: float = 0.0,
    hsv_v: float = 0.0, scale: float = 0.0, erasing: float = 0.0,
    auto_augment: str | None = None, **fixed_options,
) -> Path:
    args = SimpleNamespace(**locals())
    del args.fixed_options
    validate_fixed(fixed_options, TRAIN_FIXED)
    args.fixed_options = dict(fixed_options)
    model_path = (ROOT / model).resolve()
    args.data = str((ROOT / data).resolve())
    args.model = str(model_path)
    args.device = str(device)
    args.project = str(args.project) if args.project is not None else None
    args.save_dir = str(args.save_dir) if args.save_dir is not None else None
    device = device_for(device)
    configure_randomness(args.seed, args.deterministic)
    checkpoint = None
    if resume:
        model, checkpoint = load_model(model_path, device)
        if "training_state" not in checkpoint:
            raise ValueError("该权重仅支持推理，续训需要新版 last.pt")
        # 旧版检查点使用单组 AdamW、无预热和增强，恢复时保留其优化器布局。
        saved_config = checkpoint["training_state"]["config"]
        if "optimizer" not in saved_config:
            legacy = dict(
                optimizer="AdamW", momentum=0.9, cos_lr=True, warmup_epochs=0.0,
                warmup_momentum=0.8, warmup_bias_lr=0.0, freeze=0, nbs=None,
                cls=1.0, cls_pw=0.0, fliplr=0.0, flipud=0.0, hsv_h=0.0,
                hsv_s=0.0, hsv_v=0.0, scale=0.0, erasing=0.0, auto_augment=None,
                channels_last=False,
            )
            for key, value in legacy.items():
                setattr(args, key, value)
        # 训练超参数沿用原任务，防止恢复时默认值意外改变学习率和数据处理。
        for key, value in saved_config.items():
            if key not in {"model", "data", "device", "workers", "name", "resume", "project", "save_dir"}:
                setattr(args, key, value)
        state = checkpoint["training_state"]
        if checkpoint["epoch"] >= args.epochs or (args.patience and state["stale"] >= args.patience):
            raise ValueError("该任务已完成总轮数或已触发早停，无待恢复轮次")
    if args.epochs < 1 or args.batch < 2 or args.imgsz < 64 or args.workers < 0:
        raise ValueError("epochs>=1，batch>=2，imgsz>=64，workers>=0")
    if args.lr0 <= 0 or not 0 <= args.lrf <= 1 or args.weight_decay < 0 or not 0 <= args.dropout < 1 or args.patience < 0:
        raise ValueError("学习率、正则化或 patience 参数无效")
    validate_fixed(args.fixed_options, TRAIN_FIXED)
    for key in ("amp", "deterministic", "save", "plots", "pretrained", "cos_lr", "verbose", "channels_last"):
        if not isinstance(getattr(args, key), bool):
            raise TypeError(f"{key} 必须为 True 或 False")
    for key in ("lr0", "lrf", "weight_decay", "dropout", "momentum", "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "cls", "cls_pw"):
        if not math.isfinite(getattr(args, key)):
            raise ValueError(f"{key} 必须为有限数值")
    if not 0 <= args.momentum < 1 or not 0 <= args.warmup_momentum < 1 or args.warmup_epochs < 0 or args.warmup_bias_lr < 0:
        raise ValueError("动量必须在 [0,1) 内，预热参数不能为负")
    if args.save_period == 0 or args.save_period < -1 or (args.time is not None and args.time <= 0):
        raise ValueError("save_period 为 -1 或正整数，time 为 None 或正小时数")
    if args.cls <= 0 or args.cls_pw < 0 or (args.nbs is not None and args.nbs < 1):
        raise ValueError("cls>0，cls_pw>=0，nbs 为正整数")
    if not args.save and (args.resume or args.save_period > 0):
        raise ValueError("续训和周期保存要求 save=True")
    configure_randomness(args.seed, args.deterministic)
    train = ChestDataset(Path(args.data), "train", args.imgsz, args.smoke_test)
    valid = ChestDataset(Path(args.data), "valid", args.imgsz, args.smoke_test)
    if not resume:
        model = pretrained_model(model_path, args.dropout).to(device)
    if not resume and not args.pretrained:
        model = ClassificationModel(cfg=model.yaml, nc=10, verbose=False).to(device)
    model.model[-1].drop.p = args.dropout
    model.requires_grad_(True)
    if not isinstance(args.freeze, int) or not 0 <= args.freeze < len(model.model):
        raise ValueError("freeze 必须为保留至少一层可训练层的非负整数")
    for layer in model.model[:args.freeze]:
        layer.requires_grad_(False)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    train.transform = training_transform(args)
    training = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker)
    validation = DataLoader(valid, batch_size=args.batch, num_workers=args.workers, worker_init_fn=seed_worker)
    optimizer = make_optimizer(model, args)
    if args.cos_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr0 * args.lrf)
    else:
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=max(args.lrf, 1e-12), total_iters=args.epochs)
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
    run = output_dir("train", args.name, args.project, args.save_dir)
    train.prepare_cache(args.cache, run / "cache/train")
    valid.prepare_cache(args.cache, run / "cache/valid")
    positive_weight = None
    if args.cls_pw:
        positives = torch.stack(train.targets).sum(0)
        if (positives == 0).any():
            raise ValueError("类别加权要求训练数据每类至少一个正样本")
        positive_weight = ((len(train)-positives) / positives).pow(args.cls_pw).to(device)
    accumulation = max(1, math.ceil((args.nbs or args.batch) / args.batch))
    (run / "weights").mkdir()
    (run / "args.json").write_text(json.dumps(vars(args), default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    if resume:
        atomic_save(checkpoint, run / "weights/last.pt")
        atomic_save(best_checkpoint, run / "weights/best.pt")
        write_history(history, run)
    started = clock.monotonic()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for layer in model.model[:args.freeze]:
            layer.eval()
        for group in optimizer.param_groups:
            if "betas" in group:
                group["betas"] = (args.momentum, group["betas"][1])
            else:
                group["momentum"] = args.momentum
        total = 0.0
        for step, (images, targets, _) in enumerate(training):
            if step % accumulation == 0:
                optimizer.zero_grad(set_to_none=True)
            set_warmup(optimizer, args, epoch, step, len(training))
            if args.plots and epoch == start_epoch and step == 0:
                from torchvision.utils import save_image
                save_image(images[:16], run / "train_batch0.jpg", nrow=4)
            images = images.to(device)
            if args.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = logits(model, images.to(device))
                loss = args.cls * torch.nn.functional.binary_cross_entropy_with_logits(output, targets.to(device), pos_weight=positive_weight)
            if not torch.isfinite(loss):
                raise ValueError("训练损失非有限数值")
            chunk_start = (step // accumulation) * accumulation * args.batch
            chunk_samples = min(accumulation * args.batch, len(train)-chunk_start)
            scaler.scale(loss * len(images) / chunk_samples).backward()
            if (step+1) % accumulation == 0 or step+1 == len(training):
                scaler.step(optimizer)
                scaler.update()
            total += loss.item() * len(images)
            if args.verbose and step % 50 == 0:
                print(f"epoch={epoch+1} batch={step+1}/{len(training)} loss={loss.item():.5f}", flush=True)
        # 预热逐批覆盖 lr；轮次调度始终从该轮的目标学习率推进。
        factor = ((1+math.cos(math.pi*epoch/args.epochs))/2 if args.cos_lr else 1-epoch/args.epochs)
        for group in optimizer.param_groups:
            group["lr"] = args.lr0 * (args.lrf + (1-args.lrf)*factor)
        scheduler.step()
        probabilities, val_loss = evaluate(model, validation, device, channels_last=args.channels_last)
        _, scores, labels = aggregate(valid, probabilities)
        result = metrics(labels, scores)
        if result["auc_classes"] != 10 and not args.smoke_test:
            raise ValueError("验证集须包含全部10类的正负样本，无法计算完整 Macro-AUC")
        row = dict(epoch=epoch+1, train_loss=total/len(train), val_loss=val_loss, **result)
        history.append(row)
        if args.verbose:
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
        if args.save:
            atomic_save(checkpoint, run / "weights/last.pt")
            atomic_save(best_checkpoint, run / "weights/best.pt")
            if args.save_period > 0 and (epoch+1) % args.save_period == 0:
                atomic_save(checkpoint, run / f"weights/epoch{epoch+1}.pt")
        write_history(history, run)
        if args.plots:
            plot_history(history, run)
        if args.time is not None and clock.monotonic()-started >= args.time*3600:
            break
        if args.patience and stale >= args.patience:
            break
    print(f"训练产出 {run}")
    return run
