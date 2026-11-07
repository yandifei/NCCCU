"""从官方 CSV 训练 YOLO 多标签胸片分类模型。"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from ncccu.multilabel import (
    ROOT, CLASS_NAMES, ChestDataset, aggregate, device_for, evaluate,
    logits, metrics, output_dir, pretrained_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default=str(ROOT / "pretrain_model/yolo26n-cls.pt"))
    parser.add_argument("--data", type=str, default=str(ROOT / "datasets"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--name")
    parser.add_argument("--smoke-test", action="store_true", help="前16次检查，仅用于流程验证")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 2 or args.imgsz < 64 or args.workers < 0:
        parser.error("epochs>=1，batch>=2，imgsz>=64，workers>=0")
    if args.lr0 <= 0 or not 0 <= args.lrf <= 1 or args.weight_decay < 0 or not 0 <= args.dropout < 1 or args.patience < 0:
        parser.error("学习率、正则化或 patience 参数无效")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = device_for(args.device)
    train = ChestDataset(Path(args.data), "train", args.imgsz, args.smoke_test)
    valid = ChestDataset(Path(args.data), "valid", args.imgsz, args.smoke_test)
    model = pretrained_model(Path(args.model), args.dropout).to(device)
    training = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    validation = DataLoader(valid, batch_size=args.batch, num_workers=args.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr0, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr0 * args.lrf)
    amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    run = output_dir("train", args.name)
    (run / "weights").mkdir()
    (run / "args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    best, stale = float("-inf"), 0
    with (run / "results.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss", "macro_auc", "auc_classes", "macro_f1"])
        writer.writeheader()
        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for step, (images, targets, _) in enumerate(training):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
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
            writer.writerow(row)
            file.flush()
            print(row, flush=True)
            score = result["macro_auc"] if result["macro_auc"] is not None else -val_loss
            improved = score > best
            best = max(best, score)
            stale = 0 if improved else stale + 1
            checkpoint = {
                "format": "ncccu-multilabel-v1", "classes": list(CLASS_NAMES),
                "architecture": model.yaml, "state_dict": model.state_dict(),
                "imgsz": args.imgsz, "epoch": epoch+1, "metrics": row,
                "smoke_test": args.smoke_test,
                "preprocessing": "rgb-pad114-bilinear-div255", "aggregation": "mean",
            }
            for filename in (["last.pt", "best.pt"] if improved else ["last.pt"]):
                path = run / "weights" / filename
                temporary = path.with_suffix(".tmp")
                torch.save(checkpoint, temporary)
                temporary.replace(path)
            if args.patience and stale >= args.patience:
                break
    print(f"训练产出 {run}")


if __name__ == "__main__":
    main()
