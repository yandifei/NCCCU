"""使用本项目多标签权重，按 Study 输出赛事 CSV。"""
import csv
import json
from types import SimpleNamespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from ncccu.multilabel import ROOT, ChestDataset, aggregate, device_for, evaluate, load_model, output_dir


def predict_multilabel(
    *, model: str | Path, data: str | Path = ROOT / "datasets",
    split: str = "test", batch: int = 16, workers: int = 0,
    device: str | int = 0, gate: float = 0.5,
    name: str | None = None, smoke_test: bool = False,
) -> Path:
    args = SimpleNamespace(**locals())
    args.model, args.data = ROOT / model, ROOT / data
    if split not in {"train", "valid", "test"}:
        raise ValueError("split 必须为 train、valid 或 test")
    if args.batch < 1 or args.workers < 0 or not 0 <= args.gate <= 1:
        raise ValueError("batch>=1，workers>=0，gate 在 [0,1] 内")
    device = device_for(args.device)
    model, checkpoint = load_model(args.model, device)
    if checkpoint["smoke_test"] and not args.smoke_test:
        raise ValueError("冒烟权重仅可搭配 smoke_test=True 验证流程")
    dataset = ChestDataset(args.data, args.split, checkpoint["imgsz"], args.smoke_test)
    loader = DataLoader(dataset, batch_size=args.batch, num_workers=args.workers)
    probabilities, _ = evaluate(model, loader, device)
    keys, scores, _ = aggregate(dataset, probabilities)
    run = output_dir("predict", args.name)
    (run / "args.json").write_text(json.dumps(vars(args), default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    filename = "smoke_predictions.csv" if args.smoke_test else "submission.csv"
    with (run / filename).open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Subject_id", "Study_id", "Predict_class", "Probability", "Gate"])
        for (subject, study), values in zip(keys, scores):
            for index, probability in enumerate(values, start=1):
                writer.writerow([subject, study, index, float(probability), args.gate])
    print(f"预测 {len(dataset)} 张影像 / {len(keys)} 次检查，结果 {run / filename}")

    return run
