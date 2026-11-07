"""使用本项目多标签权重，按 Study 输出赛事 CSV。"""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from ncccu.multilabel import ROOT, ChestDataset, aggregate, device_for, evaluate, load_model, output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="train.py 生成的 best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "datasets")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gate", type=float, default=0.5)
    parser.add_argument("--name")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.batch < 1 or args.workers < 0 or not 0 <= args.gate <= 1:
        parser.error("batch>=1，workers>=0，gate 在 [0,1] 内")
    device = device_for(args.device)
    model, checkpoint = load_model(args.model, device)
    if checkpoint["smoke_test"] and not args.smoke_test:
        parser.error("冒烟权重仅可搭配 --smoke-test 验证流程")
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


if __name__ == "__main__":
    main()
