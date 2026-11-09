"""固定权重推理，按 Study 取概率均值并输出比赛 CSV。"""

import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ncccu.model import load_model
from ncccu.multilabel import ROOT, ChestDataset, aggregate, device_for, evaluate, output_dir
from ncccu.runtime import configure_randomness


def predict_multilabel(
    *, model: str | Path, data: str | Path = "datasets", split: str = "test",
    device: str | int = 0, batch: int = 1, workers: int = 4, quantize: int = 32,
    gate: float = 0.5, channels_last: bool = True, augment: bool = False,
    seed: int = 0, deterministic: bool = True, name: str | None = None, smoke_test: bool = False,
) -> Path:
    config = dict(locals())
    if split not in {"test", "valid"}:
        raise ValueError("split 须为 test 或 valid")
    if not 0 <= gate <= 1 or batch < 1 or workers < 0:
        raise ValueError("gate 须在 [0,1] 内，batch 须为正整数，workers 须为非负整数")
    target = device_for(device)
    if quantize not in {16, 32} or (quantize == 16 and target.type != "cuda"):
        raise ValueError("推理精度须为 FP32 或 CUDA FP16")
    config.update(model=str((ROOT / model).resolve()), data=str((ROOT / data).resolve()))
    configure_randomness(seed, deterministic)
    network, checkpoint = load_model(Path(config["model"]), target)
    if checkpoint["smoke_test"] and not smoke_test:
        raise ValueError("冒烟权重须搭配 smoke_test=True")
    if channels_last:
        network.to(memory_format=torch.channels_last)
    dataset = ChestDataset(Path(config["data"]), split, checkpoint["imgsz"], smoke_test)
    loader = DataLoader(dataset, batch_size=batch, num_workers=workers, pin_memory=target.type == "cuda")
    probabilities, _ = evaluate(network, loader, target, quantize=quantize, channels_last=channels_last,
                                augment=augment, verbose=True)
    keys, scores, _ = aggregate(dataset, probabilities)
    run = output_dir("predict", name)
    (run / "args.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    filename = "smoke_predictions.csv" if smoke_test else ("submission.csv" if split == "test" else "valid_predictions.csv")
    with (run / filename).open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("Subject_id", "Study_id", "Predict_class", "Probability", "Gate"))
        for (subject, study), values in zip(keys, scores):
            writer.writerows((subject, study, index, float(value), gate)
                             for index, value in enumerate(values, start=1))
    print(f"预测 {len(dataset)} 张影像 / {len(keys)} 次检查：{run / filename}", flush=True)
    return run
