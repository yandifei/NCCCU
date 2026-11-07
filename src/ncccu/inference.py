"""使用本项目多标签权重，按 Study 输出赛事 CSV。"""
import csv
import json
from types import SimpleNamespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from ncccu.runtime import configure_randomness, seed_worker, validate_fixed
from ncccu.predict_options import (
    PREDICT_FIXED, PLOT_DEFAULTS, EXPORT_DEFAULTS, option_dict,
    validate_plot, validate_export, render_predictions, export_model,
)
from ncccu.multilabel import ROOT, ChestDataset, aggregate, device_for, evaluate, load_model, output_dir


def predict_multilabel(
    *, model: str | Path, data: str | Path = ROOT / "datasets",
    split: str = "test", batch: int = 16, workers: int = 0,
    device: str | int = 0, gate: float = 0.5,
    name: str | None = None, smoke_test: bool = False,
    source: str | Path | None = None, imgsz: int | None = None,
    quantize: int = 32, channels_last: bool = False, augment: bool = False,
    deterministic: bool = True, seed: int = 0, verbose: bool = True,
    project: str | Path | None = None, save_dir: str | Path | None = None,
    save: bool = False, show: bool = False, save_txt: bool = False,
    save_conf: bool = True, show_labels: bool = True, show_conf: bool = True,
    plot_options: dict | None = None, export_options: dict | None = None,
    **fixed_options,
) -> Path:
    args = SimpleNamespace(**locals())
    validate_fixed(fixed_options, PREDICT_FIXED)
    for key in ("channels_last", "augment", "deterministic", "verbose", "save", "show", "save_txt", "save_conf", "show_labels", "show_conf"):
        if not isinstance(getattr(args, key), bool):
            raise TypeError(f"{key} 必须为 True 或 False")
    args.plot_options = option_dict(plot_options, PLOT_DEFAULTS)
    args.export_options = option_dict(export_options, EXPORT_DEFAULTS)
    validate_plot(args.plot_options)
    validate_export(args.export_options)
    configure_randomness(seed, deterministic)
    args.model, args.data = ROOT / model, ROOT / data
    if split not in {"train", "valid", "test"}:
        raise ValueError("split 必须为 train、valid 或 test")
    if args.batch < 1 or args.workers < 0 or not 0 <= args.gate <= 1:
        raise ValueError("batch>=1，workers>=0，gate 在 [0,1] 内")
    if quantize not in (16, 32):
        raise ValueError("quantize 必须为 16 或 32")
    if source is not None and (ROOT / source).resolve() != (args.data / f"{split}.csv").resolve():
        raise ValueError("source 应为 data/split 对应的官方 CSV")
    device = device_for(args.device)
    if quantize == 16 and device.type != "cuda":
        raise ValueError("FP16 推理需要 CUDA")
    model, checkpoint = load_model(args.model, device)
    if checkpoint["smoke_test"] and not args.smoke_test:
        raise ValueError("冒烟权重仅可搭配 smoke_test=True 验证流程")
    if imgsz is not None and imgsz != checkpoint["imgsz"]:
        raise ValueError("imgsz 必须与训练检查点一致")
    if channels_last:
        model.to(memory_format=torch.channels_last)
    dataset = ChestDataset(args.data, args.split, checkpoint["imgsz"], args.smoke_test)
    loader = DataLoader(dataset, batch_size=args.batch, num_workers=args.workers, worker_init_fn=seed_worker)
    probabilities, _ = evaluate(model, loader, device, quantize=quantize, channels_last=channels_last, augment=augment)
    keys, scores, _ = aggregate(dataset, probabilities)
    run = output_dir("predict", args.name, args.project, args.save_dir)
    (run / "args.json").write_text(json.dumps(vars(args), default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    filename = "smoke_predictions.csv" if args.smoke_test else "submission.csv"
    with (run / filename).open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Subject_id", "Study_id", "Predict_class", "Probability", "Gate"])
        for (subject, study), values in zip(keys, scores):
            for index, probability in enumerate(values, start=1):
                writer.writerow([subject, study, index, float(probability), args.gate])
    render_predictions(dataset, probabilities, run, args)
    export_model(model, run, checkpoint["imgsz"], device, args.export_options)
    if args.verbose:
        print(f"预测 {len(dataset)} 张影像 / {len(keys)} 次检查，结果 {run / filename}")

    return run
