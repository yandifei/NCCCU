"""按检查生成十类概率、验证诊断和可追溯的赛事预测产物。"""

import json
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ncccu.configuration import PredictionConfig
from ncccu.data import ChestDataset
from ncccu.evaluation import aggregate, evaluate, save_study_probabilities, study_report
from ncccu.model import load_model
from ncccu.runtime import ROOT, configure_randomness, device_for, file_hash, output_dir, seed_worker
from ncccu.submission import write_submission


def predict_multilabel(
    *,
    model: str | Path,
    data: str | Path = "datasets",
    split: str = "test",
    device: str | int = 0,
    batch: int = 1,
    workers: int = 4,
    quantize: int = 32,
    gate: float = 0.5,
    channels_last: bool = True,
    augment: bool = False,
    seed: int = 0,
    deterministic: bool = True,
    name: str | None = None,
    smoke_test: bool = False,
) -> Path:
    """执行固定权重推理并返回产物目录，由入口决定是否继续打包。"""
    config = PredictionConfig(
        model=model, data=data, split=split, device=device, batch=batch,
        workers=workers, quantize=quantize, gate=gate, channels_last=channels_last,
        augment=augment, seed=seed, deterministic=deterministic,
        name=name, smoke_test=smoke_test,
    )
    config.validate()
    config = replace(
        config,
        model=(ROOT / config.model).resolve(),
        data=(ROOT / config.data).resolve(),
    )
    target = device_for(config.device)
    configure_randomness(config.seed, config.deterministic)
    model_path = Path(config.model)
    model_hash = file_hash(model_path)
    network, checkpoint = load_model(model_path, target)
    if checkpoint["smoke_test"] and not config.smoke_test:
        raise ValueError("冒烟权重须搭配 smoke_test=True")
    if config.channels_last:
        network.to(memory_format=torch.channels_last)
    dataset = ChestDataset(
        Path(config.data), config.split, checkpoint["imgsz"], config.smoke_test
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch,
        num_workers=config.workers,
        pin_memory=target.type == "cuda",
        worker_init_fn=seed_worker,
    )
    print(
        f"预测配置 模型={model_path} | 训练轮次={checkpoint['epoch']} | "
        f"输入={checkpoint['imgsz']} | 精度=FP{config.quantize} | "
        f"batch={config.batch} | Study 聚合=概率均值 | gate={config.gate}",
        flush=True,
    )
    probabilities, image_loss = evaluate(
        network, loader, target,
        quantize=config.quantize, channels_last=config.channels_last,
        augment=config.augment, verbose=True,
    )
    keys, scores, _ = aggregate(dataset, probabilities)
    report = study_report(dataset, scores, gate=config.gate) if config.split == "valid" else None
    # 防止训练进程在推理期间覆写 best.pt，导致 CSV 与归档模型不对应。
    if file_hash(model_path) != model_hash:
        raise ValueError("推理期间模型权重发生变化，请选择固定检查点后重试")
    run = output_dir("predict", config.name)
    metadata = {**config.to_dict(), "model_sha256": model_hash}
    (run / "args.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    if config.smoke_test:
        filename = "smoke_predictions.csv"
    elif config.split == "test":
        filename = "submission.csv"
    else:
        filename = "valid_predictions.csv"
    write_submission(run / filename, keys, scores, config.gate)
    save_study_probabilities(run / "study_probabilities.npz", dataset, scores)
    if report is not None:
        report.update(
            image_val_loss=image_loss,
            model_sha256=model_hash,
            checkpoint_epoch=checkpoint["epoch"],
        )
        (run / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(
            f"验证 Study BCE={report['study_val_loss']:.5f} | "
            f"Macro-AUC={report['macro_auc']} | Macro-F1={report['macro_f1']:.5f}",
            flush=True,
        )
    print(f"预测 {len(dataset)} 张影像 / {len(keys)} 次检查：{run / filename}", flush=True)
    return run
