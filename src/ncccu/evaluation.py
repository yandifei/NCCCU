"""逐图推理、检查级概率聚合和多标签验证指标。"""

from typing import TypedDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ncccu.data import NUM_CLASSES, ChestDataset, StudyKey
from ncccu.thresholds import f1_scores, validate_arrays


class ValidationMetrics(TypedDict):
    macro_auc: float | None
    auc_classes: int
    macro_f1: float


def aggregate(
    dataset: ChestDataset,
    probabilities: np.ndarray,
) -> tuple[list[StudyKey], np.ndarray, np.ndarray]:
    """按官方清单顺序，对每次检查的所有视图取等权概率均值。"""
    if probabilities.shape != (len(dataset), NUM_CLASSES):
        raise ValueError("逐图概率形状须与清单影像数量和类别数量一致")
    groups: dict[StudyKey, list[int]] = {}
    for index, row in enumerate(dataset.rows):
        groups.setdefault((row["Subject_id"], row["Study_id"]), []).append(index)
    scores = np.stack([probabilities[indices].mean(axis=0) for indices in groups.values()])
    labels = np.stack([dataset.targets[indices[0]].numpy() for indices in groups.values()])
    return list(groups), scores, labels


def binary_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    """用平均秩计算单类 AUC；缺少正样本或负样本时返回 None。"""
    if target.ndim != 1 or score.ndim != 1:
        raise ValueError("单类标签与概率须为一维数组")
    labels, scores = validate_arrays(target[:, None], score[:, None])
    target, score = labels[:, 0], scores[:, 0]
    positives = target.sum()
    negatives = len(target) - positives
    if positives == 0 or negatives == 0:
        return None
    # 相同概率使用平均秩，保证同分正负样本对的 AUC 贡献为 0.5。
    order = np.argsort(score, kind="stable")
    _, starts, counts = np.unique(score[order], return_index=True, return_counts=True)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.repeat(starts + (counts + 1) / 2, counts)
    return float(
        (ranks[target == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def metrics(labels: np.ndarray, scores: np.ndarray) -> ValidationMetrics:
    """计算 Macro-AUC 和阈值为 0.5 的 Macro-F1；单一标签类别不计 AUC。"""
    labels, scores = validate_arrays(labels, scores)
    aucs = [
        auc for target, score in zip(labels.T, scores.T)
        if (auc := binary_auc(target, score)) is not None
    ]
    return {
        "macro_auc": float(np.mean(aucs)) if aucs else None,
        "auc_classes": len(aucs),
        "macro_f1": float(f1_scores(labels, scores, 0.5).mean()),
    }


def predict_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    *,
    quantize: int,
    channels_last: bool,
    augment: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回原图 logits 和独立类别概率；可选水平翻转概率平均。"""
    images = images.to(device, non_blocking=device.type == "cuda")
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    with torch.autocast(device_type=device.type, enabled=quantize == 16):
        logits = model(images).float()
        probabilities = logits.sigmoid()
        if augment:
            flipped = model(images.flip(-1)).float().sigmoid()
            probabilities = (probabilities + flipped) / 2
    if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():
        raise ValueError("预测出现非有限数值")
    return logits, probabilities


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    quantize: int = 32,
    channels_last: bool = False,
    augment: bool = False,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    """按清单下标收集逐图概率，返回未聚合的 BCE 平均损失。"""
    model.eval()
    image_count = len(loader.dataset)
    probabilities = np.empty((image_count, NUM_CLASSES), dtype=np.float32)
    total_loss = 0.0
    with tqdm(loader, desc="Validating", total=len(loader), disable=not verbose) as progress:
        for images, targets, indices in progress:
            targets = targets.to(device, non_blocking=device.type == "cuda")
            logits, batch_probabilities = predict_batch(
                model, images, device,
                quantize=quantize, channels_last=channels_last, augment=augment,
            )
            total_loss += torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="sum"
            ).item()
            probabilities[indices.numpy()] = batch_probabilities.cpu().numpy()
    return probabilities, total_loss / (image_count * NUM_CLASSES)
