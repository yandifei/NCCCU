"""逐图推理、检查级概率聚合和多标签验证指标。"""

from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ncccu.data import CLASS_NAMES, NUM_CLASSES, ChestDataset, StudyKey
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
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("逐图概率须为 [0, 1] 内的有限数值")
    scores = np.stack([probabilities[indices].mean(axis=0) for indices in dataset.study_indices])
    labels = np.stack([dataset.targets[indices[0]].numpy() for indices in dataset.study_indices])
    return list(dataset.study_keys), scores, labels


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


def _group_report(labels: np.ndarray, scores: np.ndarray, gate: float) -> dict[str, Any]:
    """汇总同一检查子集；空组与缺少正负例的类别明确返回空指标。"""
    aucs = [binary_auc(target, score) for target, score in zip(labels.T, scores.T)] if len(labels) else [None] * NUM_CLASSES
    f1 = f1_scores(labels, scores, gate) if len(labels) else [None] * NUM_CLASSES
    valid_aucs = [value for value in aucs if value is not None]
    # 转为 float64 后截断，避免模型输出 0 或 1 时计算 log(0)。
    probabilities = np.clip(scores.astype(np.float64), 1e-15, 1 - 1e-15)
    loss = -(labels * np.log(probabilities) + (1 - labels) * np.log1p(-probabilities))
    per_class = []
    for index, name in enumerate(CLASS_NAMES):
        positives = int(labels[:, index].sum())
        per_class.append({
            "class_id": index + 1,
            "class_name": name,
            "positives": positives,
            "negatives": len(labels) - positives,
            "auc": aucs[index],
            "f1": float(f1[index]) if f1[index] is not None else None,
        })
    return {
        "studies": len(labels),
        "study_val_loss": float(loss.mean()) if len(labels) else None,
        "macro_auc": float(np.mean(valid_aucs)) if valid_aucs else None,
        "auc_classes": len(valid_aucs),
        "macro_f1": float(np.mean(f1)) if len(labels) else None,
        "per_class": per_class,
    }


def study_report(
    dataset: ChestDataset, scores: np.ndarray, *, gate: float = 0.5,
) -> dict[str, Any]:
    """按 Study 统计损失、逐类 AUC，并分别评估单视图与多视图检查。"""
    if dataset.split == "test":
        raise ValueError("官方测试集没有标签，不能生成验证指标")
    labels = np.stack([dataset.targets[indices[0]].numpy() for indices in dataset.study_indices])
    if scores.shape != (len(dataset.study_keys), NUM_CLASSES):
        raise ValueError("检查概率形状与 Study 清单不一致")
    labels, scores = validate_arrays(labels, scores)
    if not np.isfinite(gate) or not 0 <= gate <= 1:
        raise ValueError("gate 须为 [0, 1] 内的有限数值")
    counts = np.asarray([len(indices) for indices in dataset.study_indices])
    report = _group_report(labels, scores, gate)
    report.update(
        patients=len({subject for subject, _ in dataset.study_keys}),
        gate=gate,
        aggregation="mean_probability",
        loss_probability_clip=1e-15,
        view_groups={
            "single_view": _group_report(labels[counts == 1], scores[counts == 1], gate),
            "multi_view": _group_report(labels[counts > 1], scores[counts > 1], gate),
        },
    )
    return report


def save_study_probabilities(path: Path, dataset: ChestDataset, scores: np.ndarray) -> None:
    """原子保存检查概率及清单标识，允许用 allow_pickle=False 读取。"""
    if scores.shape != (len(dataset.study_keys), NUM_CLASSES):
        raise ValueError("检查概率形状与 Study 清单不一致")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("检查概率须为 [0, 1] 内的有限数值")
    arrays = {
        "probabilities": scores,
        "subject_ids": np.asarray([key[0] for key in dataset.study_keys], dtype=str),
        "study_ids": np.asarray([key[1] for key in dataset.study_keys], dtype=str),
        "view_counts": np.asarray([len(indices) for indices in dataset.study_indices]),
    }
    if dataset.split != "test":
        arrays["targets"] = np.stack([
            dataset.targets[indices[0]].numpy() for indices in dataset.study_indices
        ])
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    temporary.replace(path)


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
    seen = np.zeros(image_count, dtype=bool)
    has_labels = getattr(loader.dataset, "split", None) != "test"
    with tqdm(loader, desc="Validating", total=len(loader), disable=not verbose) as progress:
        for images, targets, indices in progress:
            targets = targets.to(device, non_blocking=device.type == "cuda")
            logits, batch_probabilities = predict_batch(
                model, images, device,
                quantize=quantize, channels_last=channels_last, augment=augment,
            )
            selected = indices.numpy()
            if np.any(selected < 0) or np.any(selected >= image_count):
                raise ValueError("推理清单下标超出范围")
            if seen[selected].any() or len(np.unique(selected)) != len(selected):
                raise ValueError("推理重复处理了同一影像")
            seen[selected] = True
            if has_labels:
                total_loss += torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction="sum"
                ).item()
            probabilities[selected] = batch_probabilities.cpu().numpy()
    if not seen.all() or image_count == 0:
        raise ValueError("推理清单未完整遍历")
    return probabilities, total_loss / (image_count * NUM_CLASSES)
