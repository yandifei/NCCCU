"""验证集多标签 F1 阈值搜索；不修改模型概率或 AUC。"""

from typing import TypedDict

import numpy as np


class ThresholdSearchResult(TypedDict):
    """共同阈值、逐类阈值及用于绘图的完整候选曲线。"""

    global_threshold: float
    class_thresholds: np.ndarray
    global_macro_f1: float
    class_macro_f1: float
    curve_thresholds: np.ndarray
    curve_macro_f1: np.ndarray


def validate_arrays(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """统一标签和概率的形状、数值范围及计算精度。"""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 2 or labels.shape != scores.shape or not labels.size:
        raise ValueError("标签与概率必须是同形状的非空二维数组")
    if (
        not np.isin(labels, [0, 1]).all()
        or not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise ValueError("标签必须为 0/1，概率必须为 [0,1] 内有限值")
    return labels.astype(np.int64), scores


def f1_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    thresholds: float | np.ndarray,
) -> np.ndarray:
    """逐类计算 F1，接受共同、逐类和逐样本阈值。

    阈值按 NumPy 规则广播到 (样本数, 类别数)；逐样本共同阈值需为
    (样本数, 1)，逐样本逐类阈值需为 (样本数, 类别数)。
    """
    labels, scores = validate_arrays(labels, scores)
    gates = np.asarray(thresholds, dtype=np.float64)
    if not np.isfinite(gates).all() or ((gates < 0) | (gates > 1)).any():
        raise ValueError("阈值必须为 [0,1] 内有限值")
    try:
        gates = np.broadcast_to(gates, scores.shape)
    except ValueError as error:
        raise ValueError("阈值形状必须能广播到 (样本数, 类别数)") from error
    predicted = scores >= gates
    true_positives = (labels * predicted).sum(axis=0)
    denominator = labels.sum(axis=0) + predicted.sum(axis=0)
    return np.divide(
        2 * true_positives,
        denominator,
        out=np.zeros(labels.shape[1]),
        where=denominator != 0,
    )


def optimize_thresholds(labels: np.ndarray, scores: np.ndarray) -> ThresholdSearchResult:
    """穷尽概率断点，以排序与累计正例数计算精确的最优阈值。"""
    labels, scores = validate_arrays(labels, scores)
    positives = labels.sum(axis=0)
    if ((positives == 0) | (positives == len(labels))).any():
        raise ValueError("每类必须同时包含正负样本，才能估计阈值")
    # 概率相等的样本必须一同判定；枚举全部概率断点，避免固定网格漏掉最优值。
    candidates = np.unique(np.r_[0.0, scores.ravel(), 1.0])
    curves = []
    for target, score in zip(labels.T, scores.T):
        order = np.argsort(score, kind="stable")
        prefix = np.r_[0, np.cumsum(target[order])]
        positions = np.searchsorted(score[order], candidates, side="left")
        true_positive = target.sum() - prefix[positions]
        curves.append(2 * true_positive / (target.sum() + len(score) - positions))
    curves = np.asarray(curves).T
    macro = curves.mean(axis=1)
    # 多个候选达到同样最大值时，选更大的候选阈值。判定约定为 probability >= Gate。
    global_index = np.flatnonzero(macro == macro.max())[-1]
    class_indices = [np.flatnonzero(values == values.max())[-1] for values in curves.T]
    return {
        "global_threshold": float(candidates[global_index]),
        "class_thresholds": candidates[class_indices],
        "global_macro_f1": float(macro[global_index]),
        "class_macro_f1": float(curves.max(axis=0).mean()),
        "curve_thresholds": candidates,
        "curve_macro_f1": macro,
    }


def patient_folds(subjects: list[str], count: int = 5, seed: int = 0) -> np.ndarray:
    """按患者分配可复现的折号，同一患者的所有检查保留在同一折。"""
    patients = np.unique(subjects)
    if not 2 <= count <= len(patients):
        raise ValueError("折数须在 2 与患者数量之间")
    np.random.default_rng(seed).shuffle(patients)
    assignment = {subject: index % count for index, subject in enumerate(patients)}
    return np.array([assignment[subject] for subject in subjects])
