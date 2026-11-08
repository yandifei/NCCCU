"""验证集多标签 F1 阈值搜索；不修改模型概率或 AUC。"""
import numpy as np


def validate_arrays(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if labels.ndim != 2 or labels.shape != scores.shape or not labels.size:
        raise ValueError("标签与概率必须是同形状的非空二维数组")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("标签必须为 0/1，概率必须为 [0,1] 内有限值")
    return labels.astype(np.int64), scores


def f1_scores(labels: np.ndarray, scores: np.ndarray, thresholds: float | np.ndarray) -> np.ndarray:
    labels, scores = validate_arrays(labels, scores)
    gates = np.asarray(thresholds, dtype=np.float64)
    if not np.isfinite(gates).all() or ((gates < 0) | (gates > 1)).any():
        raise ValueError("阈值必须为 [0,1] 内有限值")
    predicted = scores >= gates
    denominator = labels.sum(0) + predicted.sum(0)
    return np.divide(2*(labels*predicted).sum(0), denominator,
                     out=np.zeros(labels.shape[1]), where=denominator != 0)


def optimize_thresholds(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels, scores = validate_arrays(labels, scores)
    positives = labels.sum(0)
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
        curves.append(2*true_positive / (target.sum() + len(score)-positions))
    curves = np.asarray(curves).T
    macro = curves.mean(1)
    # 多个候选达到同样最大值时，选更大的候选阈值。判定约定为 probability >= Gate。
    global_index = np.flatnonzero(macro == macro.max())[-1]
    class_indices = [np.flatnonzero(values == values.max())[-1] for values in curves.T]
    return dict(global_threshold=float(candidates[global_index]),
                class_thresholds=candidates[class_indices],
                global_macro_f1=float(macro[global_index]),
                class_macro_f1=float(curves.max(0).mean()),
                curve_thresholds=candidates, curve_macro_f1=macro)


def patient_folds(subjects: list[str], count: int = 5, seed: int = 0) -> np.ndarray:
    patients = np.unique(subjects)
    if not 2 <= count <= len(patients):
        raise ValueError("折数须在 2 与患者数量之间")
    np.random.default_rng(seed).shuffle(patients)
    assignment = {subject: index % count for index, subject in enumerate(patients)}
    return np.array([assignment[subject] for subject in subjects])
