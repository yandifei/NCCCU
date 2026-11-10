"""统一预测脚本和赛事测速脚本的提交 CSV 写入格式。"""

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ncccu.data import NUM_CLASSES, StudyKey

SUBMISSION_FIELDS = ("Subject_id", "Study_id", "Predict_class", "Probability", "Gate")


def write_submission(
    path: Path,
    keys: Sequence[StudyKey],
    scores: np.ndarray,
    gate: float,
    *,
    header: bool = True,
) -> None:
    """每个 Study 按官方类别 ID 1–10 写出独立概率，保留原有行顺序。"""
    if scores.shape != (len(keys), NUM_CLASSES):
        raise ValueError("提交概率须为每次检查对应的十类概率")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("提交概率须为 [0, 1] 内有限值")
    if not 0 <= gate <= 1:
        raise ValueError("Gate 须为 [0, 1] 内有限值")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        if header:
            writer.writerow(SUBMISSION_FIELDS)
        for (subject, study), values in zip(keys, scores):
            writer.writerows(
                (subject, study, class_id, float(value), gate)
                for class_id, value in enumerate(values, start=1)
            )
