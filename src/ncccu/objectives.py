"""按影像或检查计算多标签 BCE，统一训练目标的归一化方式。"""

import math

import torch
from torch.nn import functional as F


def image_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """返回图像平均损失；Study 权重由整个训练划分计算并固定。"""
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=1)
    if weights is not None:
        losses = losses * weights
    return losses.mean()


def study_mean_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """对完整检查的概率均值计算 BCE，极端 logits 下保持有限梯度。"""
    if logits.ndim != 2 or len(logits) == 0 or target.shape != logits.shape[1:]:
        raise ValueError("Study logits 须为非空的视图×类别矩阵，标签须与类别数一致")
    # 使用 log(mean(sigmoid))，避免先 sigmoid 后裁剪导致梯度饱和。
    logits = logits.float()
    normalization = math.log(len(logits))
    log_positive = torch.logsumexp(F.logsigmoid(logits), dim=0) - normalization
    log_negative = torch.logsumexp(F.logsigmoid(-logits), dim=0) - normalization
    return -(target * log_positive + (1 - target) * log_negative).mean()
