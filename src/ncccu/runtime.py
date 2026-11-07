"""确定性运行、配置校验和输出辅助。"""
import os
import random
from pathlib import Path

import numpy as np
import torch


def configure_randomness(seed: int, deterministic: bool) -> None:
    if deterministic:
        # 必须在首次创建 CUDA 上下文前配置 cuBLAS。
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def validate_fixed(options: dict, fixed: dict) -> None:
    for key, value in options.items():
        if key not in fixed:
            raise TypeError(f"未知参数 {key}")
        if value != fixed[key]:
            raise ValueError(f"当前多标签流程要求 {key}={fixed[key]!r}，收到 {value!r}")


def plot_history(history: list[dict], run: Path) -> None:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(10, 4))
    FigureCanvasAgg(figure)
    loss_axis, metric_axis = figure.subplots(1, 2)
    epochs = [row["epoch"] for row in history]
    for key in ("train_loss", "val_loss"):
        loss_axis.plot(epochs, [row[key] for row in history], label=key)
    for key in ("macro_auc", "macro_f1"):
        metric_axis.plot(epochs, [row[key] for row in history], label=key)
    for axis in (loss_axis, metric_axis):
        axis.set_xlabel("epoch")
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(run / "results.png")
