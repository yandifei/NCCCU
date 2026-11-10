"""管理运行路径、随机状态和训练曲线，所有产物统一写入 outputs。"""

import hashlib
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def device_for(value: str | int) -> torch.device:
    """将入口中的 CUDA 编号或设备名称转成 PyTorch 设备。"""
    value = str(value)
    return torch.device(f"cuda:{value}" if value.isdigit() else value)


def output_dir(kind: str, name: str | None) -> Path:
    """创建新的运行目录，拒绝覆盖已有产物或写到 outputs 之外。"""
    name = name or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if Path(name).name != name or name in {".", ".."} or any(c in name for c in "/\\:"):
        raise ValueError("name 必须是单层目录名")
    path = (ROOT / "outputs" / kind / name).resolve()
    if not path.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("全部产出必须位于 outputs 目录内")
    path.mkdir(parents=True, exist_ok=False)
    return path


def file_hash(path: Path) -> str:
    """计算文件摘要，用于追溯预测实际使用的模型权重。"""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def configure_randomness(seed: int, deterministic: bool) -> None:
    """同步 Python、NumPy 和 PyTorch 随机状态及确定性计算设置。"""
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
    """让工作进程的随机增强跟随 DataLoader 分配的随机种子。"""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def plot_history(history: list[dict[str, Any]], run: Path) -> None:
    """使用无窗口后端绘制训练与验证曲线。"""
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
