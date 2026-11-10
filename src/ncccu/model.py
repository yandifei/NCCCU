"""从公开 ImageNet 权重初始化 ConvNeXt-Small，或恢复项目检查点。"""

from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import timm
import torch
from safetensors.torch import load_file

from ncccu.data import CLASS_NAMES

ARCHITECTURE = "convnext_small.fb_in22k_ft_in1k"
CHECKPOINT_FORMAT = "ncccu-convnext-v1"


type HistoryRow = dict[str, int | float | None]


class TrainingState(TypedDict):
    """轮次边界恢复所需的状态；序列化时保持普通字典。"""

    config: dict[str, Any]
    optimizer: dict[str, Any]
    scaler: dict[str, Any]
    device_type: str
    history: list[HistoryRow]
    best: float
    stale: int


class Checkpoint(TypedDict):
    """best.pt 与 last.pt 共享的格式；后者额外保存训练状态。"""

    format: str
    architecture: str
    classes: list[str]
    imgsz: int
    epoch: int
    metrics: HistoryRow
    smoke_test: bool
    state_dict: dict[str, torch.Tensor]
    training_state: NotRequired[TrainingState]


def build_model(dropout: float, pretrained: Path | None = None) -> torch.nn.Module:
    """先载入完整 ImageNet 权重，再为比赛创建十类独立输出头。"""
    model = timm.create_model(ARCHITECTURE, pretrained=False, drop_rate=dropout)
    if pretrained is not None:
        model.load_state_dict(load_file(str(pretrained)), strict=True)
    model.reset_classifier(len(CLASS_NAMES))
    return model


def read_checkpoint(path: Path) -> Checkpoint:
    """以安全权重模式读取项目检查点，并尽早报告结构损坏。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint, dict)
            or checkpoint.get("format") != CHECKPOINT_FORMAT
            or checkpoint.get("architecture") != ARCHITECTURE
            or checkpoint.get("classes") != list(CLASS_NAMES)):
        raise ValueError(f"需要本项目新版 ConvNeXt 检查点：{path}")
    required = {"imgsz", "epoch", "metrics", "smoke_test", "state_dict"}
    if not required.issubset(checkpoint):
        raise ValueError(f"检查点缺少必需字段：{path}")
    if (
        not isinstance(checkpoint["imgsz"], int)
        or checkpoint["imgsz"] < 32
        or checkpoint["imgsz"] % 32
        or not isinstance(checkpoint["epoch"], int)
        or checkpoint["epoch"] < 1
        or not isinstance(checkpoint["metrics"], dict)
        or not isinstance(checkpoint["smoke_test"], bool)
        or not isinstance(checkpoint["state_dict"], dict)
        or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in checkpoint["state_dict"].items()
        )
    ):
        raise ValueError(f"检查点字段格式错误：{path}")
    if "training_state" in checkpoint:
        _validate_training_state(checkpoint["training_state"], path)
    return cast(Checkpoint, checkpoint)


def _validate_training_state(state: object, path: Path) -> None:
    """验证恢复所需的容器，具体参数范围由训练配置负责。"""
    required = {"config", "optimizer", "scaler", "device_type", "history", "best", "stale"}
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError(f"检查点缺少完整训练状态：{path}")
    if (
        not all(isinstance(state[key], dict) for key in ("config", "optimizer", "scaler"))
        or not isinstance(state["device_type"], str)
        or not isinstance(state["history"], list)
        or not all(isinstance(row, dict) for row in state["history"])
        or not isinstance(state["best"], (int, float))
        or not isinstance(state["stale"], int)
        or state["stale"] < 0
    ):
        raise ValueError(f"检查点训练状态格式错误：{path}")


def load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, Checkpoint]:
    """严格恢复十类模型，并切换到推理模式。"""
    checkpoint = read_checkpoint(path)
    model = build_model(dropout=0.0)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint
