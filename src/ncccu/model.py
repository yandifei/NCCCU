"""从公开 ImageNet 权重初始化 ConvNeXt-Small，或恢复项目检查点。"""

from pathlib import Path

import timm
import torch
from safetensors.torch import load_file

from ncccu.multilabel import CLASS_NAMES

ARCHITECTURE = "convnext_small.fb_in22k_ft_in1k"
CHECKPOINT_FORMAT = "ncccu-convnext-v1"


def build_model(dropout: float, pretrained: Path | None = None) -> torch.nn.Module:
    model = timm.create_model(ARCHITECTURE, pretrained=False, drop_rate=dropout)
    if pretrained is not None:
        model.load_state_dict(load_file(str(pretrained)), strict=True)
    model.reset_classifier(len(CLASS_NAMES))
    return model


def read_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("format") != CHECKPOINT_FORMAT
            or checkpoint.get("architecture") != ARCHITECTURE
            or checkpoint.get("classes") != list(CLASS_NAMES)):
        raise ValueError(f"需要本项目新版 ConvNeXt 检查点：{path}")
    return checkpoint


def load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = read_checkpoint(path)
    model = build_model(dropout=0.0)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint
