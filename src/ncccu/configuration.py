"""训练与推理的显式配置，保持 JSON 和旧检查点的字段格式。"""

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

from ncccu.runtime import device_for


def _validate_types(
    *,
    integers: Mapping[str, object],
    booleans: Mapping[str, object],
    numbers: Mapping[str, object],
) -> None:
    """先核对标量类型，避免布尔值冒充整数或把配置错误推迟到 PyTorch。"""
    for name, value in integers.items():
        if type(value) is not int:
            raise ValueError(f"{name} 须为整数")
    for name, value in booleans.items():
        if type(value) is not bool:
            raise ValueError(f"{name} 须为布尔值")
    for name, value in numbers.items():
        if type(value) not in (int, float):
            raise ValueError(f"{name} 须为数值")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """保存一次训练的参数；路径由入口相对于仓库根目录解析。"""

    model: str | Path
    data: str | Path = "datasets"
    name: str | None = None
    epochs: int = 40
    patience: int = 12
    imgsz: int = 512
    batch: int = 8
    nbs: int = 128
    workers: int = 6
    device: str | int = 0
    lr0: float = 1e-4
    lrf: float = 0.01
    warmup_epochs: int = 3
    weight_decay: float = 5e-4
    dropout: float = 0.2
    brightness: float = 0.1
    amp: bool = True
    seed: int = 0
    deterministic: bool = True
    channels_last: bool = True
    smoke_test: bool = False
    resume: bool = False
    objective: str = "image"

    def validate(self) -> None:
        """检查训练参数范围，不访问数据集或加载模型。"""
        _validate_types(
            integers=dict(
                epochs=self.epochs, patience=self.patience, imgsz=self.imgsz,
                batch=self.batch, nbs=self.nbs, workers=self.workers,
                warmup_epochs=self.warmup_epochs, seed=self.seed,
            ),
            booleans=dict(
                amp=self.amp, deterministic=self.deterministic,
                channels_last=self.channels_last, smoke_test=self.smoke_test, resume=self.resume,
            ),
            numbers=dict(
                lr0=self.lr0, lrf=self.lrf, weight_decay=self.weight_decay,
                dropout=self.dropout, brightness=self.brightness,
            ),
        )
        if min(self.batch, self.nbs, self.epochs) < 1 or self.workers < 0:
            raise ValueError("batch、nbs、epochs 须为正整数，workers 须为非负整数")
        if not isinstance(self.objective, str) or self.objective not in {
            "image", "study_weighted", "study_mean"
        }:
            raise ValueError("objective 须为 image、study_weighted 或 study_mean")
        if self.imgsz < 32 or self.imgsz % 32:
            raise ValueError("imgsz 须为 32 的正整数倍")
        if not 0 <= self.seed < 2**32 or self.seed + self.epochs - 1 >= 2**32:
            raise ValueError("seed 及每轮派生种子须在 [0, 2**32) 内")
        if (
            not 0 <= self.brightness <= 1
            or not 0 <= self.dropout < 1
            or not 0 < self.lr0 < math.inf
            or not 0 < self.lrf <= 1
            or not 0 <= self.weight_decay < math.inf
            or self.patience < 0
            or not 0 <= self.warmup_epochs < self.epochs
        ):
            raise ValueError("学习率、预热、早停或增强参数超出范围")

    def to_dict(self) -> dict[str, Any]:
        """输出普通字典，避免将 dataclass 对象写入检查点。"""
        values = asdict(self)
        values.update(model=str(self.model), data=str(self.data))
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Self:
        """读取既有配置字典；缺省参数沿用入口默认值。"""
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError("训练配置字段不完整或包含未知字段") from error


@dataclass(frozen=True, slots=True)
class PredictionConfig:
    """保存一次预测的参数；输入尺寸仍由训练检查点决定。"""

    model: str | Path
    data: str | Path = "datasets"
    split: str = "test"
    device: str | int = 0
    batch: int = 1
    workers: int = 4
    quantize: int = 32
    gate: float = 0.5
    channels_last: bool = True
    augment: bool = False
    seed: int = 0
    deterministic: bool = True
    name: str | None = None
    smoke_test: bool = False

    def validate(self) -> None:
        """检查预测范围以及半精度所需的 CUDA 设备。"""
        _validate_types(
            integers=dict(batch=self.batch, workers=self.workers, quantize=self.quantize, seed=self.seed),
            booleans=dict(
                channels_last=self.channels_last, augment=self.augment,
                deterministic=self.deterministic, smoke_test=self.smoke_test,
            ),
            numbers=dict(gate=self.gate),
        )
        if self.split not in {"test", "valid"}:
            raise ValueError("split 须为 test 或 valid")
        if not 0 <= self.gate <= 1 or self.batch < 1 or self.workers < 0:
            raise ValueError("gate 须在 [0,1] 内，batch 须为正整数，workers 须为非负整数")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed 须在 [0, 2**32) 内")
        target = device_for(self.device)
        if self.quantize not in {16, 32} or (self.quantize == 16 and target.type != "cuda"):
            raise ValueError("推理精度须为 FP32 或 CUDA FP16")

    def to_dict(self) -> dict[str, Any]:
        """输出与既有 args.json 相同的普通字段。"""
        values = asdict(self)
        values.update(model=str(self.model), data=str(self.data))
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Self:
        """从 JSON 参数或调用方保存的字典恢复配置。"""
        # 权重摘要属于产物溯源信息，不是推理参数。
        values = dict(values)
        values.pop("model_sha256", None)
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError("预测配置字段不完整或包含未知字段") from error
