"""读取官方影像清单，构造多标签目标并执行统一的图像预处理。"""

import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

CLASS_NAMES = (
    "Enlarged Cardiomediastinum",
    "Pneumothorax",
    "Consolidation",
    "Pneumonia",
    "Edema",
    "Cardiomegaly",
    "Atelectasis",
    "Lung Opacity",
    "Pleural Effusion",
    "No Finding",
)
NUM_CLASSES = len(CLASS_NAMES)
ID_FIELDS = ("Subject_id", "Study_id", "Dicom_id")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SMOKE_STUDIES = 16

type ImageRow = dict[str, str]
type StudyKey = tuple[str, str]
type ImageTransform = Callable[[Image.Image], torch.Tensor]


def read_chest_image(path: Path, imgsz: int) -> np.ndarray:
    """保留整张影像的宽高比，缩放并填充为方形 RGB 图像。"""
    with Image.open(path) as source:
        image = ImageOps.pad(
            source.convert("RGB"),
            (imgsz, imgsz),
            method=Image.Resampling.BILINEAR,
            color=(114, 114, 114),
        )
        return np.asarray(image).copy()


def normalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """将 [0, 1] 的 CHW 张量按 ImageNet 统计量归一化。"""
    mean = tensor.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = tensor.new_tensor(IMAGENET_STD)[:, None, None]
    return (tensor - mean) / std


def image_tensor(
    path: Path,
    imgsz: int,
    transform: ImageTransform | None = None,
) -> torch.Tensor:
    """训练、预测和赛事测速共用同一条像素预处理路径。"""
    image = read_chest_image(path, imgsz)
    if transform is None:
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255
    else:
        # 增强作用于未归一化的图像；只有训练集会传入 transform。
        tensor = transform(Image.fromarray(image))
    return normalize_image(tensor)


def _read_rows(root: Path, split: str) -> list[ImageRow]:
    path = root / f"{split}.csv"
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        required = set(ID_FIELDS)
        if split != "test":
            required.add("Predict_class")
        if not required.issubset(fields) or len(fields) != len(set(fields)):
            raise ValueError(f"{path.name} 缺少必需列或包含重复列名")
        rows = []
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path.name} 第 {line} 行列数不符")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path.name} 为空")
    return rows


def _read_target(row: ImageRow, split: str) -> torch.Tensor:
    target = torch.zeros(NUM_CLASSES)
    if split == "test":
        return target
    try:
        class_ids = [int(value) for value in row["Predict_class"].split(";")]
    except ValueError as error:
        raise ValueError(f"标签为空或格式错误：{row['Dicom_id']}") from error
    if len(set(class_ids)) != len(class_ids) or not set(class_ids).issubset(
        range(1, NUM_CLASSES + 1)
    ):
        raise ValueError(f"无效标签 {row['Dicom_id']}：{class_ids}")
    # 官方类别从 1 开始，张量下标从 0 开始；顺序由 CLASS_NAMES 固定。
    target[[class_id - 1 for class_id in class_ids]] = 1
    return target


class ChestDataset(Dataset):
    """只读官方 CSV 与影像，按图返回张量、检查标签和清单下标。"""

    def __init__(
        self,
        root: Path,
        split: str,
        imgsz: int,
        smoke: bool = False,
        transform: ImageTransform | None = None,
    ) -> None:
        if split not in {"train", "valid", "test"}:
            raise ValueError("split 须为 train、valid 或 test")
        if type(imgsz) is not int or imgsz < 1:
            raise ValueError("imgsz 须为正整数")
        self.root = Path(root)
        self.split = split
        self.imgsz = imgsz
        self.transform = transform
        self.rows = _read_rows(self.root, split)
        if smoke:
            # 按完整检查取样，避免截断同一 Study 的多个视图。
            studies = dict.fromkeys((row["Subject_id"], row["Study_id"]) for row in self.rows)
            selected = set(list(studies)[:SMOKE_STUDIES])
            self.rows = [
                row for row in self.rows if (row["Subject_id"], row["Study_id"]) in selected
            ]
        self.targets = self._validate_rows()

    def _validate_rows(self) -> list[torch.Tensor]:
        targets = []
        seen: set[tuple[str, ...]] = set()
        study_labels: dict[StudyKey, torch.Tensor] = {}
        for row in self.rows:
            key = tuple(row[field] for field in ID_FIELDS)
            if any(
                not value.strip()
                or value in {".", ".."}
                or any(char in value for char in '/\\:<>"|?*')
                for value in key
            ):
                raise ValueError(f"非法样本 ID {key}")
            if key in seen:
                raise ValueError(f"重复样本 {key}")
            seen.add(key)
            path = self.image_path(row)
            if not path.is_file():
                raise FileNotFoundError(path)
            target = _read_target(row, self.split)
            study = row["Subject_id"], row["Study_id"]
            previous = study_labels.setdefault(study, target)
            if not torch.equal(previous, target):
                raise ValueError(f"同一检查标签不一致 {study}")
            targets.append(target)
        return targets

    def image_path(self, row: ImageRow) -> Path:
        return (
            self.root / self.split / row["Subject_id"] / row["Study_id"]
            / f"{row['Dicom_id']}.jpg"
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image = image_tensor(self.image_path(self.rows[index]), self.imgsz, self.transform)
        return image, self.targets[index], index
