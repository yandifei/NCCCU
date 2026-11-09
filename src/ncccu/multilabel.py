"""官方 CSV 多标签数据、模型与检查级评估。"""
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = (
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation",
    "Pneumonia", "Edema", "Cardiomegaly", "Atelectasis",
    "Lung Opacity", "Pleural Effusion", "No Finding",
)


def read_chest_image(path: Path, imgsz: int) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.pad(source.convert("RGB"), (imgsz, imgsz),
                             method=Image.Resampling.BILINEAR, color=(114, 114, 114))
        return np.asarray(image).copy()


def normalize_image(tensor: torch.Tensor) -> torch.Tensor:
    mean = tensor.new_tensor((0.485, 0.456, 0.406))[:, None, None]
    std = tensor.new_tensor((0.229, 0.224, 0.225))[:, None, None]
    return (tensor - mean) / std


class ChestDataset(Dataset):
    def __init__(self, root: Path, split: str, imgsz: int, smoke: bool = False):
        self.root, self.split, self.imgsz = root, split, imgsz
        self.transform = None
        with (root / f"{split}.csv").open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            required = {"Subject_id", "Study_id", "Dicom_id"}
            if split != "test":
                required.add("Predict_class")
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{split}.csv 缺少必需列")
            self.rows = list(reader)
        if not self.rows:
            raise ValueError(f"{split}.csv 为空")
        # 冒烟验证按完整检查取样，避免截断同一 Study 的多张影像。
        if smoke:
            studies = list(dict.fromkeys((r["Subject_id"], r["Study_id"]) for r in self.rows))[:16]
            self.rows = [r for r in self.rows if (r["Subject_id"], r["Study_id"]) in studies]
        self.targets = []
        seen = set()
        study_labels = {}
        for row in self.rows:
            key = tuple(row[k] for k in ("Subject_id", "Study_id", "Dicom_id"))
            if any(not v or v in {".", ".."} or any(c in v for c in "/\\:") for v in key):
                raise ValueError(f"非法样本 ID {key}")
            if key in seen:
                raise ValueError(f"重复样本 {key}")
            seen.add(key)
            if not self.image_path(row).is_file():
                raise FileNotFoundError(self.image_path(row))
            target = torch.zeros(10)
            if split != "test":
                ids = [int(v) for v in row["Predict_class"].split(";")]
                if len(set(ids)) != len(ids) or not set(ids).issubset(range(1, 11)):
                    raise ValueError(f"无效标签 {key}: {ids}")
                target[[i - 1 for i in ids]] = 1
                previous = study_labels.setdefault(key[:2], target)
                if not torch.equal(previous, target):
                    raise ValueError(f"同一检查标签不一致 {key[:2]}")
            self.targets.append(target)

    def image_path(self, row: dict) -> Path:
        return self.root / self.split / row["Subject_id"] / row["Study_id"] / (row["Dicom_id"] + ".jpg")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple:
        image = read_chest_image(self.image_path(self.rows[index]), self.imgsz)
        if self.transform is not None:
            tensor = self.transform(Image.fromarray(image))
        else:
            tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255
        return normalize_image(tensor), self.targets[index], index


def device_for(value: str | int) -> torch.device:
    value = str(value)
    return torch.device(f"cuda:{value}" if value.isdigit() else value)


def output_dir(kind: str, name: str | None) -> Path:
    name = name or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if not name or Path(name).name != name or name in {".", ".."} or any(c in name for c in "/\\:"):
        raise ValueError("name 必须是单层目录名")
    path = (ROOT / "outputs" / kind / name).resolve()
    if not path.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("全部产出必须位于 outputs 目录内")
    path.mkdir(parents=True, exist_ok=False)
    return path


def aggregate(dataset: ChestDataset, probabilities: np.ndarray) -> tuple:
    groups = {}
    for index, row in enumerate(dataset.rows):
        groups.setdefault((row["Subject_id"], row["Study_id"]), []).append(index)
    keys = list(groups)
    scores = np.stack([probabilities[indices].mean(axis=0) for indices in groups.values()])
    labels = np.stack([dataset.targets[indices[0]].numpy() for indices in groups.values()])
    return keys, scores, labels


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    aucs = []
    for target, score in zip(labels.T, scores.T):
        positives = target.sum()
        negatives = len(target) - positives
        if positives == 0 or negatives == 0:
            continue
        order = np.argsort(score, kind="stable")
        _, starts, counts = np.unique(score[order], return_index=True, return_counts=True)
        ranks = np.empty(len(score), dtype=float)
        ranks[order] = np.repeat(starts + (counts + 1) / 2, counts)
        aucs.append(float((ranks[target == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)))
    predicted = scores >= 0.5
    tp = (predicted * labels).sum(axis=0)
    denominator = predicted.sum(axis=0) + labels.sum(axis=0)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(10), where=denominator != 0)
    return {"macro_auc": float(np.mean(aucs)) if aucs else None,
            "auc_classes": len(aucs), "macro_f1": float(f1.mean())}


def predict_batch(model: torch.nn.Module, images: torch.Tensor, device: torch.device,
                  *, quantize: int, channels_last: bool, augment: bool) -> tuple[torch.Tensor, torch.Tensor]:
    images = images.to(device, non_blocking=device.type == "cuda")
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    with torch.autocast(device_type=device.type, enabled=quantize == 16):
        output = model(images).float()
        probability = output.sigmoid()
        if augment:
            probability = (probability + model(images.flip(-1)).float().sigmoid()) / 2
    if not torch.isfinite(output).all() or (augment and not torch.isfinite(probability).all()):
        raise ValueError("预测出现非有限数值")
    return output, probability


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader, device: torch.device, *, quantize: int = 32, channels_last: bool = False, augment: bool = False, verbose: bool = False) -> tuple:
    model.eval()
    probabilities = np.empty((len(loader.dataset), 10), dtype=np.float32)
    total = 0.0
    with tqdm(loader, desc="Validating", total=len(loader), disable=not verbose) as progress:
        for images, targets, indices in progress:
            targets = targets.to(device, non_blocking=device.type == "cuda")
            output, probability = predict_batch(model, images, device, quantize=quantize,
                                                channels_last=channels_last, augment=augment)
            total += torch.nn.functional.binary_cross_entropy_with_logits(
                output, targets, reduction="sum").item()
            probabilities[indices.numpy()] = probability.cpu().numpy()
    return probabilities, total / (len(loader.dataset) * 10)
