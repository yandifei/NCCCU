"""赛事 batch=1 耗时和一致性核验，运行入口为提交 ZIP 中的 cost.py。"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

if TYPE_CHECKING:
    import numpy as np


CLASS_IDS = tuple(range(1, 11))
BENCHMARK_ROUNDS = 3
PROBABILITY_TOLERANCE = 1e-3
MAX_LATENCY_MS = 100

StudyKey = tuple[str, str]
StudyImages = dict[StudyKey, list[str]]
ReferenceScores = dict[StudyKey, dict[int, tuple[float, float]]]


def read_rows(path: Path, *, required: set[str] | None = None) -> list[dict[str, str]]:
    """在 CSV 边界拒绝重复列名和不完整行，避免字典读取静默覆盖。"""
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not fields:
            raise ValueError(f"CSV 表头为空或存在重复列名：{path}")
        if required is not None and not required.issubset(fields):
            raise ValueError(f"CSV 缺少必需列 {sorted(required - set(fields))}：{path}")
        rows: list[dict[str, str]] = []
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path} 第 {line} 行列数不符")
            rows.append(row)
    return rows


def sample_images(directory: Path, count: int) -> list[Path]:
    images = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(images) != count:
        raise ValueError(f"{directory} 须包含 {count} 张图片，实际为 {len(images)}")
    return images


def _study_images(test_csv: Path) -> StudyImages:
    groups: StudyImages = {}
    seen: set[tuple[str, str, str]] = set()
    for row in read_rows(test_csv, required={"Subject_id", "Study_id", "Dicom_id"}):
        subject, study, dicom = (row[name] for name in ("Subject_id", "Study_id", "Dicom_id"))
        key = subject, study, dicom
        if any(not value or value in {".", ".."} or any(char in value for char in "/\\:") for value in key):
            raise ValueError(f"官方测试清单存在无效样本 ID：{key}")
        if key in seen:
            raise ValueError(f"官方测试清单存在重复样本：{key}")
        seen.add(key)
        groups.setdefault((subject, study), []).append(dicom)
    if not groups:
        raise ValueError("官方 test.csv 没有检查记录")
    return groups


def _reference_scores(submission: Path, groups: StudyImages) -> ReferenceScores:
    rows = read_rows(submission, required={"Study_id", "Predict_class", "Probability", "Gate"})
    has_subject = bool(rows) and "Subject_id" in rows[0]
    study_keys = {study: (subject, study) for subject, study in groups}
    if not has_subject and len(study_keys) != len(groups):
        raise ValueError("官方测试集中存在跨患者重复的 Study_id，CSV 须包含 Subject_id")
    reference: ReferenceScores = {}
    for line, row in enumerate(rows, start=2):
        try:
            key = (row["Subject_id"], row["Study_id"]) if has_subject else study_keys.get(row["Study_id"])
            label = int(row["Predict_class"])
            probability = float(row["Probability"])
            gate = float(row["Gate"] or 0.5)
            if key not in groups or label not in CLASS_IDS:
                raise ValueError("未知检查或类别")
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in (probability, gate)):
                raise ValueError("Probability 和 Gate 须为 [0, 1] 内的有限数值")
            if label in reference.setdefault(key, {}):
                raise ValueError("重复的检查/类别")
            reference[key][label] = probability, gate
        except (TypeError, ValueError) as error:
            raise ValueError(f"submission.csv 第 {line} 行无效：{error}") from error
    for key in groups:
        if set(reference.get(key, {})) != set(CLASS_IDS):
            raise ValueError(f"提交 CSV 缺少该检查的十类概率：{key}")
    return reference


def _study_paths(
    path: Path, data: Path, groups: StudyImages, provided: dict[tuple[str, str, str], Path],
) -> tuple[StudyKey, list[Path]]:
    """补齐抽样检查的全部视图，并保持官方清单中的聚合顺序。"""
    key = path.parent.parent.name, path.parent.name
    dicoms = groups.get(key, [])
    if path.stem not in dicoms:
        raise ValueError(f"抽样图不在官方测试清单内：{path}")
    paths: list[Path] = []
    for dicom in dicoms:
        original = data / "test" / key[0] / key[1] / f"{dicom}.jpg"
        view = path if dicom == path.stem else provided.get((*key, dicom), original)
        if not view.is_file():
            raise FileNotFoundError(f"同一检查缺少配套视图：{view}")
        paths.append(view)
    return key, paths


def _load_predictor(config: Mapping[str, Any], root: Path) -> Callable[[list[Path]], np.ndarray]:
    import numpy as np
    import torch

    from ncccu.data import image_tensor
    from ncccu.evaluation import predict_batch
    from ncccu.model import load_model
    from ncccu.runtime import configure_randomness, device_for

    configure_randomness(config["seed"], config["deterministic"])
    device = device_for(config["device"])
    if config["quantize"] == 16 and device.type != "cuda":
        raise ValueError("FP16 测速须使用 CUDA 设备")
    model, checkpoint = load_model(root / "code" / config["model"], device)
    if config["channels_last"]:
        model.to(memory_format=torch.channels_last)

    @torch.inference_mode()
    def infer(paths: list[Path]) -> np.ndarray:
        scores: list[np.ndarray] = []
        for path in paths:
            images = image_tensor(path, checkpoint["imgsz"]).unsqueeze(0)
            _, probability = predict_batch(
                model, images, device, quantize=config["quantize"],
                channels_last=config["channels_last"], augment=config["augment"],
            )
            scores.append(probability.cpu().numpy()[0])
        return np.stack(scores).mean(axis=0)

    return infer


def _matches_reference(probabilities: np.ndarray, rows: dict[int, tuple[float, float]], gate: float) -> bool:
    import numpy as np

    if probabilities.shape != (len(CLASS_IDS),) or not np.isfinite(probabilities).all():
        return False
    expected, gates = np.asarray([rows[label] for label in CLASS_IDS]).T
    return bool(
        np.max(np.abs(probabilities - expected)) <= PROBABILITY_TOLERANCE
        and np.array_equal(probabilities >= gate, expected >= gates)
        and np.array_equal(
            np.argsort(-probabilities, kind="stable"), np.argsort(-expected, kind="stable"),
        )
    )


def benchmark(
    config: dict[str, Any], root: Path, submission: Path, samples: list[Path], warmups: list[Path],
) -> tuple[float, int]:
    if not samples:
        raise ValueError("测速样本不能为空")
    gate = config.get("gate")
    if not isinstance(gate, (int, float)) or not math.isfinite(gate) or not 0 <= gate <= 1:
        raise ValueError("测速 gate 须为 [0, 1] 内的有限数值")
    if config.get("quantize") not in (16, 32):
        raise ValueError("测速 quantize 须为 16 或 32")

    from ncccu.submission import write_submission

    data = root / "datasets"
    groups = _study_images(data / "test.csv")
    reference = _reference_scores(submission, groups)
    provided = {(path.parent.parent.name, path.parent.name, path.stem): path for path in samples + warmups}
    work = [_study_paths(path, data, groups, provided) for path in samples]
    warmup_work = [_study_paths(path, data, groups, provided) for path in warmups]
    infer = _load_predictor(config, root)
    for _, paths in warmup_work:
        infer(paths)

    total = 0.0
    mismatches: set[int] = set()
    with TemporaryDirectory() as directory:
        result = Path(directory) / "prediction.csv"
        for _ in range(BENCHMARK_ROUNDS):
            for index, (key, paths) in enumerate(work):
                started = perf_counter()
                probabilities = infer(paths)
                write_submission(result, [key], probabilities[None, :], gate, header=False)
                total += perf_counter() - started
                if not _matches_reference(probabilities, reference[key], gate):
                    mismatches.add(index)
    return total * 1000 / (BENCHMARK_ROUNDS * len(work)), len(mismatches)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("pictures_100", type=Path)
    parser.add_argument("pictures_5", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "code"))
    from predict import CONFIG

    latency, mismatches = benchmark(
        CONFIG, root, args.submission,
        sample_images(args.pictures_100, 100), sample_images(args.pictures_5, 5),
    )
    print(
        f"[CONCLUSION] 100 张样本、batch=1 串行、{BENCHMARK_ROUNDS} 轮，平均 {latency:.2f} ms，"
        f"不一致 {mismatches} 张，推理精度 FP{CONFIG['quantize']}。"
        "多视图检查计入全部视图推理耗时。"
    )
    if mismatches or latency > MAX_LATENCY_MS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
