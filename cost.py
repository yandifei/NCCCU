"""赛事 batch=1 耗时和一致性核验，运行入口为提交 ZIP 中的 cost.py。"""

import argparse
import csv
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def sample_images(directory: Path, count: int) -> list[Path]:
    images = sorted(path for path in directory.rglob("*")
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) != count:
        raise ValueError(f"{directory} 须包含 {count} 张图片，实际为 {len(images)}")
    return images


def benchmark(config: dict, root: Path, submission: Path, samples: list[Path],
              warmups: list[Path]) -> tuple[float, int]:
    import numpy as np
    import torch

    from ncccu.model import load_model
    from ncccu.multilabel import device_for, normalize_image, predict_batch, read_chest_image
    from ncccu.runtime import configure_randomness

    data = root / "datasets"
    groups = {}
    for row in read_rows(data / "test.csv"):
        key = row["Subject_id"], row["Study_id"]
        groups.setdefault(key, []).append(row["Dicom_id"])

    reference = {}
    for row in read_rows(submission):
        key = row.get("Subject_id", ""), row["Study_id"]
        reference.setdefault(key, {})[int(row["Predict_class"])] = (
            float(row["Probability"]), float(row["Gate"] or 0.5))

    # 用完整官方清单判定同一 Study 有几张图，避免拿单图概率与检查均值比较。
    provided = {(path.parent.parent.name, path.parent.name, path.stem): path for path in samples + warmups}

    def study_paths(path: Path) -> tuple[tuple[str, str], list[Path]]:
        key = path.parent.parent.name, path.parent.name
        dicoms = groups.get(key, [])
        if path.stem not in dicoms:
            raise ValueError(f"抽样图不在官方测试清单内：{path}")
        paths = []
        for dicom in dicoms:
            original = data / "test" / key[0] / key[1] / f"{dicom}.jpg"
            view = path if dicom == path.stem else provided.get((*key, dicom), original)
            if not view.is_file():
                raise FileNotFoundError(f"同一检查缺少配套视图：{view}")
            paths.append(view)
        return key, paths

    configure_randomness(config["seed"], config["deterministic"])
    device = device_for(config["device"])
    model, checkpoint = load_model(root / "code" / config["model"], device)
    if config["channels_last"]:
        model.to(memory_format=torch.channels_last)

    @torch.inference_mode()
    def infer(paths: list[Path]) -> np.ndarray:
        scores = []
        for path in paths:
            array = read_chest_image(path, checkpoint["imgsz"])
            images = normalize_image(torch.from_numpy(array).permute(2, 0, 1).float() / 255).unsqueeze(0)
            _, probability = predict_batch(model, images, device, quantize=config["quantize"],
                                           channels_last=config["channels_last"], augment=config["augment"])
            scores.append(probability.cpu().numpy()[0])
        return np.stack(scores).mean(axis=0)

    work = [study_paths(path) for path in samples]
    for path in warmups:
        infer(study_paths(path)[1])
    total = 0.0
    mismatches = set()
    with TemporaryDirectory() as directory:
        result = Path(directory) / "prediction.csv"
        for _ in range(3):
            for index, (key, paths) in enumerate(work):
                started = perf_counter()
                probabilities = infer(paths)
                with result.open("w", encoding="utf-8", newline="") as file:
                    csv.writer(file).writerows(
                        (*key, label, float(value), config["gate"])
                        for label, value in enumerate(probabilities, start=1))
                total += perf_counter() - started
                rows = reference.get(key, reference.get(("", key[1]), {}))
                if set(rows) != set(range(1, 11)):
                    raise ValueError(f"提交 CSV 缺少该检查的十类概率：{key}")
                expected, gates = np.asarray([rows[label] for label in range(1, 11)]).T
                if (np.max(np.abs(probabilities - expected)) > 1e-3
                        or not np.array_equal(probabilities >= config["gate"], expected >= gates)
                        or not np.array_equal(np.argsort(-probabilities, kind="stable"),
                                              np.argsort(-expected, kind="stable"))):
                    mismatches.add(index)
    return total * 1000 / (3 * len(work)), len(mismatches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("pictures_100", type=Path)
    parser.add_argument("pictures_5", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "code"))
    from predict import CONFIG

    latency, mismatches = benchmark(CONFIG, root, args.submission,
                                    sample_images(args.pictures_100, 100), sample_images(args.pictures_5, 5))
    print(f"[CONCLUSION] 100 张样本、batch=1 串行、3 轮，平均 {latency:.2f} ms，"
          f"不一致 {mismatches} 张，推理精度 FP{CONFIG['quantize']}。"
          "多视图检查计入全部视图推理耗时。")
    if mismatches or latency > 100:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
