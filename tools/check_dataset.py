"""只读检查官方 CSV、类别 ID 和影像路径，无需转换标签。"""

import argparse
import csv
from pathlib import Path


def check_split(root: Path, split: str) -> None:
    seen: set[tuple[str, str, str]] = set()
    studies: set[tuple[str, str]] = set()
    multilabel = 0
    with (root / f"{split}.csv").open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"Subject_id", "Study_id", "Dicom_id"}
        if split != "test":
            required.add("Predict_class")
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{split}.csv 缺少必需列 {sorted(required)}")
        for line, row in enumerate(reader, start=2):
            key = tuple(row[name] for name in ("Subject_id", "Study_id", "Dicom_id"))
            if any(not value or any(c in value for c in '/\\:') or value in {'.', '..'} for value in key):
                raise ValueError(f"{split}:{line} 非法样本 ID")
            if key in seen:
                raise ValueError(f"{split}:{line} 重复样本 {key}")
            seen.add(key)
            studies.add(key[:2])
            image = root / split / key[0] / key[1] / f"{key[2]}.jpg"
            if not image.is_file():
                raise FileNotFoundError(image)
            if split != "test":
                try:
                    ids = [int(value.strip()) for value in (row["Predict_class"] or "").split(";")]
                except ValueError as error:
                    raise ValueError(f"{split}:{line} 标签为空或格式错误") from error
                if not set(ids).issubset(range(1, 11)) or len(ids) != len(set(ids)):
                    raise ValueError(f"{split}:{line} 类别超出 1–10 或存在重复 {ids}")
                multilabel += len(ids) > 1
    if not seen:
        raise ValueError(f"{split}.csv 没有样本")
    print(f"{split}: {len(seen)} 张影像，{len(studies)} 次检查，多标签影像 {multilabel}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "datasets")
    args = parser.parse_args()
    for split in ("train", "valid", "test"):
        check_split(args.root, split)


if __name__ == "__main__":
    main()
