"""只读检查官方 CSV、类别 ID、检查标签一致性和影像路径。"""

import argparse
from pathlib import Path

from ncccu.data import ChestDataset
from ncccu.runtime import ROOT


def check_split(root: Path, split: str) -> None:
    """与训练共用清单校验规则，仅检查路径，不解码或修改影像。"""
    dataset = ChestDataset(root, split, imgsz=32)
    studies = {(row["Subject_id"], row["Study_id"]) for row in dataset.rows}
    multilabel_count = sum(int(target.sum().item() > 1) for target in dataset.targets)
    print(f"{split}: {len(dataset)} 张影像，{len(studies)} 次检查，多标签影像 {multilabel_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "datasets")
    args = parser.parse_args()
    for split in ("train", "valid", "test"):
        check_split(args.root, split)


if __name__ == "__main__":
    main()
