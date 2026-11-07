#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 NCCCU 胸部 X 光「多标签分类」CSV 转成逐图 txt 标签。

数据来源（本仓库 datasets/）:
    datasets/train.csv / valid.csv / test.csv
        列: Dicom_id, Subject_id, Study_id, ViewPosition, PatientOrientation,
            Rows, Columns, Predict_class
        Predict_class 为分号分隔的类别 id，例如 "1;8;9;4"（多标签，可共现）。
    图片:
        datasets/{split}/{Subject_id}/{Study_id}/{Dicom_id}.jpg

输出（默认写 datasets/labels/）:
    datasets/labels/{split}/{Subject_id}/{Study_id}/{Dicom_id}.txt
        txt 内容: 一行，空格分隔的类别索引，例如 "0 7 8"（默认 0-based，YOLO 习惯）
    datasets/labels/classes.txt
        每行一个类别名，行号即索引（0..9），顺序同官方类别 id 1..10
    datasets/labels/{split}_manifest.csv
        图片路径 / 标签路径 / 原始标签 / 标签数 的索引表

说明:
    * 本任务的 txt 是「多标签分类」用的类别索引列表，不是检测框标签
      （比赛没有 bbox 标注，detect/segment/pose/obb 都无法训练）。
    * Ultralytics 官方 -cls 模型只认 ImageFolder（一图一标签），
      读不了本脚本生成的 txt；txt 供自定义多标签 Dataset 使用。
    * 类别 id 1..10 → 索引 0..9（--id-base 1 可保留原始 1..10）。

用法:
    python tools/csv2txt.py                     # 转换 train + valid
    python tools/csv2txt.py --splits train      # 只转 train
    python tools/csv2txt.py --dry-run           # 只统计不写文件
    python tools/csv2txt.py --limit 100         # 试跑前 100 行
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

# 官方类别表：类别 id 1..10 → 名称（见 docs/比赛资料/比赛规则文档.md）
CLASS_NAMES = [
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
]
NUM_CLASSES = len(CLASS_NAMES)
VALID_IDS = set(range(1, NUM_CLASSES + 1))


def default_root() -> Path:
    """默认 datasets 目录：仓库根目录下的 datasets/。"""
    return Path(__file__).resolve().parent.parent / "datasets"


def parse_label(raw: str, id_base: int) -> list[int]:
    """解析 "1;8;9" → [0, 7, 8]（id_base=1 时保留 [1, 8, 9]）。

    - 兼容空串、nan、多余空白、重复类别；
    - 遇到非法类别 id（不在 1..10）直接抛 ValueError，避免静默写坏标签。
    """
    raw = (raw or "").strip()
    if raw == "" or raw.lower() in {"nan", "none"}:
        return []
    ids: set[int] = set()
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        cid = int(part)
        if cid not in VALID_IDS:
            raise ValueError(f"非法类别 id: {cid!r}（原始值 {raw!r}）")
        ids.add(cid)
    return sorted(i - 1 if id_base == 0 else i for i in ids)


def iter_rows(csv_path: Path, limit: int | None):
    """逐行读取 CSV（utf-8-sig 兼容 BOM），返回 dict。"""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV 为空或无法解析: {csv_path}")
        if "Predict_class" not in reader.fieldnames:
            return
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            yield row


def convert_split(
    root: Path,
    out_dir: Path,
    split: str,
    id_base: int,
    dry_run: bool,
    check_images: bool,
    one_per_line: bool,
    limit: int | None,
) -> dict:
    csv_path = root / f"{split}.csv"
    if not csv_path.exists():
        raise SystemExit(f"找不到 {csv_path}")

    stats = {
        "split": split,
        "rows": 0,
        "written": 0,
        "skipped_no_label": 0,
        "missing_images": [],
        "per_class": Counter(),
        "label_count_hist": Counter(),
        "manifest": [],
    }

    for row in iter_rows(csv_path, limit):
        stats["rows"] += 1
        subject = (row.get("Subject_id") or "").strip()
        study = (row.get("Study_id") or "").strip()
        dicom = (row.get("Dicom_id") or "").strip()
        raw = row.get("Predict_class") or ""
        try:
            ids = parse_label(raw, id_base)
        except ValueError as exc:
            raise SystemExit(
                f"{split}.csv 第 {stats['rows'] + 1} 行解析失败: {exc}"
            ) from exc

        image_path = root / split / subject / study / f"{dicom}.jpg"
        label_path = out_dir / split / subject / study / f"{dicom}.txt"

        if check_images and not image_path.exists():
            stats["missing_images"].append(str(image_path))
            continue

        if not ids:
            stats["skipped_no_label"] += 1
            continue

        content = "\n".join(str(i) for i in ids) if one_per_line else " ".join(map(str, ids))
        if not dry_run:
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(content + "\n", encoding="utf-8")
        stats["written"] += 1
        stats["label_count_hist"][len(ids)] += 1
        for i in ids:
            stats["per_class"][i] += 1
        stats["manifest"].append(
            {
                "Dicom_id": dicom,
                "Subject_id": subject,
                "Study_id": study,
                "image": image_path.relative_to(root.parent).as_posix(),
                "label": label_path.relative_to(root.parent).as_posix(),
                "classes": raw.strip(),
                "num_classes": len(ids),
            }
        )

    if not dry_run:
        stats["written"] and (out_dir / f"{split}_manifest.csv").parent.mkdir(
            parents=True, exist_ok=True
        )
        manifest_path = out_dir / f"{split}_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Dicom_id",
                    "Subject_id",
                    "Study_id",
                    "image",
                    "label",
                    "classes",
                    "num_classes",
                ],
            )
            writer.writeheader()
            writer.writerows(stats["manifest"])
    return stats


def write_classes(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classes.txt").write_text(
        "\n".join(CLASS_NAMES) + "\n", encoding="utf-8"
    )


def print_report(all_stats: list[dict], out_dir: Path, dry_run: bool) -> None:
    print("\n================ 转换结果 ================")
    for s in all_stats:
        print(
            f"[{s['split']}] CSV 行数 {s['rows']}，写出标签 {s['written']}，"
            f"无标签跳过 {s['skipped_no_label']}，图片缺失 {len(s['missing_images'])}"
        )
        if s["missing_images"]:
            for p in s["missing_images"][:5]:
                print(f"    缺失: {p}")
        if s["label_count_hist"]:
            hist = " ".join(
                f"{k}个标签:{v}" for k, v in sorted(s["label_count_hist"].items())
            )
            print(f"    每图标签数分布: {hist}")
    if all_stats and all_stats[0]["per_class"]:
        print("\n各类别正样本图数（0-based 索引）:")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {i:2d} {name:<26s} {all_stats[0]['per_class'].get(i, 0)}")
    print(f"\n标签目录: {out_dir}" + ("（dry-run，未写文件）" if dry_run else ""))
    print("==========================================")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把 NCCCU 多标签 CSV 转成逐图 txt 标签（多标签分类用）"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root(),
        help="datasets 目录（默认：仓库下 datasets/）",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="标签输出目录（默认：<root>/labels）"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid"],
        help="要转换的划分（默认 train valid；test 无标签会跳过）",
    )
    parser.add_argument(
        "--id-base",
        type=int,
        choices=(0, 1),
        default=0,
        help="写入 txt 的类别编号基数：0=转成 0..9（默认，YOLO 习惯），1=保留 1..10",
    )
    parser.add_argument(
        "--one-per-line",
        action="store_true",
        help="每个类别一行（默认一行空格分隔）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    parser.add_argument(
        "--no-check-images",
        action="store_true",
        help="不校验图片是否存在（默认校验，缺失会列出）",
    )
    parser.add_argument("--limit", type=int, default=None, help="每个划分最多处理多少行")
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    out_dir: Path = (args.out or (root / "labels")).resolve()
    if not root.is_dir():
        raise SystemExit(f"datasets 目录不存在: {root}")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_classes(out_dir)

    all_stats = []
    for split in args.splits:
        csv_path = root / f"{split}.csv"
        if not csv_path.exists():
            print(f"[跳过] 找不到 {csv_path}")
            continue
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f), [])
        if "Predict_class" not in header:
            print(f"[跳过] {split}.csv 没有 Predict_class 列（测试集无标签）")
            continue
        stats = convert_split(
            root=root,
            out_dir=out_dir,
            split=split,
            id_base=args.id_base,
            dry_run=args.dry_run,
            check_images=not args.no_check_images,
            one_per_line=args.one_per_line,
            limit=args.limit,
        )
        all_stats.append(stats)

    if not all_stats:
        print("没有可转换的划分。")
        return 1

    print_report(all_stats, out_dir, args.dry_run)
    missing = sum(len(s["missing_images"]) for s in all_stats)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
