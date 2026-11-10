"""用临时影像验证官方清单校验和各入口预处理一致性。"""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ncccu.data import ChestDataset, image_tensor
from ncccu.evaluation import aggregate
from ncccu.submission import write_submission
from tools.check_dataset import check_split


class DatasetContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        directory = self.root / "valid" / "patient" / "study"
        directory.mkdir(parents=True)
        self.first_image = directory / "a.jpg"
        Image.new("RGB", (40, 80), (20, 50, 100)).save(self.first_image)
        Image.new("RGB", (80, 40), (50, 100, 20)).save(directory / "b.jpg")
        self.header = ["Subject_id", "Study_id", "Dicom_id", "Predict_class"]
        self.rows = [["patient", "study", "a", "1;4"], ["patient", "study", "b", "1;4"]]

    def write_rows(self, rows: list[list[str]], header: list[str] | None = None) -> None:
        with (self.root / "valid.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(self.header if header is None else header)
            writer.writerows(rows)

    def test_dataset_and_benchmark_share_pixel_values(self) -> None:
        self.write_rows(self.rows)
        dataset = ChestDataset(self.root, "valid", imgsz=64)
        image, _, _ = dataset[0]
        torch.testing.assert_close(image, image_tensor(self.first_image, 64), rtol=0, atol=0)
        check_split(self.root, "valid")

    def test_rejects_duplicate_images_and_inconsistent_study_labels(self) -> None:
        for rows, message in (
            ([self.rows[0], self.rows[0]], "重复样本"),
            ([self.rows[0], ["patient", "study", "b", "2"]], "标签不一致"),
        ):
            with self.subTest(message=message):
                self.write_rows(rows)
                with self.assertRaisesRegex(ValueError, message):
                    ChestDataset(self.root, "valid", 64)
                with self.assertRaisesRegex(ValueError, message):
                    check_split(self.root, "valid")

    def test_rejects_invalid_ids_and_malformed_csv(self) -> None:
        for invalid_id in ("", " ", "..", "../patient", "C:patient"):
            with self.subTest(invalid_id=invalid_id):
                self.write_rows([[invalid_id, "study", "a", "1"]])
                with self.assertRaisesRegex(ValueError, "非法样本 ID"):
                    ChestDataset(self.root, "valid", 64)
        for row in (["patient", "study", "a"], [*self.rows[0], "extra"]):
            with self.subTest(row=row):
                self.write_rows([row])
                with self.assertRaisesRegex(ValueError, "列数不符"):
                    ChestDataset(self.root, "valid", 64)
        self.write_rows(self.rows, ["Subject_id", "Study_id", "Dicom_id", "Dicom_id"])
        with self.assertRaisesRegex(ValueError, "缺少必需列或包含重复列名"):
            ChestDataset(self.root, "valid", 64)

    def test_aggregation_and_csv_preserve_official_class_order(self) -> None:
        self.write_rows(self.rows)
        dataset = ChestDataset(self.root, "valid", 64)
        probabilities = np.array([[0.2] * 10, [0.8] * 10], dtype=np.float32)
        keys, scores, labels = aggregate(dataset, probabilities)
        np.testing.assert_array_equal(scores, np.full((1, 10), 0.5))
        np.testing.assert_array_equal(labels, [[1, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
        submission = self.root / "submission.csv"
        write_submission(submission, keys, scores, 0.4)
        with submission.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual([int(row["Predict_class"]) for row in rows], list(range(1, 11)))
        self.assertTrue(all(row["Probability"] == "0.5" and row["Gate"] == "0.4" for row in rows))
        with self.assertRaisesRegex(ValueError, "逐图概率形状"):
            aggregate(dataset, probabilities[:1])

    def test_invalid_probabilities_do_not_create_submission(self) -> None:
        target = self.root / "invalid.csv"
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    write_submission(target, [("patient", "study")], np.full((1, 10), value), 0.5)
                self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
