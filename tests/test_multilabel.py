"""使用标准库 unittest 验证多标签指标和 CSV 边界。"""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from ncccu.multilabel import ChestDataset, aggregate, metrics


class MultilabelTests(unittest.TestCase):
    def test_auc_ties_and_f1(self):
        labels = np.tile([[0], [1], [0], [1]], (1, 10))
        perfect = metrics(labels, labels.astype(float))
        self.assertEqual(perfect["macro_auc"], 1.0)
        self.assertEqual(perfect["macro_f1"], 1.0)
        self.assertEqual(metrics(labels, np.full((4, 10), 0.5))["macro_auc"], 0.5)

    def test_csv_and_study_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "valid" / "subject" / "study"
            image.mkdir(parents=True)
            for name in ("a", "b"):
                Image.new("RGB", (40, 80)).save(image / f"{name}.jpg")
            def write(label):
                with (root / "valid.csv").open("w", encoding="utf-8", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Subject_id", "Study_id", "Dicom_id", "Predict_class"])
                    for name in ("a", "b"):
                        writer.writerow(["subject", "study", name, label])
            write("1;4;8;9")
            data = ChestDataset(root, "valid", 64)
            tensor, target, _ = data[0]
            self.assertEqual(tuple(tensor.shape), (3, 64, 64))
            self.assertEqual(target.tolist(), [1,0,0,1,0,0,0,1,1,0])
            keys, scores, _ = aggregate(data, np.array([[0.2]*10, [0.8]*10]))
            self.assertEqual(len(keys), 1)
            np.testing.assert_allclose(scores, 0.5)
            for label in ("", "1;1", "11", "x"):
                write(label)
                with self.assertRaises(ValueError):
                    ChestDataset(root, "valid", 64)
            write("1")
            (image / "a.jpg").unlink()
            with self.assertRaises(FileNotFoundError):
                ChestDataset(root, "valid", 64)
            with (root / "test.csv").open("w", encoding="utf-8", newline="") as file:
                file.write("Subject_id,Study_id,Dicom_id\nsubject,study,b\n")
            target_dir = root / "test" / "subject" / "study"
            target_dir.mkdir(parents=True)
            Image.new("RGB", (40, 80)).save(target_dir / "b.jpg")
            self.assertEqual(len(ChestDataset(root, "test", 64)), 1)


if __name__ == "__main__":
    unittest.main()
