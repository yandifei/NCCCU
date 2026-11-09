"""核对轮次恢复、真实 CSV 生成和多视图测速一致性。"""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from cost import benchmark
from ncccu import inference, model, multilabel, training


def tiny_model(dropout: float, pretrained: Path | None = None) -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Conv2d(3, 8, 1), torch.nn.AdaptiveAvgPool2d(1),
                               torch.nn.Flatten(), torch.nn.Dropout(dropout), torch.nn.Linear(8, 10))


class ResumeTest(unittest.TestCase):
    def test_epoch_resume_prediction_and_benchmark(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "datasets"
            data.mkdir()
            for split in ("train", "valid", "test"):
                with (data / f"{split}.csv").open("w", encoding="utf-8", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Subject_id", "Study_id", "Dicom_id", "Predict_class"])
                    for subject in range(4):
                        directory = data / split / str(subject) / "study"
                        directory.mkdir(parents=True)
                        for view in range(2 if subject == 0 else 1):
                            Image.new("RGB", (34, 40), (50 + subject * 40, 10 + view * 40, 110)).save(directory / f"{view}.jpg")
                            labels = "1;2;3;4;5" if subject % 2 else "6;7;8;9;10"
                            writer.writerow([subject, "study", view, labels])
            pretrained = root / "test.safetensors"
            pretrained.touch()
            args = dict(model=pretrained, data=data, epochs=2, patience=0, imgsz=32,
                        batch=2, nbs=4, workers=0, device="cpu", warmup_epochs=1)
            with (patch.object(multilabel, "ROOT", root), patch.object(training, "ROOT", root),
                  patch.object(inference, "ROOT", root), patch.object(training, "build_model", tiny_model),
                  patch.object(model, "build_model", tiny_model)):
                complete = training.train_multilabel(**args, name="complete")
                original = training.evaluate
                validations = 0

                def interrupt(*args, **kwargs):
                    nonlocal validations
                    validations += 1
                    if validations == 2:
                        raise InterruptedError("模拟第二轮中断")
                    return original(*args, **kwargs)

                with patch.object(training, "evaluate", interrupt), self.assertRaises(InterruptedError):
                    training.train_multilabel(**args, name="resume")
                last = root / "outputs/train/resume/weights/last.pt"
                resumed = training.train_multilabel(model=last, resume=True, device="cpu")
                self.assertEqual(resumed, last.parent.parent)
                expected = model.read_checkpoint(complete / "weights/last.pt")
                actual = model.read_checkpoint(last)
                self.assertEqual(expected["training_state"]["history"], actual["training_state"]["history"])
                for key, tensor in expected["state_dict"].items():
                    torch.testing.assert_close(tensor, actual["state_dict"][key], rtol=0, atol=0)
                predicted = inference.predict_multilabel(model=resumed / "weights/best.pt", data=data,
                                                         device="cpu", workers=0)
                submission = predicted / "submission.csv"
                with submission.open(encoding="utf-8") as file:
                    self.assertEqual(len(list(csv.DictReader(file))), 40)
                config = json.loads((predicted / "args.json").read_text(encoding="utf-8"))
                _, mismatches = benchmark(config, root, submission, [data / "test/0/study/0.jpg"], [])
                self.assertEqual(mismatches, 0)
