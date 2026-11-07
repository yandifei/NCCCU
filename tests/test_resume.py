"""真实模型验证完整训练与中断恢复得到相同权重。"""
import csv
import io
from contextlib import redirect_stdout
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from ncccu import training, multilabel
from ncccu.inference import predict_multilabel


class ResumeTests(unittest.TestCase):
    def test_interruption_resume_and_prediction(self):
        weights = multilabel.ROOT / "pretrain_model/yolo26n-cls.pt"
        if not weights.is_file():
            self.skipTest("需要本地 yolo26n-cls.pt")
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, original_threads)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "datasets"
            data.mkdir()
            for split in ("train", "valid", "test"):
                with (data / f"{split}.csv").open("w", encoding="utf-8", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Subject_id", "Study_id", "Dicom_id", "Predict_class"])
                    for i, label in enumerate(("1;2;3;4;5", "6;7;8;9;10")):
                        path = data / split / f"subject{i}" / f"study{i}"
                        path.mkdir(parents=True)
                        Image.new("RGB", (70, 90), (80+i*40, 90, 120)).save(path / "image.jpg")
                        writer.writerow([f"subject{i}", f"study{i}", "image", label])
            config = dict(model=weights, data=data, epochs=2, batch=2, imgsz=64,
                          dropout=0.2, device="cpu", workers=0, amp=False)
            with patch.object(multilabel, "ROOT", root):
                console = io.StringIO()
                with redirect_stdout(console):
                    complete = training.train_multilabel(**config, name="complete")
                for column in ("Epoch", "GPU_mem", "BCE_loss", "Instances", "Size", "Macro-AUC", "Best-AUC", "Delta", "100%"):
                    self.assertIn(column, console.getvalue())
                evaluate = training.evaluate
                calls = 0

                def interrupted(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise RuntimeError("模拟第二轮中断")
                    return evaluate(*args, **kwargs)

                with patch.object(training, "evaluate", side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, "模拟"):
                        training.train_multilabel(**config, name="interrupted")
                last = root / "outputs/train/interrupted/weights/last.pt"
                saved = torch.load(last, weights_only=True)
                self.assertEqual(saved["epoch"], 1)
                resumed = training.train_multilabel(
                    model=last, data=data, resume=True, device="cpu", name="resumed")
                first = torch.load(complete / "weights/last.pt", weights_only=True)
                second = torch.load(resumed / "weights/last.pt", weights_only=True)
                self.assertEqual(second["epoch"], 2)
                for key in first["state_dict"]:
                    torch.testing.assert_close(first["state_dict"][key], second["state_dict"][key], rtol=0, atol=0)
                for key in ("scheduler", "history", "best", "stale"):
                    self.assertEqual(first["training_state"][key], second["training_state"][key])
                # 迁移旧版仅有基础参数的检查点，保留原单组优化器和无预热行为。
                legacy_keys = {
                    "model", "data", "resume", "epochs", "imgsz", "batch", "workers",
                    "device", "lr0", "lrf", "weight_decay", "dropout", "patience",
                    "seed", "amp", "name", "smoke_test",
                }
                saved["training_state"]["config"] = {
                    key: value for key, value in saved["training_state"]["config"].items()
                    if key in legacy_keys
                }
                legacy_path = root / "outputs/legacy.pt"
                torch.save(saved, legacy_path)
                migrated = training.train_multilabel(
                    model=legacy_path, data=data, resume=True, device="cpu", name="migrated",
                    warmup_epochs=3, warmup_bias_lr=0.1)
                migrated_state = torch.load(migrated / "weights/last.pt", weights_only=True)
                for key in first["state_dict"]:
                    torch.testing.assert_close(first["state_dict"][key], migrated_state["state_dict"][key], rtol=0, atol=0)
                predicted = predict_multilabel(
                    model=resumed / "weights/best.pt", data=data, device="cpu", name="predicted")
                with (predicted / "submission.csv").open(encoding="utf-8") as file:
                    self.assertEqual(len(list(csv.DictReader(file))), 20)
                with self.assertRaisesRegex(ValueError, "仅支持推理"):
                    training.train_multilabel(
                        model=resumed / "weights/best.pt", data=data, resume=True, device="cpu")
                with self.assertRaisesRegex(ValueError, "已完成"):
                    training.train_multilabel(
                        model=resumed / "weights/last.pt", data=data, resume=True, device="cpu")


if __name__ == "__main__":
    unittest.main()
