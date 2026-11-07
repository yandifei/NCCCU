"""验证入口实际传参、启用的产物开关和确定性恢复。"""
import ast
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from ncccu import multilabel, training
from ncccu.inference import predict_multilabel
from ncccu.runtime import validate_fixed
from ncccu.train_options import TRAIN_FIXED


def entry_parameters(filename: str, function: str) -> dict:
    tree = ast.parse((multilabel.ROOT / filename).read_text(encoding="utf-8-sig"))
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == function)
    return {item.arg: ast.literal_eval(item.value) for item in call.keywords}


class EnabledOptionsTests(unittest.TestCase):
    def test_unknown_and_inapplicable_options_fail(self):
        with self.assertRaises(TypeError):
            validate_fixed({"deterministc": True}, TRAIN_FIXED)
        with self.assertRaises(ValueError):
            validate_fixed({"box": 7.5}, TRAIN_FIXED)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_cuda_determinism_with_workers(self):
        data = multilabel.ROOT / "datasets"
        if not (data / "train.csv").exists():
            self.skipTest("需要本地官方数据")
        args = entry_parameters("train.py", "train_multilabel")
        args.update(model=multilabel.ROOT / "pretrain_model/yolo26n-cls.pt",
                    data=data, epochs=2, imgsz=64, batch=4, nbs=4, smoke_test=True,
                    workers=3, save_period=1, warmup_bias_lr=0.001)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(multilabel, "ROOT", Path(directory)):
                args["name"] = "first"
                first = training.train_multilabel(**args)
                args["name"] = "second"
                second = training.train_multilabel(**args)
                left = torch.load(first / "weights/last.pt", map_location="cpu", weights_only=True)
                right = torch.load(second / "weights/last.pt", map_location="cpu", weights_only=True)
                for key in left["state_dict"]:
                    torch.testing.assert_close(left["state_dict"][key], right["state_dict"][key], rtol=0, atol=0)
                self.assertEqual(left["metrics"], right["metrics"])
                restored = training.train_multilabel(
                    model=first / "weights/epoch1.pt", data=data, resume=True,
                    device=0, workers=3, name="restored")
                recovered = torch.load(restored / "weights/last.pt", map_location="cpu", weights_only=True)
                for key in left["state_dict"]:
                    torch.testing.assert_close(left["state_dict"][key], recovered["state_dict"][key], rtol=0, atol=0)
                self.assertEqual(left["metrics"], recovered["metrics"])
                self.assertTrue(left["training_state"]["scaler"])
                self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_entry_options_outputs_and_resume(self):
        weights = multilabel.ROOT / "pretrain_model/yolo26n-cls.pt"
        if not weights.is_file():
            self.skipTest("需要本地预训练权重")
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        train_args = entry_parameters("train.py", "train_multilabel")
        predict_args = entry_parameters("predict.py", "predict_multilabel")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "datasets"
            data.mkdir()
            for split in ("train", "valid", "test"):
                with (data / f"{split}.csv").open("w", encoding="utf-8", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Subject_id", "Study_id", "Dicom_id", "Predict_class"])
                    for index in range(4):
                        path = data / split / f"subject{index}" / f"study{index}"
                        path.mkdir(parents=True)
                        Image.new("RGB", (80, 90), (30+index*25, 90, 120)).save(path / "a.jpg")
                        label = "1;2;3;4;5" if index % 2 else "6;7;8;9;10"
                        writer.writerow([f"subject{index}", f"study{index}", "a", label])
            train_args.update(model=weights, data=data, device="cpu", workers=0, epochs=2,
                              imgsz=64, batch=2, nbs=4, save_period=1, warmup_epochs=1,
                              fliplr=0.5, erasing=0.2, dropout=0.1, amp=False,
                              warmup_bias_lr=0.001, name="continuous")
            with patch.object(multilabel, "ROOT", root):
                complete = training.train_multilabel(**train_args)
                self.assertTrue((complete / "results.png").is_file())
                self.assertTrue((complete / "weights/epoch1.pt").is_file())
                self.assertEqual(len(list((complete / "cache/train").glob("*.npy"))), 4)
                original = training.evaluate
                count = 0

                def interrupt(*args, **kwargs):
                    nonlocal count
                    count += 1
                    if count == 2:
                        raise RuntimeError("模拟中断")
                    return original(*args, **kwargs)

                train_args["name"] = "interrupted"
                with patch.object(training, "evaluate", side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, "模拟中断"):
                        training.train_multilabel(**train_args)
                resumed = training.train_multilabel(
                    model=root / "outputs/train/interrupted/weights/last.pt", data=data,
                    resume=True, device="cpu", name="resumed")
                expected = torch.load(complete / "weights/last.pt", weights_only=True)
                actual = torch.load(resumed / "weights/last.pt", weights_only=True)
                for key in expected["state_dict"]:
                    torch.testing.assert_close(expected["state_dict"][key], actual["state_dict"][key], rtol=0, atol=0)
                self.assertEqual(expected["training_state"]["history"], actual["training_state"]["history"])
                predict_args.update(model=resumed / "weights/best.pt", data=data,
                                    source=data / "test.csv", device="cpu", workers=0,
                                    imgsz=64, name="predicted")
                predict_args["export_options"].update(imgsz=64, device="cpu")
                predicted = predict_multilabel(**predict_args)
                self.assertTrue((predicted / "model.torchscript").is_file())
                self.assertEqual(len(list((predicted / "images").rglob("*.jpg"))), 4)
                self.assertEqual(len(list((predicted / "labels").rglob("*.txt"))), 4)
                predict_args.update(name="repeat", save=False, save_txt=False)
                predict_args["export_options"]["enabled"] = False
                repeat = predict_multilabel(**predict_args)
                self.assertEqual((predicted / "submission.csv").read_bytes(), (repeat / "submission.csv").read_bytes())
                self.assertTrue(torch.are_deterministic_algorithms_enabled())


if __name__ == "__main__":
    unittest.main()
