"""用独立的合成文件验证打包路径与失败保护。"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from tools import package_submission as packaging


class PackageSubmissionTests(unittest.TestCase):
    def test_package_and_reject_incomplete_submission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(name: str, content: str) -> Path:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                return path

            weights = write("outputs/train/run/weights/best.pt", "test weight")
            pretrain = write("pretrain_model/base.safetensors", "test pretrained weight")
            write("outputs/train/run/args.json", json.dumps({
                "model": str(pretrain), "imgsz": 512,
            }))
            write("outputs/predict/run/args.json", json.dumps({
                "model": str(weights), "split": "test", "imgsz": 512, "gate": 0.3, "quantize": 32,
            }))
            write("datasets/test.csv", "Subject_id,Study_id\nA,001\nA,001\n")
            write("datasets/private.jpg", "test image")
            write("outputs/train/run/weights/last.pt", "test last checkpoint")
            write("README.md", "test readme")
            write("pyproject.toml", "[project]\nname = 'ncccu'\n")
            write("uv.lock", '[[package]]\nname = "torch"\nversion = "2.15.0+cu134"\n')
            write("src/ncccu/inference.py", "# test module")
            declaration = root / "declaration.xlsx"
            with ZipFile(declaration, "w") as workbook:
                workbook.writestr("xl/workbook.xml", "<workbook/>")
            header = "Study_id,Predict_class,Probability,Gate\n"
            rows = [f"001,{label},0.7,0.3\n" for label in range(1, 11)]
            submission = write("outputs/predict/run/submission.csv", header + "".join(rows[:-1]))
            args = dict(
                team_name="测试队", training_run=root / "outputs/train/run", model=weights,
                prediction_run=submission.parent, submission=submission, pretrain_model=pretrain,
                licenses=(write("LICENSE.txt", "test license"),),
                cost_script=write("cost.py", "# test benchmark"),
                declaration=declaration, output_dir=root / "outputs/submission",
            )
            with patch.object(packaging, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "缺少 1 条"):
                    packaging.package_submission(**args)
                self.assertFalse(args["output_dir"].exists())
                submission.write_text(header + "".join(rows) + rows[0], encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "重复"):
                    packaging.package_submission(**args)
                submission.write_text(header + "".join(rows), encoding="utf-8")
                prediction_args_path = submission.parent / "args.json"
                prediction_args = packaging.read_args(prediction_args_path)
                prediction_args["model_sha256"] = "0" * 64
                prediction_args_path.write_text(json.dumps(prediction_args), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    packaging.package_submission(**args)
                self.assertFalse(args["output_dir"].exists())
                # 旧运行记录没有摘要时仍可打包；新记录中的摘要不进入函数参数。
                del prediction_args["model_sha256"]
                prediction_args_path.write_text(json.dumps(prediction_args), encoding="utf-8")
                draft = packaging.package_submission(**(args | {"declaration": None}))
                self.assertIn("待核对", draft.name)
                with ZipFile(draft) as archive:
                    self.assertIn("待填写核验表.md", archive.namelist())
                    self.assertNotIn("模型核验材料与数据申明清单.xlsx", archive.namelist())
                prediction_args["model_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
                prediction_args_path.write_text(json.dumps(prediction_args), encoding="utf-8")
                target = packaging.package_submission(**args)
                with self.assertRaises(FileExistsError):
                    packaging.package_submission(**args)
            with ZipFile(target) as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(set(archive.namelist()), {
                    "code/train.py", "code/predict.py", "code/requirements.txt", "code/uv.lock",
                    "code/pyproject.toml", "code/README.md", "code/src/ncccu/inference.py",
                    "model/best.pt", "pretrain_model/base.safetensors", "licenses/LICENSE.txt",
                    "cost.py", "模型核验材料与数据申明清单.xlsx", "submission.csv",
                })
                for name in ("code/train.py", "code/predict.py"):
                    source = archive.read(name).decode("utf-8")
                    compile(source, name, "exec")
                    self.assertNotIn(str(root).replace("\\", "\\\\"), source)
                self.assertIn(b"../model/best.pt", archive.read("code/predict.py"))
                self.assertNotIn(b"model_sha256", archive.read("code/predict.py"))
                self.assertEqual(archive.read("submission.csv"), submission.read_bytes())

    def test_read_args_requires_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "args.json"
            for value in ([], None, "invalid"):
                with self.subTest(value=value):
                    path.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "JSON 对象"):
                        packaging.read_args(path)

    def test_submission_rejects_invalid_columns_and_nonfinite_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_csv = root / "test.csv"
            test_csv.write_text("Subject_id,Study_id\nA,001\n", encoding="utf-8")
            submission = root / "submission.csv"
            invalid = (
                "Study_id,Predict_class,Probability,Gate,Gate\n001,1,0.7,0.5,0.2\n",
                "Study_id,Predict_class,Probability,Gate\n001,1,nan,0.5\n",
                "Study_id,Predict_class,Probability,Gate\n001,1,0.7,inf\n",
                "Study_id,Predict_class,Probability,Gate\n001,1,0.7\n",
            )
            for content in invalid:
                with self.subTest(content=content):
                    submission.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        packaging.validate_submission(submission, test_csv)

    def test_atomic_publish_preserves_concurrent_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "submission.zip"
            original_hardlink = Path.hardlink_to

            def concurrent_publish(path: Path, source: Path) -> None:
                path.write_bytes(b"existing package")
                original_hardlink(path, source)

            with patch.object(Path, "hardlink_to", concurrent_publish):
                with self.assertRaises(FileExistsError):
                    packaging._publish_archive(target, {}, {"readme.txt": "test"}, {})
            self.assertEqual(target.read_bytes(), b"existing package")
            self.assertEqual(list(target.parent.iterdir()), [target])

    def test_archive_hash_is_verified_before_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.pt"
            model.write_bytes(b"modified weight")
            target = root / "outputs/submission.zip"
            with self.assertRaisesRegex(ValueError, "归档期间"):
                packaging._publish_archive(
                    target, {"model/model.pt": model}, {},
                    {"model/model.pt": hashlib.sha256(b"original weight").hexdigest()},
                )
            self.assertFalse(target.exists())
            self.assertEqual(list(target.parent.iterdir()), [])
