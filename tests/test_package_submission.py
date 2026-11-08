"""用独立的合成文件验证打包路径与失败保护。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from tools import package_submission as packaging


class PackageSubmissionTests(unittest.TestCase):
    def test_package_and_reject_incomplete_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(name: str, content: str) -> Path:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                return path

            weights = write("outputs/train/run/weights/best.pt", "test weight")
            pretrain = write("pretrain_model/base.pt", "test pretrained weight")
            write("outputs/train/run/args.json", json.dumps({
                "model": str(pretrain), "pretrained": True, "imgsz": 512,
                "fixed_options": {"val": True},
            }))
            write("outputs/predict/run/args.json", json.dumps({
                "model": str(weights), "split": "test", "imgsz": 512, "gate": 0.3, "quantize": 32,
                "fixed_options": {"task": "classify"},
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
                draft = packaging.package_submission(**(args | {"declaration": None}))
                self.assertIn("待核对", draft.name)
                with ZipFile(draft) as archive:
                    self.assertIn("待填写核验表.md", archive.namelist())
                    self.assertNotIn("模型核验材料与数据申明清单.xlsx", archive.namelist())
                target = packaging.package_submission(**args)
                with self.assertRaises(FileExistsError):
                    packaging.package_submission(**args)
            with ZipFile(target) as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(set(archive.namelist()), {
                    "code/train.py", "code/predict.py", "code/requirements.txt", "code/uv.lock",
                    "code/pyproject.toml", "code/README.md", "code/src/ncccu/inference.py",
                    "model/best.pt", "pretrain_model/base.pt", "licenses/LICENSE.txt",
                    "cost.py", "模型核验材料与数据申明清单.xlsx", "submission.csv",
                })
                for name in ("code/train.py", "code/predict.py"):
                    source = archive.read(name).decode("utf-8")
                    compile(source, name, "exec")
                    self.assertNotIn(str(root).replace("\\", "\\\\"), source)
                self.assertIn(b"../model/best.pt", archive.read("code/predict.py"))
                self.assertEqual(archive.read("submission.csv"), submission.read_bytes())
