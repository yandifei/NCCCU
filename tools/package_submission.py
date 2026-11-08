"""检查并打包 NCCCU 最终提交材料。"""

import csv
import hashlib
import json
import tempfile
import tomllib
from pathlib import Path
from pprint import pformat
from zipfile import ZIP_DEFLATED, ZipFile, is_zipfile


ROOT = Path(__file__).resolve().parents[1]


def validate_submission(submission: Path, test_csv: Path) -> None:
    with test_csv.open(encoding="utf-8-sig", newline="") as file:
        studies = {(row["Subject_id"], row["Study_id"]) for row in csv.DictReader(file)}
    with submission.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or ())
        if not {"Study_id", "Predict_class", "Probability", "Gate"} <= fields:
            raise ValueError("submission.csv 缺少 Study_id、Predict_class、Probability 或 Gate")
        has_subject = "Subject_id" in fields
        study_ids = {study for _, study in studies}
        if not has_subject and len(study_ids) != len(studies):
            raise ValueError("官方测试集中存在跨患者重复的 Study_id，CSV 须包含 Subject_id")
        expected_studies = studies if has_subject else {("", study) for study in study_ids}
        expected = {(subject, study, label) for subject, study in expected_studies for label in range(1, 11)}
        if not expected:
            raise ValueError("官方 test.csv 没有检查记录")
        seen = set()
        for line, row in enumerate(reader, start=2):
            try:
                key = (row["Subject_id"] if has_subject else "", row["Study_id"], int(row["Predict_class"]))
                probability = float(row["Probability"])
                gate = 0.5 if row["Gate"] == "" else float(row["Gate"])
                if None in row or key not in expected or key in seen:
                    raise ValueError("未知或重复的检查/类别，或列数不符")
                if not (0 <= probability <= 1 and 0 <= gate <= 1):
                    raise ValueError("Probability 和 Gate 须在 [0, 1] 内")
            except (TypeError, ValueError) as error:
                raise ValueError(f"submission.csv 第 {line} 行无效：{error}") from error
            seen.add(key)
        if missing := expected - seen:
            raise ValueError(f"submission.csv 缺少 {len(missing)} 条官方检查/类别记录")


def read_args(path: Path) -> dict:
    args = json.loads(path.read_text(encoding="utf-8"))
    args.update(args.pop("fixed_options", {}))
    return args


def entry_script(module: str, function: str, args: dict) -> str:
    return (
        "from pathlib import Path\n"
        "import sys\n\n"
        "CODE = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(CODE / 'src'))\n"
        f"from ncccu.{module} import {function}\n\n"
        f"CONFIG = {pformat(args, sort_dicts=False)}\n\n"
        "if __name__ == '__main__':\n"
        f"    {function}(**CONFIG)\n"
    )


def package_submission(
    *, team_name: str, training_run: Path, model: Path, prediction_run: Path,
    submission: Path, pretrain_model: Path, licenses: tuple[Path, ...],
    cost_script: Path, declaration: Path | None, output_dir: Path,
) -> Path:
    if not team_name.strip() or any(char in team_name for char in '<>:"/\\|?*'):
        raise ValueError("队伍名不能为空或包含文件名非法字符")
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(ROOT / "outputs"):
        raise ValueError("打包目录必须位于仓库 outputs 下")
    suffix = "最终提交材料" if declaration is not None else "待核对_缺核验表"
    target = output_dir / f"{team_name}_{suffix}.zip"
    if target.exists():
        raise FileExistsError(f"目标 ZIP 已存在：{target}")
    train_args_path = training_run / "args.json"
    predict_args_path = prediction_run / "args.json"
    test_csv = ROOT / "datasets/test.csv"
    files = {
        f"model/{model.name}": model,
        f"pretrain_model/{pretrain_model.name}": pretrain_model,
        "cost.py": cost_script,
        "submission.csv": submission,
        "code/pyproject.toml": ROOT / "pyproject.toml",
        "code/uv.lock": ROOT / "uv.lock",
        "code/README.md": ROOT / "README.md",
    }
    if declaration is not None:
        files["模型核验材料与数据申明清单.xlsx"] = declaration
    if not licenses or len({path.name for path in licenses}) != len(licenses):
        raise ValueError("须提供许可证全文，文件名不能重复")
    files.update({f"licenses/{path.name}": path for path in licenses})
    files.update({f"code/src/ncccu/{path.name}": path for path in (ROOT / "src/ncccu").glob("*.py")})
    required = [*files.values(), train_args_path, predict_args_path, test_csv]
    missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError("缺少以下文件或文件为空：\n" + "\n".join(f"  - {path}" for path in missing))
    if model.stat().st_size > 500_000_000:
        raise ValueError("最终推理权重解压后超过 500 MB")
    if declaration is not None:
        if declaration.suffix.lower() != ".xlsx" or not is_zipfile(declaration):
            raise ValueError("核验材料须为已填写的官方 XLSX 文件")
        with ZipFile(declaration) as workbook:
            if "xl/workbook.xml" not in workbook.namelist():
                raise ValueError("核验材料缺少 Excel 工作簿内容")

    train_args, predict_args = read_args(train_args_path), read_args(predict_args_path)
    if train_args.get("smoke_test") or train_args.get("resume") or not train_args.get("pretrained"):
        raise ValueError("请选择从公开预训练权重开始的正式训练记录")
    if (ROOT / train_args["model"]).resolve() != pretrain_model.resolve():
        raise ValueError("所选预训练权重与训练 args.json 不一致")
    if predict_args.get("smoke_test") or predict_args.get("split") != "test":
        raise ValueError("预测记录须来自正式测试集推理")
    if (ROOT / predict_args["model"]).resolve() != model.resolve():
        raise ValueError("所选最终权重与预测 args.json 不一致")
    validate_submission(submission, test_csv)

    # 包内入口读取归档中的权重，官方数据由复核方放到解压根目录的 datasets。
    train_args.update(model=f"../pretrain_model/{pretrain_model.name}", data="../datasets",
                      project=None, save_dir=None)
    predict_args.update(model=f"../model/{model.name}", data="../datasets", source="../datasets/test.csv",
                        project=None, name=None, save_dir=None, save=False, save_txt=False, show=False,
                        export_options={"enabled": False})
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    dependencies = {"torch", "torchvision", "ultralytics"}
    requirements = "".join(
        f"{item['name']}=={item['version'].split('+')[0]}\n"
        for item in lock["package"] if item["name"] in dependencies
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    # 在同一目录先完成 ZIP，再用排他创建保护已有提交包。
    with pretrain_model.open("rb") as file:
        pretrain_hash = hashlib.file_digest(file, "sha256").hexdigest()
    with model.open("rb") as file:
        model_hash = hashlib.file_digest(file, "sha256").hexdigest()
    with tempfile.TemporaryDirectory(dir=output_dir) as scratch:
        temporary = Path(scratch) / "submission.zip"
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
            for name, path in files.items():
                archive.write(path, name)
            archive.writestr("code/train.py", entry_script("training", "train_multilabel", train_args))
            archive.writestr("code/predict.py", entry_script("inference", "predict_multilabel", predict_args))
            archive.writestr("code/requirements.txt", requirements)
            if declaration is None:
                archive.writestr("待填写核验表.md", (
                    "此包缺少已填写的官方 XLSX 核验表。\n"
                    f"\n- 预训练权重：{pretrain_model.name}\n- 预训练 SHA-256：{pretrain_hash}\n"
                    f"- 推理权重：{model.name}\n- 推理 SHA-256：{model_hash}\n"
                    f"- 推理精度：FP{predict_args['quantize']}\n"
                    "- 官方模板及规则：https://www.ncccu.org.cn/index/Paper/case1.html\n"
                ))
        # os.link 原子发布，目标存在时失败，且临时包和目标位于同一卷。
        target.hardlink_to(temporary)

    print(f"预训练权重 SHA-256：{pretrain_hash}")
    print(f"打包完成：{target}")
    return target
