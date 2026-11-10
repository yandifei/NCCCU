"""检查并打包 NCCCU 最终提交材料。"""

import csv
import hashlib
import json
import tempfile
import tomllib
from pathlib import Path
from pprint import pformat
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, is_zipfile


ROOT = Path(__file__).resolve().parents[1]
CLASS_IDS = tuple(range(1, 11))
SUBMISSION_COLUMNS = {"Study_id", "Predict_class", "Probability", "Gate"}
DEPENDENCIES = {"torch", "torchvision", "timm", "safetensors", "numpy", "pillow", "matplotlib", "tqdm"}

StudyKey = tuple[str, str]
RunConfig = dict[str, Any]


def _test_studies(test_csv: Path) -> set[StudyKey]:
    """读取检查集合；同一检查的多张影像允许重复出现。"""
    with test_csv.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        if not {"Subject_id", "Study_id"}.issubset(fields) or len(fields) != len(set(fields)):
            raise ValueError("官方 test.csv 缺少检查标识列或存在重复列名")
        studies: set[StudyKey] = set()
        for line, row in enumerate(reader, start=2):
            subject, study = row["Subject_id"], row["Study_id"]
            if None in row or not subject or not study:
                raise ValueError(f"官方 test.csv 第 {line} 行的检查标识或列数无效")
            studies.add((subject, study))
    if not studies:
        raise ValueError("官方 test.csv 没有检查记录")
    return studies


def validate_submission(submission: Path, test_csv: Path) -> None:
    studies = _test_studies(test_csv)
    with submission.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        fields = set(columns)
        if len(columns) != len(fields):
            raise ValueError("submission.csv 存在重复列名")
        if not SUBMISSION_COLUMNS <= fields:
            raise ValueError("submission.csv 缺少 Study_id、Predict_class、Probability 或 Gate")
        has_subject = "Subject_id" in fields
        study_ids = {study for _, study in studies}
        if not has_subject and len(study_ids) != len(studies):
            raise ValueError("官方测试集中存在跨患者重复的 Study_id，CSV 须包含 Subject_id")
        expected_studies = studies if has_subject else {("", study) for study in study_ids}
        expected = {(subject, study, label) for subject, study in expected_studies for label in CLASS_IDS}
        seen: set[tuple[str, str, int]] = set()
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


def read_args(path: Path) -> RunConfig:
    """运行记录保持普通字典，同时在边界拒绝损坏的 JSON 结构。"""
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"运行参数须为 JSON 对象：{path}")
    return config


def entry_script(module: str, function: str, args: RunConfig) -> str:
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


def _submission_target(team_name: str, output_dir: Path, declaration: Path | None) -> Path:
    if not team_name.strip() or any(char in team_name for char in '<>:"/\\|?*'):
        raise ValueError("队伍名不能为空或包含文件名非法字符")
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("打包目录必须位于仓库 outputs 下")
    suffix = "最终提交材料" if declaration is not None else "待核对_缺核验表"
    target = output_dir / f"{team_name}_{suffix}.zip"
    if target.exists():
        raise FileExistsError(f"目标 ZIP 已存在：{target}")
    return target


def _archive_files(
    model: Path, pretrain_model: Path, submission: Path, licenses: tuple[Path, ...],
    cost_script: Path, declaration: Path | None,
) -> dict[str, Path]:
    """明确列出交付文件，避免将数据集、缓存或训练断点带入归档。"""
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
    return files


def _validate_files(paths: list[Path], model: Path, declaration: Path | None) -> None:
    missing = [path for path in paths if not path.is_file() or path.stat().st_size == 0]
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



def _model_path(config: RunConfig, source: Path) -> Path:
    value = config.get("model")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"运行参数缺少有效的 model 路径：{source}")
    return (ROOT / value).resolve()


def _archive_configs(
    train_args_path: Path, predict_args_path: Path, model: Path, pretrain_model: Path, model_hash: str,
) -> tuple[RunConfig, RunConfig]:
    """核对权重来源后，仅为包内入口改写路径，保留其他实际运行参数。"""
    train_args, predict_args = read_args(train_args_path), read_args(predict_args_path)
    if train_args.get("smoke_test") or train_args.get("resume") or pretrain_model.suffix != ".safetensors":
        raise ValueError("请选择从公开 safetensors 权重开始的正式 ConvNeXt 训练记录")
    if _model_path(train_args, train_args_path) != pretrain_model.resolve():
        raise ValueError("所选预训练权重与训练 args.json 不一致")
    if predict_args.get("smoke_test") or predict_args.get("split") != "test":
        raise ValueError("预测记录须来自正式测试集推理")
    if _model_path(predict_args, predict_args_path) != model.resolve():
        raise ValueError("所选最终权重与预测 args.json 不一致")
    if predict_args.get("quantize") not in (16, 32):
        raise ValueError("预测 args.json 的 quantize 须为 16 或 32")
    recorded_hash = predict_args.pop("model_sha256", None)
    if recorded_hash is not None and recorded_hash != model_hash:
        raise ValueError("最终权重 SHA-256 与预测记录不一致，请选择生成该预测时的权重")

    # 包内入口读取归档中的权重，官方数据由复核方放到解压根目录的 datasets。
    train_args.update(model=f"../pretrain_model/{pretrain_model.name}", data="../datasets", name=None)
    predict_args.update(model=f"../model/{model.name}", data="../datasets", name=None)
    return train_args, predict_args


def _requirements() -> str:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    return "".join(
        f"{item['name']}=={item['version'].split('+')[0]}\n"
        for item in lock["package"] if item["name"] in DEPENDENCIES
    )


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _publish_archive(
    target: Path, files: dict[str, Path], texts: dict[str, str], hashes: dict[str, str],
) -> None:
    """先完成临时 ZIP，再在同一卷上排他发布，失败时不留下半成品。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as scratch:
        temporary = Path(scratch) / "submission.zip"
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
            for name, path in files.items():
                archive.write(path, name)
            for name, content in texts.items():
                archive.writestr(name, content)
        # 校验实际归档的字节，防止训练同时更新权重时发布不匹配的包。
        with ZipFile(temporary) as archive:
            for name, expected_hash in hashes.items():
                with archive.open(name) as file:
                    if hashlib.file_digest(file, "sha256").hexdigest() != expected_hash:
                        raise ValueError(f"归档期间权重内容发生变化：{files[name]}")
        # os.link 原子发布，目标存在时失败，且临时包和目标位于同一卷。
        target.hardlink_to(temporary)


def package_submission(
    *, team_name: str, training_run: Path, model: Path, prediction_run: Path,
    submission: Path, pretrain_model: Path, licenses: tuple[Path, ...],
    cost_script: Path, declaration: Path | None, output_dir: Path,
) -> Path:
    target = _submission_target(team_name, output_dir, declaration)
    train_args_path = training_run / "args.json"
    predict_args_path = prediction_run / "args.json"
    test_csv = ROOT / "datasets/test.csv"
    files = _archive_files(model, pretrain_model, submission, licenses, cost_script, declaration)
    _validate_files([*files.values(), train_args_path, predict_args_path, test_csv], model, declaration)
    validate_submission(submission, test_csv)
    pretrain_hash, model_hash = _sha256(pretrain_model), _sha256(model)
    train_args, predict_args = _archive_configs(
        train_args_path, predict_args_path, model, pretrain_model, model_hash,
    )
    texts = {
        "code/train.py": entry_script("training", "train_multilabel", train_args),
        "code/predict.py": entry_script("inference", "predict_multilabel", predict_args),
        "code/requirements.txt": _requirements(),
    }
    if declaration is None:
        texts["待填写核验表.md"] = (
            "此包缺少已填写的官方 XLSX 核验表。\n"
            f"\n- 预训练权重：{pretrain_model.name}\n- 预训练 SHA-256：{pretrain_hash}\n"
            f"- 推理权重：{model.name}\n- 推理 SHA-256：{model_hash}\n"
            f"- 推理精度：FP{predict_args['quantize']}\n"
            "- 官方模板及规则：https://www.ncccu.org.cn/index/Paper/case1.html\n"
        )
    _publish_archive(target, files, texts, {
        f"model/{model.name}": model_hash,
        f"pretrain_model/{pretrain_model.name}": pretrain_hash,
    })
    print(f"预训练权重 SHA-256：{pretrain_hash}")
    print(f"打包完成：{target}")
    return target
