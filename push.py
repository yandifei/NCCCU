"""从已有预测目录收集运行记录并生成赛事提交包。"""

import argparse
from pathlib import Path
from typing import Sequence

from tools.package_submission import ROOT, package_submission, read_args


# 一次配置，之后每次预测自动沿用。核验表内容须与当前权重及推理精度对应。
TEAM_NAME = "队伍名"
DECLARATION = ROOT / "outputs/合规材料/模型核验材料与数据申明清单.xlsx"
LICENSES = (ROOT / "LICENSE.txt", ROOT / "docs/ConvNeXt-Apache-2.0.txt",
            ROOT / "docs/ConvNeXt-upstream-MIT.txt")
OUTPUT_DIR = ROOT / "outputs/submission"


def package_prediction(prediction_run: str | Path) -> Path | None:
    prediction_run = (ROOT / prediction_run).resolve()
    prediction = read_args(prediction_run / "args.json")
    if prediction.get("split") not in ("test", "valid"):
        raise ValueError("预测 args.json 的 split 须为 test 或 valid")
    if prediction["split"] != "test" or prediction.get("smoke_test"):
        print("仅正式测试集预测生成提交包。")
        return None
    model_name = prediction.get("model")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("预测 args.json 缺少有效的 model 路径")
    model = (ROOT / model_name).resolve()
    training_run = model.parent.parent if model.parent.name == "weights" else model.parent
    training = read_args(training_run / "args.json")
    pretrain_name = training.get("model")
    if not isinstance(pretrain_name, str) or not pretrain_name.strip():
        raise ValueError("训练 args.json 缺少有效的 model 路径")
    pretrain = ROOT / pretrain_name
    declaration = DECLARATION if DECLARATION.is_file() else None
    if declaration is None:
        print(f"缺少已填写的核验表，将生成待核对包：{DECLARATION}")
    elif TEAM_NAME == "队伍名":
        raise ValueError("生成最终包前，请在 push.py 填写实际队伍名")
    return package_submission(
        team_name=TEAM_NAME,
        training_run=training_run,
        model=model,
        prediction_run=prediction_run,
        submission=prediction_run / "submission.csv",
        pretrain_model=pretrain,
        licenses=LICENSES,
        cost_script=ROOT / "cost.py",
        declaration=declaration,
        output_dir=OUTPUT_DIR / prediction_run.name,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="使用已有预测目录重新打包，无需重新推理")
    parser.add_argument("prediction_run", help="包含 args.json 和 submission.csv 的预测目录")
    try:
        package_prediction(parser.parse_args(argv).prediction_run)
    except (OSError, ValueError) as error:
        raise SystemExit(f"打包失败：{error}") from None


if __name__ == "__main__":
    main()
