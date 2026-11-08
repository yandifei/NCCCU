from pathlib import Path

from tools.package_submission import package_submission


if __name__ == "__main__":
    root = Path(__file__).resolve().parent

    # 按实际参赛信息修改；选择历史最高分对应的模型和预测目录。
    team_name = "队伍名"
    training_run = root / "outputs/train/v2_512"
    model = training_run / "weights/best.pt"
    prediction_run = root / "outputs/predict/ncccu_cls_v2_512_test"
    submission = prediction_run / "submission.csv"

    # 合规材料：填写官方表格，并提供与提交预测使用同一推理逻辑的测速脚本。
    pretrain_model = root / "pretrain_model/yolo26x-cls.pt"
    licenses = (root / "LICENSE.txt",)
    cost_script = root / "cost.py"
    declaration = root / "outputs/合规材料/模型核验材料与数据申明清单.xlsx"
    output_dir = root / "outputs/submission"

    try:
        package_submission(
            team_name=team_name,
            training_run=training_run,
            model=model,
            prediction_run=prediction_run,
            submission=submission,
            pretrain_model=pretrain_model,
            licenses=licenses,
            cost_script=cost_script,
            declaration=declaration,
            output_dir=output_dir,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"打包失败：{error}") from None
