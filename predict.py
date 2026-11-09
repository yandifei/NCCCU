from ncccu.inference import predict_multilabel
from push import package_prediction


if __name__ == "__main__":
    prediction_run = predict_multilabel(
        model="./outputs/train/v3_convnext_small_512/weights/best.pt",
        data="./datasets",
        split="test",
        device=0,
        batch=1,  # 与赛事 cost.py 的单图前向一致。
        workers=4,
        quantize=32,
        gate=0.5,  # 新模型应在官方 valid 上重新校准。
        channels_last=True,
        augment=False,
        seed=0,
        deterministic=True,
        name=None,
        smoke_test=False,
    )
    try:
        package_prediction(prediction_run)
    except (OSError, ValueError) as error:
        raise SystemExit(f"预测已保存到 {prediction_run}，打包失败：{error}") from None
