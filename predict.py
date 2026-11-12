"""第六版预测入口，聚合检查的全部视图，保存概率并自动打包赛事材料。"""

from ncccu.inference import predict_multilabel
from push import package_prediction


# Windows 工作进程需要入口保护，避免重复加载模型和启动推理。
if __name__ == "__main__":
    prediction_run = predict_multilabel(
        model="./outputs/train/v6_convnext_small_640/weights/best.pt",  # 修改训练 name 后同步选择对应 best.pt。
        data="./datasets",  # 官方 CSV 和影像的根目录。
        split="test",  # test 自动打包，valid 仅输出验证集预测。
        device=0,  # PyTorch 中的 CUDA 设备编号。
        batch=1,  # 与赛事 cost.py 的单图前向一致。
        workers=4,  # 读取影像的工作进程数。
        quantize=32,  # 32 为 FP32；16 为 CUDA 混合精度。
        gate=0.5,  # 新模型应在官方 valid 上重新校准。
        channels_last=True,  # 使用 channels-last 内存布局。
        augment=False,  # 开启后对原图与水平翻转图的概率取均值。
        seed=0,  # 固定随机种子。
        deterministic=True,  # 启用 PyTorch 确定性计算。
        name=None,  # None 自动生成目录名；也可指定未使用的名称。
        smoke_test=False,  # 仅验证前 16 次检查，产物不能用于正式提交。
    )
    try:
        package_prediction(prediction_run)
    except (OSError, ValueError) as error:
        raise SystemExit(f"预测已保存到 {prediction_run}，打包失败：{error}") from None
