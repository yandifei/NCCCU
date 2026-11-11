"""第四版训练入口，使用 FP32 和较低学习率微调 ConvNeXt-Small。"""

from ncccu.training import train_multilabel


# Windows 工作进程需要入口保护。
if __name__ == "__main__":
    train_multilabel(
        model="./pretrain_model/convnext_small.fb_in22k_ft_in1k.safetensors",  # ImageNet 初始化权重。
        data="./datasets",  # 包含官方 CSV 和影像的数据目录。
        name="v4_fp32_lr3e5_512",  # 输出至 outputs/train 下的独立运行目录。
        imgsz=512,  # 保持宽高比缩放并填充后的输入边长。
        epochs=16,  # 训练上限，同时决定余弦学习率的衰减周期。
        patience=6,  # 验证 Macro-AUC 连续 6 轮无提升时早停。
        batch=8,  # 每 16 个微批累积一次，名义更新批量为 128。
        nbs=128,  # 名义更新批量；实际每组累积 ceil(nbs / batch) 个微批。
        workers=6,  # 数据加载工作进程数量。
        device=0,  # CUDA 设备索引；CPU 验证时可改为 "cpu"。
        lr0=0.00003,  # 降低微调学习率，减小预训练参数的每步更新幅度。
        lrf=0.01,  # 余弦调度的最终学习率比例。
        warmup_epochs=1,  # 第 1 轮线性预热，随后按 16 轮周期余弦衰减。
        weight_decay=0.0005,  # AdamW 权重衰减。
        dropout=0.2,  # 分类头 Dropout 概率。
        brightness=0.1,  # 训练亮度扰动幅度；0 表示关闭。
        amp=False,  # 关闭混合精度，使用 FP32 训练。
        seed=0,  # 每轮种子在此基础上递增，便于确定性恢复。
        deterministic=True,  # 强制确定性算法，并由运行配置关闭 TF32。
        channels_last=True,  # 使用 channels-last 内存布局。
        resume=False,  # 续训时改为 True，并将 model 指向该次训练的 weights/last.pt。
        smoke_test=False,  # 仅取各划分前 16 个完整 Study 做冒烟验证。
    )
