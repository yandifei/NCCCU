from ncccu.training import train_multilabel

# 关键点：保留入口保护，避免 Windows 创建数据加载进程时递归执行
if __name__ == '__main__':
    # 直接修改下面的参数，然后运行本文件
    train_multilabel(
        model=r"./pretrain_model/yolo26n-cls.pt",  # 初始分类预训练模型，路径相对项目根目录
        resume=False,  # 续训时改为 True，并将 model 改为上次的 weights/last.pt
        data=r"./datasets",  # 官方 CSV 和影像所在目录
        epochs=100,  # 新训练的总轮数；续训沿用检查点中原定的总轮数
        imgsz=224,  # 输入尺寸，等比例缩放并填充，保留完整胸片
        patience=20,  # 验证 Macro-AUC 连续多少轮未提升时停止，0 表示关闭早停
        batch=16,  # 每批影像数，根据显存调整
        lr0=0.001,  # AdamW 初始学习率
        lrf=0.01,  # 余弦调度结束时，学习率相对初始值的比例
        weight_decay=0.0005,  # 权重衰减
        dropout=0.0,  # 分类头的随机丢弃比例
        amp=True,  # CUDA 上开启混合精度训练
        device=0,  # PyTorch 的第一张显卡；使用 CPU 时改为 "cpu"
        workers=0,  # 数据加载进程数，Windows 可按需要改为 3
        seed=0,  # 随机种子
        name=None,  # outputs/train/ 下的目录名；None 自动使用时间戳，续训也创建新目录
        smoke_test=False,  # True 仅取前 16 次完整检查验证流程
    )
