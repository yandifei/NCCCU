from ncccu.inference import predict_multilabel

# 关键点：保留入口保护，避免 Windows 创建数据加载进程时递归执行
if __name__ == '__main__':
    # 将 model 改为你训练生成的 best.pt，直接运行本文件即可预测
    predict_multilabel(
        model=r"./outputs/train/你的训练目录/weights/best.pt",  # 本项目训练的多标签权重
        data=r"./datasets",  # 官方 CSV 和影像所在目录
        split="test",  # 预测划分，可选 train、valid、test
        batch=16,  # 推理批次大小，根据显存调整
        device=0,  # PyTorch 的第一张显卡；使用 CPU 时改为 "cpu"
        workers=0,  # 数据加载进程数
        gate=0.5,  # 写入提交表的分类阈值，全部 10 类概率都会输出
        name=None,  # outputs/predict/ 下的目录名；None 自动使用时间戳
        smoke_test=False,  # 使用冒烟权重验证流程时改为 True
    )
