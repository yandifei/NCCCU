from ncccu.training import train_multilabel


# Windows 工作进程需要入口保护。
if __name__ == "__main__":
    train_multilabel(
        model="./pretrain_model/convnext_small.fb_in22k_ft_in1k.safetensors",
        data="./datasets",
        name="v3_convnext_small_512",
        imgsz=512,
        epochs=40,
        patience=12,
        batch=8,  # 每 16 个微批累积一次，名义更新批量为 128。
        nbs=128,
        workers=6,
        device=0,
        lr0=0.0001,
        lrf=0.01,
        warmup_epochs=3,
        weight_decay=0.0005,
        dropout=0.2,
        brightness=0.1,
        amp=True,
        seed=0,
        deterministic=True,
        channels_last=True,
        resume=False,  # 续训时改为 True，并将 model 指向该次训练的 weights/last.pt。
        smoke_test=False,
    )
