"""训练配置对应的优化器和图像增强。"""
import math

import torch
from torchvision import transforms as T

TRAIN_FIXED = {
    "exist_ok": False, "cls_remap": False, "single_cls": False, "classes": None,
    "rect": False, "multi_scale": 0.0, "close_mosaic": 0, "quantize": None,
    "fraction": 1.0, "profile": False, "distill_model": None, "dis": 0.0,
    "box": 0.0, "dfl": 0.0, "pose": 0.0, "kobj": 0.0, "rle": 0.0,
    "angle": 0.0, "dlog": 0.0, "dgrad": 0.0, "dlam": 0.0,
    "overlap_mask": False, "mask_ratio": 1, "val": True, "split": "valid",
    "nms": False, "max_det": 0, "tile": False, "compile": False,
    "degrees": 0.0, "translate": 0.0, "shear": 0.0, "perspective": 0.0,
    "bgr": 0.0, "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0,
    "copy_paste": 0.0, "copy_paste_mode": "flip", "augmentations": None,
    "trainer": None, "validator": None,
}


def training_transform(args):
    for name in ("fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v", "erasing", "scale"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"{name} 必须在 [0,1] 内")
    if args.hsv_h > 0.5 or args.scale == 1:
        raise ValueError("hsv_h<=0.5，scale<1")
    transforms = []
    if args.scale:
        transforms.append(T.RandomResizedCrop(args.imgsz, scale=(1-args.scale, 1.0), ratio=(1.0, 1.0)))
    if args.fliplr:
        transforms.append(T.RandomHorizontalFlip(args.fliplr))
    if args.flipud:
        transforms.append(T.RandomVerticalFlip(args.flipud))
    policies = {"randaugment": T.RandAugment, "autoaugment": T.AutoAugment, "augmix": T.AugMix}
    if args.auto_augment is not None:
        if args.auto_augment not in policies:
            raise ValueError("auto_augment 必须为 None、randaugment、autoaugment 或 augmix")
        if any((args.hsv_h, args.hsv_s, args.hsv_v)):
            raise ValueError("自动增强和 HSV 扰动请二选一，避免参数被忽略")
        transforms.append(policies[args.auto_augment]())
    elif any((args.hsv_h, args.hsv_s, args.hsv_v)):
        transforms.append(T.ColorJitter(brightness=args.hsv_v, saturation=args.hsv_s, hue=args.hsv_h))
    transforms.append(T.ToTensor())
    if args.erasing:
        transforms.append(T.RandomErasing(p=args.erasing))
    return T.Compose(transforms)


def make_optimizer(model, args):
    constructors = {
        "AdamW": torch.optim.AdamW, "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD, "RMSProp": torch.optim.RMSprop,
    }
    if args.optimizer not in constructors:
        raise ValueError("optimizer 显式选择 AdamW、Adam、SGD 或 RMSProp")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if args.warmup_bias_lr:
        parameters = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.endswith(".bias")], "is_bias": False},
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.endswith(".bias")], "is_bias": True},
        ]
    kwargs = dict(lr=args.lr0, weight_decay=args.weight_decay)
    if args.optimizer in {"AdamW", "Adam"}:
        kwargs["betas"] = (args.momentum, 0.999)
    else:
        kwargs["momentum"] = args.momentum
    return constructors[args.optimizer](parameters, **kwargs)


def set_warmup(optimizer, args, epoch: int, step: int, batches: int) -> None:
    position = epoch + step / batches
    progress = min(1.0, position / args.warmup_epochs) if args.warmup_epochs else 1.0
    factor = ((1+math.cos(math.pi*epoch/args.epochs))/2 if args.cos_lr else 1-epoch/args.epochs)
    target_lr = args.lr0 * (args.lrf + (1-args.lrf)*factor)
    for group in optimizer.param_groups:
        initial = args.warmup_bias_lr if group.get("is_bias", False) else 0.0
        group["lr"] = initial + (target_lr-initial)*progress
        momentum = args.warmup_momentum + (args.momentum-args.warmup_momentum)*progress
        if "betas" in group:
            group["betas"] = (momentum, group["betas"][1])
        else:
            group["momentum"] = momentum
