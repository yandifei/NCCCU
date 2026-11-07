"""多标签推理的可视化、精度及导出配置。"""
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from ncccu.multilabel import CLASS_NAMES, logits
from ncccu.runtime import validate_fixed

PREDICT_FIXED = {
    "conf": None, "iou": None, "rect": False, "dnn": False, "tile": False,
    "max_det": 0, "vid_stride": 1, "stream_buffer": False, "visualize": False,
    "agnostic_nms": False, "classes": None, "retina_masks": False, "embed": None,
    "stream": False, "compile": False, "nms": False, "save_frames": False,
    "save_crop": False, "show_boxes": False, "line_width": None,
    "exist_ok": False, "task": "classify",
}
PLOT_DEFAULTS = {
    "conf": True, "line_width": None, "font_size": 16, "font": "default",
    "pil": True, "img": None, "kpt_radius": 0, "kpt_line": False, "labels": True,
    "boxes": False, "masks": False, "probs": True, "show": False, "save": True,
    "filename": None, "color_mode": "class", "txt_color": (255, 255, 255),
}
EXPORT_DEFAULTS = {
    "enabled": False, "format": "torchscript", "imgsz": None, "batch": 1,
    "dynamic": False, "quantize": 32, "simplify": False, "opset": None,
    "device": None, "data": None, "fraction": 1.0, "workspace": None,
    "nms": False, "conf": None, "iou": None, "name": "model", "optimize": False,
}


def option_dict(options: dict | None, defaults: dict) -> dict:
    values = dict(defaults)
    if options:
        unknown = options.keys() - defaults.keys()
        if unknown:
            raise TypeError(f"未知配置 {sorted(unknown)}")
        values.update(options)
    return values


def validate_plot(options: dict) -> None:
    validate_fixed({key: options[key] for key in (
        "line_width", "pil", "img", "kpt_radius", "kpt_line", "boxes",
        "masks", "filename", "color_mode",
    )}, PLOT_DEFAULTS)
    if options["font_size"] < 1 or len(options["txt_color"]) != 3:
        raise ValueError("font_size 为正数，txt_color 为三个颜色分量")


def render_predictions(dataset, probabilities, run: Path, args) -> None:
    options = args.plot_options
    validate_plot(options)
    if not (args.save or args.show or args.save_txt):
        return
    font = None
    if args.save or args.show:
        font = (ImageFont.load_default(size=options["font_size"]) if options["font"] == "default"
                else ImageFont.truetype(options["font"], options["font_size"]))
    for row, values in zip(dataset.rows, probabilities):
        relative = Path(row["Subject_id"]) / row["Study_id"] / row["Dicom_id"]
        if args.save_txt:
            path = run / "labels" / relative.with_suffix(".txt")
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for index, value in enumerate(values):
                if value >= args.gate:
                    lines.append(f"{index+1} {float(value):.8f}" if args.save_conf else str(index+1))
            path.write_text("\n".join(lines)+"\n", encoding="utf-8")
        if args.save or args.show:
            with Image.open(dataset.image_path(row)) as source:
                image = source.convert("RGB")
                image.thumbnail((1024, 1024))
            panel_height = (options["font_size"]+5)*10+10
            canvas = Image.new("RGB", (max(image.width, 450), image.height+panel_height))
            canvas.paste(image)
            draw = ImageDraw.Draw(canvas)
            if options["probs"]:
                for index, value in enumerate(values):
                    text = []
                    if args.show_labels and options["labels"]:
                        text.append(CLASS_NAMES[index])
                    if args.show_conf and options["conf"]:
                        text.append(f"{float(value):.4f}")
                    draw.text((5, image.height+5+index*(options["font_size"]+5)),
                              " ".join(text), fill=tuple(reversed(options["txt_color"])), font=font)
            if args.save and options["save"]:
                path = run / "images" / relative.with_suffix(".jpg")
                path.parent.mkdir(parents=True, exist_ok=True)
                canvas.save(path)
            if args.show and options["show"]:
                canvas.show()


class ProbabilityModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return logits(self.model, images).sigmoid()


def validate_export(options: dict) -> None:
    if options["enabled"]:
        keys = (
            "format", "dynamic", "quantize", "simplify", "opset", "data",
            "fraction", "workspace", "nms", "conf", "iou", "optimize",
        )
        validate_fixed({key: options[key] for key in keys}, EXPORT_DEFAULTS)


def export_model(model, run: Path, imgsz: int, device: torch.device, options: dict) -> None:
    if not options["enabled"]:
        return
    fixed = {key: options[key] for key in (
        "format", "dynamic", "quantize", "simplify", "opset", "data", "fraction",
        "workspace", "nms", "conf", "iou", "optimize",
    )}
    validate_fixed(fixed, EXPORT_DEFAULTS)
    if options["imgsz"] not in (None, imgsz) or options["device"] not in (None, str(device)):
        raise ValueError("导出尺寸和设备须与当前预测一致")
    name = options["name"]
    if not name or Path(name).name != name or any(c in name for c in "/\\:") or name in {".", ".."}:
        raise ValueError("导出 name 必须为单层文件名")
    if options["batch"] < 1:
        raise ValueError("导出 batch 必须为正整数")
    example = torch.zeros(options["batch"], 3, imgsz, imgsz, device=device)
    wrapper = ProbabilityModel(model.eval())
    with torch.inference_mode():
        exported = torch.jit.trace(wrapper, example)
        torch.testing.assert_close(exported(example), wrapper(example))
        probe = torch.linspace(0, 1, example.numel(), device=device).reshape_as(example)
        torch.testing.assert_close(exported(probe), wrapper(probe))
        exported.save(str(run / f"{name}.torchscript"))
