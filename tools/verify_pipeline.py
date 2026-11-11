"""用官方数据的前 16 次检查，验证真实 GPU 训练、预测和入口衔接。"""

import json
import runpy
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import torch

from ncccu.inference import predict_multilabel
from ncccu.runtime import ROOT
from ncccu.training import train_multilabel


def main() -> None:
    """沿用入口中的模型和参数，仅缩短轮数并切换到冒烟验证模式。"""
    name = f"pipeline_smoke_{datetime.now():%Y%m%d_%H%M%S}"
    training_run: Path | None = None
    prediction_run: Path | None = None
    torch.cuda.reset_peak_memory_stats()

    def train_smoke(**parameters: object) -> Path:
        nonlocal training_run
        parameters.update(name=name, epochs=2, warmup_epochs=1, smoke_test=True, resume=False)
        training_run = train_multilabel(**parameters)
        return training_run

    def predict_smoke(**parameters: object) -> Path:
        nonlocal prediction_run
        if training_run is None:
            raise RuntimeError("须先完成冒烟训练")
        parameters.update(
            model=training_run / "weights/best.pt", split="valid", name=name, smoke_test=True
        )
        prediction_run = predict_multilabel(**parameters)
        return prediction_run

    # 与直接启动根目录脚本保持相同的导入路径，让 predict.py 能导入 push.py。
    with patch.object(sys, "path", [str(ROOT), *sys.path]):
        with patch("ncccu.training.train_multilabel", train_smoke):
            runpy.run_path(str(ROOT / "train.py"), run_name="__main__")
        with patch("ncccu.inference.predict_multilabel", predict_smoke):
            runpy.run_path(str(ROOT / "predict.py"), run_name="__main__")
    print(json.dumps({
        "training_run": str(training_run),
        "prediction_run": str(prediction_run),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
