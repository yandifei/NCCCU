# NCCCU 胸片多标签分类

本项目对官方胸片做十类多标签分类，输出每次检查的十个独立概率。当前版本使用 ConvNeXt-Small，从公开 ImageNet 权重初始化，并在官方训练集微调。验证按 Study 聚合，用十类 Macro-AUC 选模。

| 入口 | 用途 |
| --- | --- |
| `train.py` | 训练并保存最佳模型，支持从本次的 last.pt 续训 |
| `predict.py` | 读取最佳权重，预测测试集并调用打包 |
| `push.py` | 使用已有预测目录重新打包 |
| `cost.py` | 在解压后的提交包内做单图测速和一致性核验 |

参数直接在各入口中修改。按以下顺序运行：

```powershell
uv sync --locked
uv run --no-sync python tools/check_dataset.py
uv run --no-sync python train.py
uv run --no-sync python predict.py
```

目录用途见 [仓库指南](AGENTS.md)。预训练权重下载、训练参数、续训、验证和版本差异见 [训练与推理](docs/训练与推理说明.md)。提交材料说明见 [赛事提交打包](docs/赛事提交打包.md)。

项目代码采用 [AGPL-3.0](LICENSE.txt)。所用权重的模型卡及许可证在训练说明中单独记录。
