<div align="center">
  <img src="https://github.com/yandifei/ArisuQQChatAI/blob/main/%E5%B1%95%E7%A4%BA%E9%A1%B9%E7%9B%AE%E7%9A%84%E5%9B%BE%E7%89%87/%E7%88%B1%E4%B8%BD%E4%B8%9D.png?raw=true" width="170" alt="NCCCU 项目头像">
  <h1 align="center">NCCCU · 胸部 X 光片多标签分类</h1>
  <p>
    2026 年第八届全国高校计算机能力挑战赛 · 人工智能挑战赛 参赛工程
    <br/>
    检查（Study）级 × 10 类独立概率，主指标 <b>Macro-AUC</b>
  </p>
</div>

<div align="center">

![Python](https://img.shields.io/badge/python-3.15.2-3776AB?logo=python&logoColor=white&style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.15.0-EE4C2C?logo=pytorch&logoColor=white&style=flat-square)
![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.204-111111?logo=github&logoColor=white&style=flat-square)
![License](https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square)
![Deadline](https://img.shields.io/badge/deadline-2026--11--13%2023%3A59-critical?style=flat-square)
[![Stars](https://img.shields.io/github/stars/yandifei/NCCCU?style=social)](https://github.com/yandifei/NCCCU)

</div>

<div align="center">
🌟 如果这个仓库对你有帮助，点一下右上角的 Star 支持一下~
</div>

<br/>

## 📖 项目简介

胸片是筛查肺部疾病最常用、最经济的影像手段，但人工阅片主观性强、耗时耗力。本届赛题要求基于官方标注的胸片数据，做**多标签**疾病分类：一张片子可以同时存在多种病变，模型需要对 10 个类别**各自独立**输出概率，而不是从 10 类里选一个。

本仓库是完整的参赛工程：数据清点与标签转换、数据集配置、多标签训练链路、检查级聚合与提交产物，全部在此沉淀，文档统一收在 [`docs/`](docs/)。

> ⚠️ 关键认知：赛题文案里的「检测与分类」是医学语境的泛称，**实际评分只考察分类**——没有 bbox / mask 标注，提交格式与指标都是「检查级 × 10 类概率」。判定过程见 [赛题技术分析](docs/比赛资料/赛题技术分析-YOLO任务类型判断.md)。

## 🎯 赛题速览

| 项目 | 内容 |
| --- | --- |
| 任务 | 胸部 X 光片多标签分类（10 类，含「无异常发现」） |
| 数据 | 训练 18 513 / 验证 859 / 测试 1 414 次检查（对应 30 344 / 1 422 / 2 218 张影像） |
| 主指标 | 宏平均 AUC（Macro-AUC），副指标 Macro-F1 仅在主指标相等时二次排序 |
| 提交物 | `submission.csv`（每检查 × 10 行）+ 合规材料 zip |
| 提交次数 | 每日最多 2 次，全程仅 1 次撤销机会 |
| 硬约束 | 模型解压后 ≤ 500 MB；单图端到端 ≤ 100 ms（batch=1）；**仅可用官方数据** |
| 预训练 | 仅允许**非医学领域**开源权重（ImageNet / COCO / LAION 等） |
| 截止 | 2026-11-13 23:59（北京时间） |

## 🏷 十个类别

| id | 中文 | 英文 |
| --- | --- | --- |
| 1 | 纵隔心影增大 | Enlarged Cardiomediastinum |
| 2 | 气胸 | Pneumothorax |
| 3 | 肺部实变 | Consolidation |
| 4 | 肺炎 | Pneumonia |
| 5 | 肺水肿 | Edema |
| 6 | 心影扩大 | Cardiomegaly |
| 7 | 肺不张 | Atelectasis |
| 8 | 肺部阴影 | Lung Opacity |
| 9 | 胸腔积液 | Pleural Effusion |
| 10 | 无异常发现 | No Finding |

官方 id 是 `1–10`，代码内部统一用 `0–9` 索引，映射写在 [`datasets/data.yaml`](datasets/data.yaml) 的 `official_id_to_index` 里。

## 🧠 技术方案

```text
官方 CSV (Predict_class，分号多标签)
        │  tools/csv2txt.py
        ▼
datasets/labels/**/*.txt  ──►  自定义 Dataset（10 维 multi-hot）
                                      │
                                      ▼
                 yolo26*-cls 骨干（ImageNet 权重，非医学，合规）
                                      │  分类头改 10 维 + 去掉 softmax
                                      ▼
                 BCEWithLogits / ASL 损失  ──►  按 Macro-AUC 选模
                                      │
                                      ▼
             逐图 sigmoid → 按 Study 聚合（mean / max）→ submission.csv
```

与原生 `yolo*-cls` 的差异决定了不能「一个 `model.train(data=...)` 就跑」：

| 维度 | 官方 cls 默认 | 本题要求 |
| --- | --- | --- |
| 数据组织 | ImageFolder `train/<类名>/<图>`，一图一标签 | `subject/study/image` 三级目录 + 多标签标注表，必须自写 Dataset |
| 标签与损失 | 单标签 softmax + 交叉熵 | 10 类独立 sigmoid + BCE / ASL |
| 评估 | top-1 / top-5 accuracy | Macro-AUC，验证与早停都按它来 |
| 输入尺寸 | 224 方形裁剪 | 胸片长宽比大，建议 letterbox / 384–512 |
| 输出 | `result.probs`（softmax 向量） | 10 维概率，自行写 csv |

工程要点：检查级聚合只对同一 Study 的多张图做确定性均值/取最大（规则明确允许），**不得**用测试集统计量拟合任何参数；TTA 与多模型集成允许，但同样受 500 MB / 100 ms 约束。

## 📁 目录结构

```text
NCCCU/
├── src/ncccu/            # 可复用代码：数据层、模型头改造、损失、评估
├── tools/
│   ├── csv2txt.py        # 官方 CSV → 逐图 txt 标签 + manifest
│   └── README.md         # 工具指引（文档统一收在 docs/）
├── datasets/             # 官方数据（不入库）：影像、CSV、派生标签、data.yaml
├── pretrain_model/       # 非医学预训练权重 yolo26n/s/m/l/x-cls.pt（不入库）
├── model/                # 本地训练产出（不入库）
├── docs/                 # 全部文档：赛制、数据分析、参数参考
├── train.py              # 训练入口（🚧 待实现）
├── predict.py            # 推理入口 / 环境自检
├── pyproject.toml        # uv 依赖与构建配置
└── AGENTS.md             # 仓库协作与代码规范
```

## 🚀 快速开始

### 1. 环境

需要 Python 3.15.2、[uv](https://docs.astral.sh/uv/)；训练需要 NVIDIA GPU。

```bash
uv sync                                   # 安装并锁定依赖
uv run --no-sync python predict.py        # 自检：PyTorch 版本、CUDA、GPU 型号
```

> ⚠️ 装完先看 `predict.py` 的输出。本地 `.venv` 若被解析成 `2.15.0+cpu`，CUDA 会显示不可用，训练前务必确认是 GPU 版本。

### 2. 数据

把官方数据放到 `datasets/`，保留原始影像与官方 CSV，不要增删样本：

```text
datasets/
├── train.csv / valid.csv / test.csv      # test.csv 无 Predict_class 列
├── train/{Subject_id}/{Study_id}/{Dicom_id}.jpg
├── valid/{Subject_id}/{Study_id}/{Dicom_id}.jpg
├── test/{Subject_id}/{Study_id}/{Dicom_id}.jpg
└── data.yaml                             # 本项目自定义配置
```

### 3. 生成逐图标签

```bash
uv run python tools/csv2txt.py --dry-run --limit 100   # 先小批试跑，不写文件
uv run python tools/csv2txt.py                         # 生成 train + valid
```

产出 `datasets/labels/`：每图一个 `.txt`（如 `2 4 5 8`）、`classes.txt`、`train_manifest.csv`、`valid_manifest.csv`。参数与实测结果见 [数据集标签转换说明](docs/数据集标签转换说明.md)。

### 4. 训练与推理

```bash
uv run python train.py      # 🚧 训练入口尚未实现
uv run python predict.py    # 当前仅打印环境信息
```

## ⚙️ 配置说明

[`datasets/data.yaml`](datasets/data.yaml) 是本项目的**自定义**配置，由自写 Dataset 读取：10 类名、官方 id ↔ 索引映射、影像 / 标签 / CSV 路径、实测规模、提交约定。

它**不是** Ultralytics 原生任务配置，不能交给 `yolo` 训练：

- `detect / segment / pose / obb` 需要 bbox / 多边形，本赛只有图像级多标签；
- `classify` 只认 ImageFolder 目录（`data=目录`，不读 yaml），且一图只能一个类别；
- 当前 Ultralytics 8.4.204 没有多标签分类任务，源码中不存在 multi-label 支持。

## ✅ 合规红线

- **预训练**：仅非医学开源权重（ImageNet 等）；**禁止** RETFound 及眼底 / 病理 / 放射 / 超声等医学预训练权重，蒸馏教师同理；须申报许可证全文、权重链接、SHA-256、model card；
- **数据**：只用官方数据，外部医学数据（含无标签）一律禁止；
- **测试集**：不得参与训练、伪标签、归一化统计量估计、checkpoint 选择或阈值拟合；
- **验证集**：可并入训练，但并入后不得再用于早停 / 选模 / 调参；
- **推理**：只做固定参数前向，BN 用训练集统计量、`eval` 模式；TTA 与集成允许；
- **体积与速度**：解压后 ≤ 500 MB，单图端到端 ≤ 100 ms；`cost.py` 必须复用 `predict.py` 的推理逻辑，100 张逐张一致性核验。

## ⚠️ 本地与评测环境差异

| 项 | 本仓库 | 评测容器 |
| --- | --- | --- |
| Python | 3.15.2 | **3.12** |
| PyTorch | 2.15.0 (+cu134) | **2.10.0** (CUDA 12.8) |
| 推理框架 | 同上 | TensorRT 10.9.0.34 / ONNX Runtime GPU 1.26.0 |
| 硬件 | RTX 5080 | **T4 16GB ×1** |

提交前必须在评测镜像内验证可运行性；更稳的做法是导出 ONNX / TensorRT，绕开运行时版本差异。

## 📚 文档

| 文档 | 说明 |
| --- | --- |
| [比赛规则文档](docs/比赛资料/比赛规则文档.md) | 赛制、任务、数据、指标、提交要求的完整整理 |
| [比赛参赛规则](docs/比赛资料/比赛参赛规则.md) / [补充说明](docs/比赛资料/参赛规则补充说明.md) | 官方 PDF 的文本转写 |
| [赛题技术分析](docs/比赛资料/赛题技术分析-YOLO任务类型判断.md) | 为什么这是分类不是检测，以及建模方案对比 |
| [数据集标签转换说明](docs/数据集标签转换说明.md) | `csv2txt.py` 用法、输出结构与实测统计 |
| [Ultralytics 训练参数参考](docs/Ultralytics训练参数参考.md) | 训练参数逐项说明 |
| [Ultralytics 预测参数参考](docs/Ultralytics预测参数参考.md) | 推理与导出参数逐项说明 |
| [AGENTS.md](AGENTS.md) | 仓库目录约定、代码风格与提交规范 |

## 🗺 路线图

- [x] 数据清点：影像、CSV、标注完整性核对
- [x] 逐图标签转换：`tools/csv2txt.py` + 30 344 / 1 422 张标签
- [x] 数据集配置：`datasets/data.yaml`
- [ ] 多标签 `Dataset`：三级目录读取 + 10 维 multi-hot + 增强
- [ ] 训练循环：骨干加载、10 维 sigmoid 头、BCE / ASL、AMP
- [ ] 评估：Macro-AUC 选模 + Study 聚合策略对比
- [ ] 提交链路：`submission.csv` 导出、`cost.py`、100 张一致性核验
- [ ] 部署：ONNX / TensorRT 导出、100 ms 实测、500 MB 体积核算
- [ ] 合规材料：许可证、SHA-256、model card、清单 xlsx、`code/` 打包

## 🙏 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics)：`yolo26*-cls` 分类链路与预训练权重（AGPL-3.0）；
- [PyTorch](https://pytorch.org/) / [torchvision](https://github.com/pytorch/vision)：训练与推理底座；
- [uv](https://github.com/astral-sh/uv)：依赖与虚拟环境管理；
- 全国高等学校计算机教育研究会及大赛组委会：赛题与数据集。

## 📄 许可证

本仓库代码以 **AGPL-3.0** 发布，全文见 [LICENSE.txt](LICENSE.txt)；所用预训练权重同样遵循 Ultralytics 的 AGPL-3.0 许可。

赛事数据、赛题材料版权归主办方所有，仅限本次竞赛使用，不随本仓库分发。
