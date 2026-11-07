# 赛题技术分析：这是 YOLO 的哪类任务？

> 面向 Ultralytics YOLO26 任务体系（检测 / 分割 / 语义分割 / 深度估计 / 分类 / 姿态估计 / OBB）作出的判定与建模建议。
> 规则依据见 [比赛规则文档.md](比赛规则文档.md)，任务口径以官网与官方 PDF 为准。

---

## 一、结论（一句话）

**这是「图像分类（Classification）」任务，且具体是多标签分类（multi-label classification）。**

在 Ultralytics 任务族里对应 **Classify / `-cls`**，即 `yolo26n-cls.pt`、`yolo26s-cls.pt`、`yolo26m-cls.pt`、`yolo26l-cls.pt`、`yolo26x-cls.pt` 这一类模型；
**不是** Detection、Segmentation、Semantic Segmentation、Depth Estimation、Pose、OBB 中的任何一类。

> 关键提醒：Ultralytics 官方 `-cls` 家族是**单标签 softmax**（ImageNet 式，输出一个类别 + 置信度，top-1/top-5）；本题是**10 类可共现的多标签**，因此**不能直接套用官方单标签分类流程**，必须把「10 类独立概率」这件事做出来（见 §四）。

---

## 二、判定依据（为什么是分类）

| 依据 | 事实 | 推导 |
| --- | --- | --- |
| **标注形态** | 只有「检查（Study）级」的 10 类标签，10 类可同时出现；**没有**任何边界框、掩码、关键点、旋转角、深度图标注 | 检测/分割/姿态/OBB/深度都无法训练 |
| **提交产物** | `submission.csv`：每个 `Study_id` × 10 个 `Predict_class` 的 `Probability` + `Gate` 阈值 | 输出是「整图 → 类别概率向量」，不是坐标集合 |
| **评价指标** | 主指标 **Macro-AUC**、副指标 **Macro-F1**，均为分类指标（AUC 只看概率排序，F1 依赖二值阈值） | 评测体系完全围绕分类概率设计 |
| **任务原文** | 「聚焦于**多标签胸部疾病分类**任务」「输出其可能包含的所有**疾病类别概率**」 | 明确定义为分类 |
| **图像尺度** | 胸部 X 光整幅图像（无固定尺寸），病变是弥漫性/全局征象（如心影扩大、肺水肿） | 不需要、也不适合做逐目标定位 |

> 赛题背景里的「进行多目标胸部病变的**检测**与分类」是医学语境下的泛称（临床报告常写「检出/发现」），
> 但**赛事实际只提供分类标签、只评分类指标、只收分类概率**——备考时应按分类任务对待。

### 七类任务逐一排除

| Ultralytics 任务 | 是否适用 | 理由 |
| --- | --- | --- |
| Detection（检测） | ❌ | 无 bbox 标注；提交格式无坐标；指标非 mAP |
| Segmentation（实例分割） | ❌ | 无实例 mask 标注 |
| Semantic Segmentation（语义分割） | ❌ | 无像素级标注，评测不看像素 |
| Depth Estimation（深度估计） | ❌ | 无深度 GT，赛题不涉及 |
| **Classification（分类）** | ✅ **本题** | Study 级 10 类多标签概率 + Macro-AUC |
| Pose（姿态估计） | ❌ | 无关键点标注 |
| OBB（旋转框检测） | ❌ | 无旋转框标注 |

---

## 三、与「标准 YOLO26-cls」的五处关键差异

| 维度 | 官方 `yolo26*-cls` 默认 | 本题要求 | 影响 |
| --- | --- | --- | --- |
| 数据集组织 | ImageFolder：`train/类名/图片`，一图一标签 | `images/subject_x/study_y/image_z.jpg` 三级目录 + 多标签标注表 | 必须自写 `Dataset`，不能直接用 `data="path/to/dataset"` |
| 标签与损失 | 单标签 softmax + 交叉熵，`probs` 之和为 1 | 10 类**独立 sigmoid** + BCE（可换 ASL/Focal），各类概率互不归一 | 必须**替换分类头 + 损失函数** |
| 验证指标 | top-1 / top-5 accuracy | **Macro-AUC**（+ Macro-F1） | 验证/早停逻辑要改成按 Macro-AUC 选模 |
| 推理输出 | `result.probs`（softmax 向量） | 10 维独立概率，直接写入 csv | 导出逻辑与 `predict.py`/`cost.py` 自写 |
| 输入尺寸 | 默认 224 方形（RandomResizedCrop/CenterCrop） | 胸片长宽比差异大、细节重要 | 建议 letterbox 或 384–512 输入，避免裁掉肺野 |

> 也就是说：**任务类型 = 分类；实现形态 = 把 YOLO26-cls 改造成多标签分类器**。
> 直接 `model.train(data=...)` 走 ImageFolder 单标签流程，会与赛题标注体系和评价指标不匹配。

---

## 四、建模方案

### 方案 A（推荐）：在 Ultralytics 分类链路上改造为多标签

1. **预训练**：从 `yolo26n-cls.pt`（或 `s`/`m`，视速度预算）加载 ImageNet 权重——属非医学通用数据，符合《规则》第 1 条。
2. **数据层**：自写 `Dataset`，按 `subject/study/image` 读取图片，返回 `(image_tensor, 10 维 multi-hot 标签)`；采样与划分严格用官方给定的 train/val（测试集无标签）。
3. **模型层**：把分类头最后一层 Linear 改为 **10 维输出**（去掉 softmax）；可保留 YOLO26-cls 骨干与 neck。
4. **损失**：`BCEWithLogitsLoss`（可加 `pos_weight` 处理不平衡），或 **ASL（Asymmetric Loss）/ Focal Loss**；10 类各自独立，不做 softmax。
5. **训练**：`imgsz` 224–512（T4 上 384 较稳）、AMP FP16、余弦/OneCycle 学习率；验证集**只**用来算 Macro-AUC 与选模。
6. **推理**：`sigmoid(logits)` → 每图 10 概率；按 `Study_id` 聚合多张图后写 `submission.csv`（`Gate` 默认 0.5）。
7. **部署**：导出 ONNX / TensorRT（`model.export(format="engine", half=True)`），确保单图端到端 ≤100ms、权重 ≤500MB；`cost.py` 复用 `predict.py` 的推理函数。

### 方案 B：YOLO26-cls 仅作骨干 + 自写 PyTorch 训练循环

适合需要精细控制的场景：
- 分层学习率（骨干小、头部大）、冻结/解冻策略；
- 自定义 Macro-AUC 早停、类别加权、图像→检查聚合方式（max / mean / attention）；
- 训练循环完全自控，天然满足「验证集并入训练后不得再用于早停」等合规约束（把选择逻辑前置）。

代价：要自己实现数据加载、增强、导出与速度测试；但本赛只有 10 类、2 万张量级，成本可控。

### 方案 C（不推荐）：用检测/分割模型硬套

- 没有框/掩码标注，无法训练有监督检测头；
- 即便用弱监督/伪框，提交格式也不接受坐标，评测也不看定位；
- 把 10 类当互斥单标签（softmax）会**丢失多标签共现信息**，直接损害 Macro-AUC。

---

## 五、关键工程点（结合规则）

1. **检查级聚合（Study 多图）**：一个 `Study_id` 可能有多张片（不同体位/时间）。允许对同患者/同检查的输出做**确定性聚合**（均值/取最大/投票）——属《补充说明》第七条允许范围；但**不得**跨测试图汇总统计量去拟合参数。
2. **分辨率与延迟**：T4 16GB、单图 ≤100ms、模型 ≤500MB。`yolo26n-cls`（2.8M 参数）留有余量；胸片细节敏感，可试 384–512 + FP16/TensorRT，用 `cost.py` 实测（100 张 × 3 轮）。
3. **类别不平衡与共现**：10 类（含 `No Finding`）分布不均，建议 ASL/Focal、`pos_weight`、过采样；多标签共现是信息而非噪声，**不要**强行拆成 10 个独立二分类模型（成本高、共享特征差）。
4. **预训练合规**：ImageNet 权重合规；但须申报**许可证全文（Ultralytics 为 AGPL-3.0）、权重链接、SHA-256、model card**。**严禁** RETFound / 眼底 / 病理 / 放射 / 超声等医学预训练权重（含蒸馏教师）。
5. **数据红线**：测试集不参与任何训练/选模/统计量估计；验证集一旦并入训练，不得再用于早停/阈值/超参选择；外部医学数据（含无标签）一律禁止。
6. **提交细节**：`submission.csv` 每个 Study 10 行、UTF-8、概率 [0,1]、`Gate` 默认 0.5；每日 2 次提交、仅 1 次撤销。
7. **允许的增益**：TTA（翻转/多裁剪）与多模型集成**明确允许**；但集成后仍受 500MB / 100ms 约束。
8. **可复现**：固定随机种子、eval 模式推理、保留训练脚本与权重哈希，提交 zip 含 `code/`、`cost.py`、清单 xlsx。

---

## 六、建议的 8 天冲刺节奏（截至 2026-11-13 23:59）

| 天 | 任务 |
| --- | --- |
| D1 | 数据清点、目录/标注解析、baseline（`yolo26n-cls` + 改造头 + BCE）跑通，产出首版 `submission.csv` |
| D2 | 验证集 Macro-AUC 评估管线、图像→Study 聚合策略对比（max/mean） |
| D3–4 | 分辨率/模型规模/损失（BCE vs ASL）/增强消融；处理类别不平衡 |
| D5 | 集成 + TTA 试验；导出 ONNX/TensorRT，实测速度 |
| D6 | 写 `cost.py` 并做 100 张一致性核验（|Δp|≤1e-3） |
| D7 | 合规材料：许可证、SHA-256、model card、xlsx 清单、`code/` 目录 |
| D8 | 留一次提交余量，核对 10 行/Study、UTF-8、Gate 列，按时提交 |

---

## 七、附：本仓库实测数据（2026-11-07 解压后）

| 划分 | CSV 行（=图像数） | 检查数（Study） | 患者数 | 标签 |
| --- | --- | --- | --- | --- |
| train | 30 344 | 18 513 | 11 055 | `Predict_class` 分号多标签，如 `1;8;9;4` |
| valid | 1 422 | 859 | 228 | 同上 |
| test | 2 218 | 1 414 | 136 | **无标签列** |

- 一个检查平均约 1.64 张图（train），所以「图像数 ≠ 检查数」，提交按检查聚合。
- 已生成逐图 txt 标签（0-based）：`datasets/labels/`，脚本见 [tools/csv2txt.py](../../tools/csv2txt.py)，用法与统计见 [数据集标签转换说明.md](../数据集标签转换说明.md)。
- ⚠️ 当前 `datasets/data.yaml` 是 `channels: 5` + `infrared/depth` + 12 类（person/boat/animal…）的另一套检测配置，**与本赛题无关**，不要用于本任务。

---

## 八、参考

- Ultralytics 图像分类任务文档（YOLO26-cls 使用 `-cls` 后缀、ImageNet 预训练、默认 softmax 单标签）：<https://docs.ultralytics.com/tasks/classify/>
- 本仓库既有资料：[Ultralytics 训练参数参考](../Ultralytics训练参数参考.md)、[Ultralytics 预测参数参考](../Ultralytics预测参数参考.md)
- 官方赛题页：<http://www.ncccu.org.cn/index/Paper/case1.html>
- 规则原文：[比赛参赛规则（官方PDF转写）](比赛参赛规则（官方PDF转写）.md)、[参赛规则补充说明（官方PDF转写）](参赛规则补充说明（官方PDF转写）.md)
