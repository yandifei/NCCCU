# Ultralytics 分类训练参数参考（通用参数完整版 + Classify 专项）

> 历史资料，对应此前的 YOLO 版本。当前入口与参数见 [ConvNeXt-Small 训练与推理](训练与推理说明.md)。

> 版本：2026-11-07。**本文件包含完整通用训练参数表，以及单独的分类任务参考**。通用模式文档面向多种视觉任务；有参数并不代表它会对分类训练生效。原上传文档的完整参数行与说明均保留，额外列出官网新增参数。分类训练以 `*-cls.pt` 模型和分类目录数据集为准。

## 官方和原始资料

- [通用训练模式](https://docs.ultralytics.com/zh/modes/train)
- [通用预测模式](https://docs.ultralytics.com/zh/modes/predict)
- [分类任务说明](https://docs.ultralytics.com/zh/tasks/classify)
- [分类数据集格式](https://docs.ultralytics.com/zh/datasets/classify)
- 本文「完整通用训练表」「完整通用增强表」以你上传的《Ultralytics训练参数参考(2).md》为基础，参数名称、类型、默认值、说明最大程度保持原文件；另增加「分类适用性」列。`tile` / `split` 是对照官网补充的条目，特别标明。
- `✅ 通用可用`：分类训练一般可使用；`⭐ 分类重点`：分类强相关或分类专用；`⚠️ 条件适用/需验证`：通用入口有此参数，但分类实际效果受具体训练器/版本影响；`— 其他任务/分类不使用`：主要用于检测、分割、姿态等其他任务。
- 默认值是**通用模式页面值**，不是为 YOLO 分类任务优化的建议值；示例的具体超参数需根据数据集调试。

## 1. 完整通用训练参数表（`model.train()`）

**共 66 项**：原文件训练表全部保留，另增加官网 `tile`、`split` 两项。分类并不需要逐一传入全部参数。

| 参数 | 类型 | 默认值 | 描述 | 分类适用性 |
| --- | --- | --- | --- | --- |
| `model` | `str` | `None` | 指定用于训练的模型文件。接受指向 `.pt` 预训练模型或 `.yaml` 配置文件的路径。对于定义模型结构或初始化权重至关重要。 | ✅ 通用可用 |
| `data` | `str` | `None` | 数据集 YAML 的路径（例如 `coco8.yaml`），其中包含训练和验证数据的路径、类别名称和类别数量。分类任务则使用数据集目录或内置数据集名称（例如 `imagenet10`）。 | ✅ 通用可用 |
| `epochs` | `int` | `100` | 训练周期总数。每个周期表示完整遍历整个数据集一次。调整此值会影响训练时长和模型性能。 | ✅ 通用可用 |
| `time` | `float` | `None` | 最大训练时间（小时）。设置后，它会覆盖 `epochs` 参数，使训练在达到指定时长后自动停止。适用于受时间限制的训练场景。 | ✅ 通用可用 |
| `patience` | `int` | `100` | 在验证指标没有改善的情况下，等待多少个周期后提前停止训练。通过在性能趋于稳定时停止训练，有助于防止过拟合。 | ✅ 通用可用 |
| `batch` | `int` 或 `float` | `16` | 批次大小有三种模式：设置为整数（例如 `batch=16`）、自动模式（使用 60% 的 GPU 内存，`batch=-1`），或带指定利用率比例的自动模式（`batch=0.70`）。 | ✅ 通用可用 |
| `imgsz` | `int` | `640` | 训练目标图像尺寸。图像会被调整为边长等于指定值的正方形（如果为 `rect=False`）；对于 YOLO 模型会保持宽高比，但 RT-DETR 不会。会影响模型准确率和计算复杂度。 | ✅ 通用可用 |
| `save` | `bool` | `True` | 启用训练检查点和最终模型权重的保存。适用于恢复训练或部署模型。 | ✅ 通用可用 |
| `save_period` | `int` | `-1` | 模型检查点的保存频率，以周期数指定。值为 -1 时禁用此功能。适用于在长时间训练期间保存中间模型。 | ✅ 通用可用 |
| `cache` | `bool` | `False` | 启用将数据集图像缓存到内存（`True`/`ram`）、磁盘（`disk`），或禁用缓存（`False`）。通过减少磁盘 I/O 提高训练速度，但会增加内存使用量。 | ✅ 通用可用 |
| `tile` | `bool` 或 `int` | `False` | 将每张源图像扩展为固定大小的训练切片，并裁剪相应标签、保留负样本窗口；`True` 使用 `imgsz`，正整数指定切片大小。此参数为 2026-11-07 官网新增项，原上传表没有。 | ⚠️ 条件适用/需验证 |
| `device` | `int` 或 `str` 或 `list` | `None` | 指定训练使用的计算设备：单个 GPU（`device=0`）、多个 GPU（`device=[0,1]`）、CPU（`device=cpu`）、Apple silicon 上的 MPS（`device=mps`）、Huawei Ascend NPU（`device=npu:0` 或 `device=npu:0,1`），或自动选择空闲 GPU（`device=-1`）或多个空闲 GPU（`device=[-1,-1]`）。 | ✅ 通用可用 |
| `workers` | `int` | `8` | 数据加载的工作线程数（Multi-GPU 训练时为每个 `RANK`）。会影响数据预处理和输入模型的速度，在多 GPU 配置中尤其有用。 | ✅ 通用可用 |
| `project` | `str` | `None` | 保存训练输出的项目目录名称。便于有序存储不同实验。 | ✅ 通用可用 |
| `name` | `str` | `None` | 训练运行名称。用于在项目文件夹中创建子目录，以存储训练日志和输出。 | ✅ 通用可用 |
| `exist_ok` | `bool` | `False` | 如果为 True，则允许覆盖现有的项目/名称目录。便于迭代实验，无需手动清除之前的输出。 | ✅ 通用可用 |
| `save_dir` | `str` | `None` | 指定保存运行输出的确切目录，覆盖 `project`/`name` 组合。路径将按原样使用，不会自动递增，因此连续运行会重复使用同一目录。 | ✅ 通用可用 |
| `pretrained` | `bool` 或 `str` | `True` | 决定是否从预训练权重开始训练。可以是布尔值，也可以是要加载的权重文件路径。`pretrained=False` 会使用随机初始化的权重进行训练，同时保留模型架构。 | ✅ 通用可用 |
| `cls_remap` | `bool` | `True` | 跨数据集微调时，会将预训练分类头中与新模型类别名称匹配的行复制到新模型中，因此重叠类别会保留已学习的偏置；当分类头宽度不变时，还会保留其权重。无论类别数量不同，还是数量相同但类别顺序不同，均适用。 | ⭐ 分类重点 |
| `optimizer` | `str` | `'auto'` | 训练所使用的优化器。选项包括 `SGD`、`MuSGD`、`Adam`、`Adamax`、`AdamW`、`NAdam`、`RAdam`、`RMSProp`，或 `auto`，根据训练迭代次数选择 `AdamW` 或 `MuSGD`。会影响收敛速度和稳定性。 | ✅ 通用可用 |
| `seed` | `int` | `0` | 设置训练的随机种子，确保使用相同配置运行时结果具有可复现性。 | ✅ 通用可用 |
| `deterministic` | `bool` | `True` | 强制使用确定性算法，确保结果可复现，但由于限制使用非确定性算法，可能会影响性能和速度。 | ✅ 通用可用 |
| `verbose` | `bool` | `True` | 启用训练期间的详细输出，在控制台中显示进度条、每周期指标和其他训练信息。 | ✅ 通用可用 |
| `single_cls` | `bool` | `False` | 训练期间将多类别数据集中的所有类别视为单一类别。适用于二分类任务，或关注对象是否存在而非其类别的场景。 | ⚠️ 条件适用/需验证 |
| `classes` | `list[int]` | `None` | 指定要训练的类别 ID 列表。适用于过滤掉某些类别，仅关注指定类别。 | ⚠️ 条件适用/需验证 |
| `rect` | `bool` | `False` | 启用最小填充策略——批次中的图像只进行最小程度的填充以达到统一尺寸，最长边等于 `imgsz`。可以提高效率和速度，但可能影响模型准确率。 | ⚠️ 条件适用/需验证 |
| `multi_scale` | `float` | `0.0` | 每个批次随机改变 `imgsz`，变化范围为 +/- `multi_scale`（例如从 `0.25` -> `0.75x` 到 `1.25x`），并四舍五入到模型步幅的倍数；`0.0` 会禁用多尺度训练。 | ⚠️ 条件适用/需验证 |
| `cos_lr` | `bool` | `False` | 使用余弦学习率调度器，在各个周期内按照余弦曲线调整学习率。有助于管理学习率，从而实现更好的收敛。 | ✅ 通用可用 |
| `close_mosaic` | `int` | `10` | 在最后 N 个周期中禁用马赛克数据增强，以便在训练结束前稳定训练。设置为 0 时禁用此功能。 | — 其他任务/分类不使用 |
| `resume` | `bool` | `False` | 从最后保存的检查点恢复训练。自动加载模型权重、优化器状态和周期计数，无缝继续训练。 | ✅ 通用可用 |
| `amp` | `bool` 或 `str` | `True` | 设置训练精度：`True` 或 `"fp16"` 使用 FP16，`"bf16"` 在受支持的 CUDA 设备上使用 BF16，`False` 或 `"fp32"` 使用 FP32。 | ✅ 通用可用 |
| `quantize` | `int` 或 `str` | `None` | 设为 `8`（或 `"int8"`）以进行 INT8 量化感知训练（QAT），该训练在循环中使用伪量化进行微调，从而使权重能够承受 INT8 导出。 | ⚠️ 条件适用/需验证 |
| `fraction` | `float`、`int` 或 `list` | `1.0` | 数据集子集，可指定比例/数量或 `[train, val, test]` 列表。`1` 表示完整拆分，超过 `1` 的整数表示图像数量，只有可选的测试项接受 `0`/`0.0` 来表示无数据。包含两个项目的列表会保留完整测试集。 | ✅ 通用可用 |
| `profile` | `bool` | `False` | 启用训练期间对 ONNX 和 TensorRT 速度进行分析，有助于优化模型部署。 | ✅ 通用可用 |
| `freeze` | `int` 或 `list` | `None` | 冻结模型的前 N 层，或按索引或模块名称指定的层（`23.cv2`，不含开头的 `model.`），从而减少可训练参数数量。适用于微调或迁移学习。 | ✅ 通用可用 |
| `lr0` | `float` | `0.01` | 初始学习率（即 `SGD=1E-2`、`Adam=1E-3`）。调整此值对优化过程至关重要，会影响模型权重更新的速度。 | ✅ 通用可用 |
| `lrf` | `float` | `0.01` | 最终学习率占初始学习率的比例 =（`lr0 * lrf`），与调度器结合使用，以随时间调整学习率。 | ✅ 通用可用 |
| `momentum` | `float` | `0.937` | SGD 的动量因子或 Adam 优化器的 beta1，影响当前更新中对历史梯度的整合。 | ✅ 通用可用 |
| `weight_decay` | `float` | `0.0005` | L2 正则化项，通过惩罚较大的权重来防止过拟合。 | ✅ 通用可用 |
| `warmup_epochs` | `float` | `3.0` | 学习率预热周期数，将学习率从较低值逐步提高到初始学习率，以在训练早期稳定训练。 | ✅ 通用可用 |
| `warmup_momentum` | `float` | `0.8` | 预热阶段的初始动量，在预热期间逐步调整到设定的动量。 | ✅ 通用可用 |
| `warmup_bias_lr` | `float` | `0.1` | 预热阶段偏置参数的学习率，有助于稳定最初几个周期的模型训练。在默认 `optimizer='auto'` 下会自动设置为 `0.0`，因此需要显式指定优化器才能使用它。 | ✅ 通用可用 |
| `distill_model` | `str` | `None` | 用于知识蒸馏的教师模型检查点路径（例如 `yolo26x.pt`）。设置后，学生模型会在冻结教师模型的指导下，通过额外的蒸馏损失进行训练。 | ⚠️ 条件适用/需验证 |
| `dis` | `float` | `6.0` | 添加到标准检测损失中的蒸馏损失权重。值越大，教师模型特征指导的影响越大。 | — 其他任务/分类不使用 |
| `box` | `float` | `7.5` | 损失函数中框损失分量的权重，影响模型对准确预测边界框坐标的重视程度。 | — 其他任务/分类不使用 |
| `cls` | `float` | `0.5` | 总损失函数中的分类损失权重，影响正确类别预测相对于其他分量的重要性。 | ⚠️ 条件适用/需验证 |
| `cls_pw` | `float` | `0.0` | 使用类别频率倒数处理类别不平衡时的类别加权幂。`0.0` 禁用类别加权，`1.0` 应用完整的频率倒数加权。0 到 1 之间的值表示部分加权。 | ⚠️ 条件适用/需验证 |
| `dfl` | `float` | `1.5` | 框距离回归项的权重：当检测头使用 `reg_max > 1` 时为分布式焦点损失（DFL），在无 DFL 的 YOLO26（`reg_max: 1`）中为对归一化框距离计算的 L1 损失。 | — 其他任务/分类不使用 |
| `pose` | `float` | `12.0` | 姿态估计模型中姿态损失的权重，影响模型对准确预测姿态关键点的重视程度。 | — 其他任务/分类不使用 |
| `kobj` | `float` | `1.0` | 姿态估计模型中关键点目标性损失的权重，用于平衡检测置信度与姿态准确率。 | — 其他任务/分类不使用 |
| `rle` | `float` | `1.0` | 姿态估计模型中残差对数似然估计损失的权重，影响关键点定位的精度。 | — 其他任务/分类不使用 |
| `angle` | `float` | `1.0` | obb 模型中角度损失的权重，影响定向边界框角度预测的精度。 | — 其他任务/分类不使用 |
| `dlog` | `float` | `1.0` | 深度估计模型中尺度不变对数（SILog）损失的权重，这是驱动深度准确率的主要项。 | — 其他任务/分类不使用 |
| `dgrad` | `float` | `0.5` | 深度估计模型中梯度损失的权重，用于惩罚深度边缘误差并促使表面边界更加清晰。 | — 其他任务/分类不使用 |
| `dlam` | `float` | `1.0` | 深度估计模型中 SILog 损失的方差关注因子。`1.0` 使损失完全具有尺度不变性，而 `0.0` 会将其简化为普通的对数均方根误差。 | — 其他任务/分类不使用 |
| `nbs` | `int` | `64` | 用于归一化损失的名义批次大小。 | ✅ 通用可用 |
| `overlap_mask` | `bool` | `True` | 决定是否将对象掩码合并为单个掩码进行训练，或为每个对象分别保留掩码。发生重叠时，合并过程中较小的掩码会叠加在较大的掩码上方。 | — 其他任务/分类不使用 |
| `mask_ratio` | `int` | `4` | 分割掩码的下采样比例，影响训练期间使用的掩码分辨率。 | — 其他任务/分类不使用 |
| `dropout` | `float` | `0.0` | 分类任务中的正则化丢弃率，通过在训练期间随机忽略单元来防止过拟合。 | ⭐ 分类重点 |
| `val` | `bool` | `True` | 启用训练期间的验证，从而可以定期在独立数据集上评估模型性能。 | ✅ 通用可用 |
| `split` | `str` | 'val' | 选择训练期间用于验证的划分：`val`、`test` 或 `train`；用于选择 best.pt 和提前停止。此参数为 2026-11-07 官网新增项，原上传表没有。 | ✅ 通用可用 |
| `nms` | `bool`，可选 | `None` | 选择用于轮次验证、检查点选择和早停的推理头。`None` 或 `True` 使用带 NMS 的一对多（one-to-many）；`False` 在可用时使用无 NMS（NMS-free）的头。两个头都保留其训练损失。 | — 其他任务/分类不使用 |
| `plots` | `bool` | `True` | 生成并保存训练和验证指标图，以及预测示例，为模型性能和学习进展提供可视化洞察。 | ✅ 通用可用 |
| `compile` | `bool` 或 `str` | `False` | 使用 `backend='inductor'` 启用 PyTorch 2.x 的 `torch.compile` 图编译。接受 `True` → `"default"`、`False` → 禁用，或使用 `"default"`、`"reduce-overhead"`、`"max-autotune-no-cudagraphs"` 等字符串模式。不支持时会发出警告并回退到 eager 模式。 | ✅ 通用可用 |
| `channels_last` | `bool` | `None` | 在训练期间为卷积使用 channels_last (NHWC) 内存格式。`None` 会在配备 PyTorch 1.11 或更新版本的 CUDA 上自动启用它，但在测得运行较慢的 Windows 系统上除外。`False` 会禁用它，而 `True` 会显式请求它。PyTorch 1.10 及更旧版本、CPU 和 MPS 默认保持 NCHW 格式。 | ✅ 通用可用 |
| `max_det` | `int` | `300` | 训练验证期间每张图像的最大检测数。对于 detect、segment、pose 和 OBB，默认的 300 只有在训练集/验证集标注对象最大数量超过 300 时才会增加到该数量。其他值保持固定；超出限制会触发警告。 | — 其他任务/分类不使用 |

> **分类特别注意**：`imgsz=640` 是通用表默认值；分类预训练基准使用 `224`。`single_cls=True` 并不是图像二分类开关，普通二分类需要在数据集中设置两个类别目录。`box`、`dfl`、`pose`、`nms` 等虽保留在表中，但不属于普通整图分类流程。`cls` 是通用损失权重名，不等于分类任务类别数量。

## 2. 完整通用数据增强参数表（Augmentation Settings）

**共 19 项**：原文件数据增强全表保留，包括 `mosaic`、`copy_paste` 等分类不使用的配置，同时标注分类支持情况。

| 参数 | 类型 | 默认值 | 支持的任务 | 范围 | 描述 | 分类适用性 |
| --- | --- | --- | --- | --- | --- | --- |
| `hsv_h` | `float` | `0.015` | `detect`、`segment`、`semantic`、`depth`、`classify`、`pose`、`obb` | `0.0 - 1.0` | 按色轮的一定比例调整图像色调，引入颜色变化。帮助模型适应不同的光照条件。对于 `classify`，仅当 `auto_augment=None` 时应用。 | ✅ 分类有条件（auto_augment=None） |
| `hsv_s` | `float` | `0.7` | `detect`、`segment`、`semantic`、`depth`、`classify`、`pose`、`obb` | `0.0 - 1.0` | 按一定比例调整图像饱和度，影响颜色的强度。适用于模拟不同的环境条件。对于 `classify`，仅当 `auto_augment=None` 时应用。 | ✅ 分类有条件（auto_augment=None） |
| `hsv_v` | `float` | `0.4` | `detect`、`segment`、`semantic`、`depth`、`classify`、`pose`、`obb` | `0.0 - 1.0` | 按一定比例修改图像的明度（亮度），帮助模型在各种光照条件下保持良好表现。对于 `classify`，仅当 `auto_augment=None` 时应用。 | ✅ 分类有条件（auto_augment=None） |
| `degrees` | `float` | `0` | `detect`、`segment`、`semantic`、`depth`、`pose`、`obb` | `0.0 - 180` | 在指定的角度范围内随机旋转图像，提升模型识别不同方向对象的能力。 | — 任务支持列表不含 classify |
| `translate` | `float` | `0.1` | `detect`、`segment`、`semantic`、`depth`、`pose`、`obb` | `0.0 - 1.0` | 按图像尺寸的一定比例水平和垂直平移图像，帮助模型学习检测部分可见的对象。 | — 任务支持列表不含 classify |
| `scale` | `float \| tuple` | `0.5` | `detect`、`segment`、`semantic`、`depth`、`classify`、`pose`、`obb` | `0 - 1`，或显式的 `(min, max)` 元组（不适用于 `classify`） | 按照增益因子缩放图像，模拟距离摄像头远近不同的对象。 | ✅ 分类可用 |
| `shear` | `float` | `0` | `detect`、`segment`、`semantic`、`depth`、`pose`、`obb` | `-180 - +180` | 按指定角度剪切图像，模拟从不同角度观察对象的效果。 | — 任务支持列表不含 classify |
| `perspective` | `float` | `0` | `detect`、`segment`、`semantic`、`depth`、`pose`、`obb` | `0.0 - 0.001` | 对图像应用随机透视变换，增强模型理解三维空间中对象的能力。 | — 任务支持列表不含 classify |
| `flipud` | `float` | `0` | `detect`、`segment`、`semantic`、`depth`、`classify`、`pose`、`obb` | `0.0 - 1.0` | 以指定概率将图像上下翻转，在不影响对象特征的情况下增加数据变化。 | ✅ 分类可用 |
| `fliplr` | `float` | `0.5` | `detect`、`segment`、`semantic`、`depth`、`classify`、`pose`、`obb` | `0.0 - 1.0` | 以指定概率将图像左右翻转，适用于学习对称对象并增加数据集多样性。 | ✅ 分类可用 |
| `bgr` | `float` | `0` | `detect`、`segment`、`semantic`、`depth`、`pose`、`obb` | `0.0 - 1.0` | 以指定概率将图像通道从 RGB 翻转为 BGR，适用于增强模型对错误通道顺序的鲁棒性。 | — 任务支持列表不含 classify |
| `mosaic` | `float` | `1` | `detect`、`segment`、`semantic`、`pose`、`obb` | `0.0 - 1.0` | 将四张训练图像合并为一张，模拟不同的场景构成和对象交互。对复杂场景理解非常有效。 | — 任务支持列表不含 classify |
| `mixup` | `float` | `0` | `detect`、`segment`、`semantic`、`pose`、`obb` | `0.0 - 1.0` | 混合两张图像及其标签，创建合成图像。通过引入标签噪声和视觉变化，增强模型的泛化能力。 | — 任务支持列表不含 classify |
| `cutmix` | `float` | `0` | `detect`、`segment`、`pose`、`obb` | `0.0 - 1.0` | 组合两张图像的部分区域，在保留清晰区域的同时创建局部混合效果。通过创建遮挡场景增强模型鲁棒性。 | — 任务支持列表不含 classify |
| `copy_paste` | `float` | `0` | `segment`、`obb` | `0.0 - 1.0` | 要粘贴的符合条件对象所占的比例；`flip` 会在图像内镜像这些对象，而 `mixup` 还会将该值用作跨图像应用概率。 | — 任务支持列表不含 classify |
| `copy_paste_mode` | `str` | `flip` | `segment`、`obb` | - | 指定要使用的 `copy-paste` 策略。选项包括 `'flip'` 和 `'mixup'`。 | — 任务支持列表不含 classify |
| `auto_augment` | `str` | `randaugment` | `classify` | - | 应用预定义的增强策略（`'randaugment'`、`'autoaugment'` 或 `'augmix'`），通过增加视觉多样性来提升模型性能。 | ⭐ 分类专用 |
| `erasing` | `float` | `0.4` | `classify` | `0.0 - 1.0` | 在训练期间随机擦除图像区域，促使模型关注不太明显的特征。 | ⭐ 分类专用 |
| `augmentations` | `list` | `None` | `detect`、`segment`、`semantic`、`depth`、`pose`、`obb` | - | 用于高级数据增强的自定义 Albumentations 变换（仅限 Python API）。接受变换对象列表，以满足特殊的增强需求。 | — 任务支持列表不含 classify |

> `hsv_h` / `hsv_s` / `hsv_v` 在分类内置增强中通常要求 `auto_augment=None` 才按相应路径应用。`degrees`、`translate`、`mosaic` 等没有被官方数据增强支持表列为 `classify`，因此不能只因为通用接口接受参数就断定它能改变分类训练。

## 3. 分类专用训练参数表（单独保留）

这一节只列**分类直接专用/分类头强相关**的参数；共享训练参数在第 1 节有全部定义，在第 6 节有分类可用的配置速查。

| 参数 | 类型 | 官方通用默认值 | 适用层次 | 分类训练中的作用及建议 |
|---|---|---|---|---|
| `dropout` | `float` | `0.0` | 分类专用训练设置 | 分类头 Dropout 丢弃率，可缓解过拟合；建议先保持默认，再在验证集上比较 `0.1`、`0.2` 等。 |
| `auto_augment` | `str` | `randaugment` | 分类专用数据增强 | 可选 `randaugment` / `autoaugment` / `augmix`，亦可设 `None`；用于分类的 torchvision 自动增强策略。 |
| `erasing` | `float` | `0.4` | 分类专用数据增强 | 随机擦除概率；细粒度瑕疵、边缘细节识别时建议验证关闭擦除。 |
| `cls_remap` | `bool` | `True` | 分类头迁移相关 | 不同数据集之间迁移时，根据可匹配的类别名称复用预训练分类头的对应权重；不负责指定类别数。 |

### 3.1 分类还会使用的共享增强参数

| 参数 | 默认值 | 分类使用条件 / 注意事项 |
|---|---|---|
| `hsv_h` | `0.015` | 对应色调扰动；通常在 `auto_augment=None` 时才由内置分类增强使用。 |
| `hsv_s` | `0.7` | 对应饱和度扰动；同上。 |
| `hsv_v` | `0.4` | 对应亮度扰动；同上。 |
| `scale` | `0.5` | 分类缩放/裁切强度参数；建议检查是否破坏类别判别特征。 |
| `flipud` | `0.0` | 上下翻转概率，方向相关任务一般关闭。 |
| `fliplr` | `0.5` | 左右翻转概率，文字和左右方向敏感任务应慎用。 |

### 3.2 一组可复制的分类训练起点

```python
from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")
model.train(
    data="dataset",     # 分类数据集根目录（含 train/、val/ 类别子目录）
    epochs=100,
    imgsz=224,          # 分类常用起点，不是通用默认值
    batch=16,
    patience=20,
    optimizer="auto",
    pretrained=True,
    dropout=0.0,
    auto_augment="randaugment",
    erasing=0.4,
    project="runs/classify",
    name="baseline",
    plots=True,
)
```

---

## 后续：分类任务操作说明、模型基准与完整示例

以下延续先前的分类训练版，包括数据集格式、分类可用通用参数速查表、任务不适用的参数说明、Python/CLI 代码、验证指标，以及裁剪与 Letterbox 自定义变换。

## 4. 任务说明与命令入口

- `classify` 预测整张图片所属的一个类别，不输出目标框；采用分类目录数据集，训练时通常不需要 YOLO 检测标签 TXT 或 `data.yaml`。
- 分类模型名带 `-cls`：如 `yolo26n-cls.pt`、`yolo26s-cls.pt`。`yolo26n.pt` **是不带分类头的检测模型**，不要误用。
- Python：`YOLO("yolo26n-cls.pt").train(data="dataset", ...)`。
- CLI：`yolo classify train model=yolo26n-cls.pt data=dataset ...`。
- 用于验证：`model.val()` 或 `yolo classify val ...`；预测：`model.predict(...)`。

### 4.1 分类预训练模型（官网 ImageNet 验证集基准）

官方表格是 **224×224** 输入下的 ImageNet 指标，不是自定义数据集的保证结果；CPU 数字为 ONNX、T4 数字为 TensorRT10 的示例基准。

| 模型 | 输入像素 | Top-1 (%) | Top-5 (%) | CPU ONNX(ms) | T4 TensorRT10(ms) | 参数(M) | FLOPs(B) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26n-cls | 224 | 71.4 | 90.1 | 5.0 ± 0.3 | 1.1 ± 0.0 | 2.8 | 0.4 |
| YOLO26s-cls | 224 | 76.0 | 92.9 | 7.9 ± 0.2 | 1.3 ± 0.0 | 6.7 | 1.5 |
| YOLO26m-cls | 224 | 78.1 | 94.2 | 17.2 ± 0.4 | 2.0 ± 0.0 | 11.6 | 4.8 |
| YOLO26l-cls | 224 | 79.0 | 94.6 | 23.2 ± 0.3 | 2.8 ± 0.0 | 14.1 | 6.0 |
| YOLO26x-cls | 224 | 79.9 | 95.0 | 41.4 ± 0.9 | 3.8 ± 0.0 | 29.6 | 13.5 |

> 实验起点建议：先用 `yolo26n-cls.pt` + `imgsz=224` 做基线，再按精度、速度与显存预算比较 s/m/l/x。文档中的 `imgsz=640` 只是通用训练模式默认值，不应机械沿用到每个分类实验。

## 5. 分类数据集结构

```text
dataset/
├── train/
│   ├── cat/
│   │   ├── 001.jpg
│   │   └── 002.jpg
│   ├── dog/
│   │   └── 003.jpg
│   └── bird/
│       └── 004.jpg
├── val/                 # 也可用 valid/ 或 validation/
│   ├── cat/
│   ├── dog/
│   └── bird/
└── test/                # 可选；每类同样建子目录
    ├── cat/
    ├── dog/
    └── bird/
```

- `data` 写根目录 `dataset`（不是检测任务常见的 `dataset.yaml`）。每张图片的类别由其所在类别子目录确定。
- 官网说明：若无验证目录、但有 `test`，会使用 `test` 进行验证；如根目录直接是类别文件夹、无 `train`，框架可以自动作 80/20 拆分生成 `<root>_split`。**实际项目建议主动固定 train/val/test 划分**，防止数据泄漏并保证复现实验。
- 分类默认是**单标签多类别**：一张图属于一个类别。若要为同一张图同时输出多个独立标签，则需要另行设计多标签任务。
- 分类与检测不同：不会读每张图片的边界框 TXT；如要先识别图中小目标的位置，再给目标分类，需自行准备裁剪图像或级联检测与分类。

## 6. `model.train()` 训练参数——分类任务可使用/重点关注

下面的**默认值来自通用 `modes/train` 表**；“建议/注意”是面向图像分类任务的使用解释。并非每项都有分类页面的独立实测示例。`None` 表示未显式设置。

| 参数 | 类型 | 通用默认值 | 分类训练作用与注意 |
| --- | --- | --- | --- |
| `model` | `str` | `None` | 加载分类 `.pt`/`.yaml`；使用 `*-cls.pt`。 |
| `data` | `str` | `None` | 分类数据集根目录或内置名称（如 `mnist160`、`imagenet10`），不是检测 YAML。 |
| `epochs` | `int` | `100` | 完整训练轮数；先跑短基线再增加。 |
| `time` | `float` | `None` | 最长运行小时数；设置后优先于 `epochs`。 |
| `patience` | `int` | `100` | 验证指标不改进时的早停耐心值。 |
| `batch` | `int/float` | `16` | 批大小；`-1` 为自动约 60% GPU 内存，`0.70` 为约 70%。 |
| `imgsz` | `int` | `640` | 通用默认值；YOLO26-cls 官方基准用 224，分类实际视图片特征选值。 |
| `save` | `bool` | `True` | 保存训练输出与模型检查点。 |
| `save_period` | `int` | `-1` | 每 N 个 epoch 额外保存；-1 不另存周期权重。 |
| `cache` | `bool/str` | `False` | 图像数据缓存：`True`/`ram`、`disk`、`False`。 |
| `device` | `int/str/list` | `None` | `0` CUDA GPU、`cpu`、`mps`、`-1` 自动选空闲 GPU、多卡 `[0,1]` 等。 |
| `workers` | `int` | `8` | DataLoader 工作进程；Windows 或内存紧张时可减小。 |
| `project` | `str` | `None` | 日志与权重所在项目目录。 |
| `name` | `str` | `None` | 本次运行子目录名称。 |
| `exist_ok` | `bool` | `False` | 允许使用已有同名运行目录；避免误覆盖应谨慎开启。 |
| `save_dir` | `str` | `None` | 指定精确的运行输出路径；覆盖 project/name。 |
| `pretrained` | `bool/str` | `True` | 使用预训练权重微调；模型直接加载 `.pt` 更直观。 |
| `cls_remap` | `bool` | `True` | 跨分类数据集微调时按可匹配的类别名复用分类头内容；注意名称和顺序。 |
| `optimizer` | `str` | `'auto'` | 自动选择 AdamW/MuSGD；需强制控制 lr0 则显式指定优化器。 |
| `seed` | `int` | `0` | 伪随机种子；结合 deterministic 改善可复现性。 |
| `deterministic` | `bool` | `True` | 尽量使用确定性算法，速度可能受影响。 |
| `verbose` | `bool` | `True` | 详细训练日志。 |
| `cos_lr` | `bool` | `False` | 采用余弦学习率调度。 |
| `resume` | `bool` | `False` | 从 checkpoint 恢复优化器及 epoch 状态，不等于普通微调。 |
| `amp` | `bool/str` | `True` | 自动混合精度；`fp16`/`bf16`/`fp32` 依设备支持。 |
| `quantize` | `int/str` | `None` | 8/`int8`：QAT，属于特殊进阶流程，需按目标设备验证。 |
| `fraction` | `float/int/list` | `1.0` | 仅取部分数据做快速试验，支持比例/数量/分割列表；生产训练用全量。 |
| `profile` | `bool` | `False` | 训练期间对 ONNX/TensorRT 速度作分析；只在需要时启用。 |
| `freeze` | `int/list` | `None` | 冻结部分层进行迁移学习。 |
| `lr0` | `float` | `0.01` | 初始学习率；`optimizer=auto` 时会忽略用户手设的 `lr0`。 |
| `lrf` | `float` | `0.01` | 最终学习率相对初始学习率的比例。 |
| `momentum` | `float` | `0.937` | 动量或优化器相应参数；自动优化器选择会调整行为。 |
| `weight_decay` | `float` | `0.0005` | 权重衰减，缓解过拟合。 |
| `warmup_epochs` | `float` | `3.0` | 学习率预热周期数。 |
| `warmup_momentum` | `float` | `0.8` | 适用于带动量的优化器，Adam 系列 beta1 不参与该预热。 |
| `warmup_bias_lr` | `float` | `0.1` | 偏置预热学习率；auto 优化器时自动置 0.0。 |
| `nbs` | `int` | `64` | 标称 batch，用于梯度累积/权重衰减缩放。 |
| `dropout` | `float` | `0.0` | **分类专用**：丢弃率，用于分类头正则化。 |
| `val` | `bool` | `True` | 训练期间执行验证。 |
| `split` | `str` | `'val'` | 训练中用 `val` / `test` / `train` 作为验证分割；通常保持 `val`。 |
| `plots` | `bool` | `True` | 绘制并保存曲线和可视化文件。 |
| `compile` | `bool/str` | `False` | PyTorch 2 `torch.compile` 加速选项；不支持会回退。 |
| `channels_last` | `bool` | `None` | 可选内存格式优化；具体设备支持以当前框架为准。 |

### 6.1 优化器特殊说明

- `optimizer="auto"` 时，官网说明较短训练一般选择 AdamW、较长训练可能选择 MuSGD；自动模式还会决定学习率，显式写 `lr0` 不保证生效。
- 如想做严格的学习率对照实验，可以固定 `optimizer="AdamW", lr0=0.001`，再改变 `lrf`、`weight_decay`、`cos_lr`。
- `cls_remap` 涉及新旧类别名称匹配；不要把它当成设置分类类别数量的参数，类别数量通常由数据目录解析。

## 7. 分类数据增强参数（严格按官方任务支持表）

| 参数 | 类型 | 默认值 | 任务适用性 | 含义/限制 |
| --- | --- | --- | --- | --- |
| `auto_augment` | `str` | `randaugment` | classify 专属 | `randaugment` / `autoaugment` / `augmix`，也可尝试 `None`；自动增强。 |
| `erasing` | `float` | `0.4` | classify 专属 | 随机擦除概率；对依赖局部微小特征的场景建议对照关闭。 |
| `hsv_h` | `float` | `0.015` | 分类有条件 | 色调扰动；**内置分类增强仅在 `auto_augment=None` 时应用**。 |
| `hsv_s` | `float` | `0.7` | 分类有条件 | 饱和度扰动；**内置分类增强仅在 `auto_augment=None` 时应用**。 |
| `hsv_v` | `float` | `0.4` | 分类有条件 | 明度扰动；**内置分类增强仅在 `auto_augment=None` 时应用**。 |
| `scale` | `float` | `0.5` | 支持 classify | 缩放/裁剪相关幅度；分类不支持检测可用的 `(min,max)` 元组形式。 |
| `flipud` | `float` | `0.0` | 支持 classify | 上下翻转概率；方向有语义时保持 0。 |
| `fliplr` | `float` | `0.5` | 支持 classify | 左右翻转概率；文字、手势、方向有语义时谨慎开启。 |

> 分类增强的关键事实：官方默认训练变换是 `torchvision.transforms.RandomResizedCrop`，验证/推理默认使用 `CenterCrop`。这与检测中常见的 Mosaic/Letterbox 流程不同；长宽比悬殊、主体靠边的图像有被裁掉的风险。

## 8. 通用训练表里**不应照搬**到分类任务的参数

下面把通用页面里的其它参数集中列出，避免误以为“表里有 = 分类训练一定有效”。“未列入分类支持表”表示不能依赖其在分类训练中生效，不等同于断言底层所有版本都会报错。

| 参数 | 通用默认值 | 为什么不直接用于 classify |
| --- | --- | --- |
| `tile` | `False` | 高分辨率切片及切片标签；分类指南未给出分类支持保证。 |
| `single_cls` | `False` | 检测场景将多个类合并的设置；并非“二分类开关”，普通二分类只建两个类别目录。 |
| `classes` | `None` | 通用训练类别筛选；分类目录任务宜通过数据集明确类别，不建议依赖该参数。 |
| `rect` | `False` | 通用矩形填充；**分类默认采用 torchvision 裁剪**，不要认为它能自动避免裁切。 |
| `multi_scale` | `0.0` | 通用按批次多尺度；分类默认有专门裁剪/缩放增强，不作为首选控制。 |
| `close_mosaic` | `10` | Mosaic 结束轮数；分类官方增强支持列表没有 mosaic。 |
| `distill_model` | `None` | 教师模型蒸馏：分类页面未给出分类用法；需要先核对当前版本。 |
| `dis` | `6.0` | 官方描述为检测蒸馏损失权重，不列为普通分类训练参数。 |
| `box` | `7.5` | 边界框定位损失，非整图分类。 |
| `cls` | `0.5` | 通用表中的分类损失权重，不等同于分类任务的类别数；不要假定修改后会改变分类损失。 |
| `cls_pw` | `0.0` | 类频率加权幂；分类是否实际消费该参数需按当前训练器版本核对。 |
| `dfl` | `1.5` | 框距离回归损失，非整图分类。 |
| `pose` | `12.0` | 姿态损失，非分类。 |
| `kobj` | `1.0` | 关键点目标性损失，非分类。 |
| `rle` | `1.0` | 姿态似然损失，非分类。 |
| `angle` | `1.0` | 旋转框角度损失，非分类。 |
| `dlog` | `1.0` | 深度损失，非分类。 |
| `dgrad` | `0.5` | 深度梯度损失，非分类。 |
| `dlam` | `1.0` | 深度损失设置，非分类。 |
| `overlap_mask` | `True` | 分割掩码重叠策略，非分类。 |
| `mask_ratio` | `4` | 分割掩码分辨率配置，非分类。 |
| `nms` | `None` | 检测头验证与 NMS，非分类。 |
| `max_det` | `300` | 每图检测对象数上限，非分类。 |
| `degrees/translate/shear/perspective` | `见官网` | 通用增强表不把 classify 列为支持任务。 |
| `bgr/mosaic/mixup/cutmix/copy_paste/copy_paste_mode` | `见官网` | 通用增强表不把 classify 列为支持任务。 |
| `augmentations` | `None` | Python 自定义 Albumentations 清单，官网支持任务表**不含 classify**；分类定制应使用 torchvision 数据集/训练器。 |


## 9. 官方分类任务对应的训练代码（按用途重新编写）

### 9.1 安装、核查版本

```bash
pip install -U ultralytics
python -c "import ultralytics; print(ultralytics.__version__)"
```

### 9.2 从官方 `.pt` 预训练模型迁移学习（推荐）

```python
from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")  # 注意 -cls
train_results = model.train(
    data="dataset",               # 分类数据集根目录
    epochs=100,
    imgsz=224,
    batch=16,
    device=0,                    # 无 CUDA GPU 改成 "cpu" 或省略
    patience=20,
    optimizer="auto",
    pretrained=True,
    dropout=0.0,
    auto_augment="randaugment",
    erasing=0.4,
    project="runs/classify",
    name="exp1",
    plots=True,
)
```

命令行等价示例：

```bash
yolo classify train model=yolo26n-cls.pt data=dataset epochs=100 imgsz=224 batch=16 device=0 patience=20 optimizer=auto project=runs/classify name=exp1
```

### 9.3 官方 `mnist160` 小型演示与三种初始化模式

```python
from ultralytics import YOLO

# A. 从零构建：权重随机初始化（通常不是小数据集的首选）
model_a = YOLO("yolo26n-cls.yaml")

# B. 从预训练权重初始化（最常用）
model_b = YOLO("yolo26n-cls.pt")

# C. 先构建结构，再转移预训练权重
model_c = YOLO("yolo26n-cls.yaml").load("yolo26n-cls.pt")

# 下面任选其中一个；MNIST160 示例采用较低 imgsz=64
model_b.train(data="mnist160", epochs=100, imgsz=64)
```

```bash
yolo classify train model=yolo26n-cls.pt data=mnist160 epochs=100 imgsz=64
```

### 9.4 显式学习率、冻结层与继续训练

```python
from ultralytics import YOLO

# 可重复的参数对照实验（把优化器固定下来）
model = YOLO("yolo26n-cls.pt")
model.train(data="dataset", imgsz=224, epochs=80, optimizer="AdamW", lr0=0.001,
            lrf=0.01, weight_decay=0.0005, cos_lr=True, freeze=0,
            project="runs/classify", name="adamw_fixed")

# 中断后从 last.pt 继续：不重新初始化优化器
resume_model = YOLO("runs/classify/exp1/weights/last.pt")
resume_model.train(resume=True)
```

### 9.5 双 GPU 及 Windows 脚本注意

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26n-cls.pt")
    model.train(data="dataset", epochs=100, imgsz=224, device=[0, 1])
```

- Windows 多进程脚本建议加上 `if __name__ == "__main__":`。官方多 GPU 文档还提示，某些 Windows/PyTorch DDP 组合需要使用 Linux 或 WSL2。

## 10. 分类验证与指标

```python
from ultralytics import YOLO

best = YOLO("runs/classify/exp1/weights/best.pt")
metrics = best.val(data="dataset", split="val", imgsz=224)
print("Top-1:", metrics.top1)
print("Top-5:", metrics.top5)
```

```bash
yolo classify val model=runs/classify/exp1/weights/best.pt data=dataset imgsz=224
```

- `metrics.top1`：验证集上真实类别恰好是第一名预测的比例。
- `metrics.top5`：真实类别出现在概率最高的前 5 类中的比例；类别总数少于 5 时辨别力有限。
- `best.pt` 应结合验证集混淆矩阵、各类别召回率、误分类样本共同分析；**不要使用目标检测的 mAP50-95 来评价整图分类**。
- 官方示例允许 `model.val()` 不传参数以沿用模型内保存的训练设置；自定义数据集建议明确写 `data=`，避免路径失效回退到默认演示集。

## 11. 官方分类页面的“自定义变换”能力（保持宽高比）

官网指出默认 `RandomResizedCrop` / `CenterCrop` 可能裁掉长条形图片的有效边缘。要保留整图，可以用 `ClassifyLetterBox` 替换默认裁切，**并同时**定制训练/验证数据集，确保训练和独立验证预处理一致。下面代码是根据官网同一接口重新组织的**可改造示例**，不是原网页代码逐字复制：

```python
import numpy as np
import torch
import torchvision.transforms as T

from ultralytics import YOLO
from ultralytics.data.augment import ClassifyLetterBox
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator


class LetterboxDataset(ClassificationDataset):
    def __init__(self, root, args, augment=False, prefix="", names=None):
        super().__init__(root, args, augment, prefix, names)
        preserve_image = T.Compose([
            np.asarray,
            ClassifyLetterBox(args.imgsz),
            T.ToPILImage(),
        ])
        if augment:
            transforms = [
                preserve_image,
                T.RandomHorizontalFlip(p=args.fliplr),
                T.RandomVerticalFlip(p=args.flipud),
                T.RandAugment(interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
                T.RandomErasing(p=args.erasing, inplace=True),
            ]
        else:
            transforms = [
                preserve_image,
                T.ToTensor(),
                T.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ]
        self.torch_transforms = T.Compose(transforms)


class LetterboxTrainer(ClassificationTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        return LetterboxDataset(
            root=img_path, args=self.args, augment=(mode == "train"),
            prefix="train" if mode == "train" else self.args.split,
            names=self.data["names"],
        )


class LetterboxValidator(ClassificationValidator):
    def build_dataset(self, img_path):
        return LetterboxDataset(
            root=img_path, args=self.args, augment=False,
            prefix=self.args.split, names=self.names,
        )


if __name__ == "__main__":
    model = YOLO("yolo26n-cls.pt")
    model.train(data="dataset", trainer=LetterboxTrainer,
                epochs=100, imgsz=224, batch=16, device=0)
    model.val(data="dataset", validator=LetterboxValidator, imgsz=224)
```

**限制和部署注意：**

1. 上面的 `LetterboxValidator` 用于独立 `model.val`；普通 `model.predict` 默认仍有自己的中心裁剪预处理。若部署时必须保持整个长条形图，应再实现与训练一致的分类 Predictor 前处理，或在输入模型之前预先完成统一 Letterbox，并通过实际样例检查结果。
2. 上面示范 `RandAugment` 是写死的，不表示能完整响应 `auto_augment` 字符串的所有取值。若要保留官网示例中的亮度/颜色 `ColorJitter`，可在 `ToTensor` 前额外添加 `T.ColorJitter(...)`；要精确匹配不同增强策略需自行扩展。
3. 对 Torch/Torchvision/Ultralytics 版本敏感；建议在你安装的目标版本运行并确认尺寸及训练/验证一致，再投入完整训练。

## 12. 常见错误检查单

- 加载了 `yolo26n.pt` 而不是 `yolo26n-cls.pt` → 模型任务错误。
- `data` 指向检测 YAML，而实际是分类文件夹 → 数据格式错误。
- `train` / `val` 各类别文件夹名不一致 → 评估类别映射有问题。
- 数据集存在同一场景/同一对象的近重复图像跨 train/val → 数据泄漏。
- `imgsz=640` 直接照抄检测训练 → 显存、速度与裁切行为未评估。
- 调了 `hsv_*` 但仍用默认 `auto_augment=randaugment` → 内置分类流水线下 `hsv_*` 不会以预期方式应用。
- 把 `single_cls=True` 当成二分类开关 → 错误；分类二分类只需两个类别子目录。
- 训练用自定义 Letterbox，验证仍用默认 CenterCrop → 数据分布不一致。

## 13. 索引

- 预测输出 `result.probs`、Top-1/Top-5 取值与导出格式：请查看同目录 `Ultralytics分类预测参数参考.md`。
- 官方分类接口 https://docs.ultralytics.com/zh/tasks/classify
- 官方训练模式 https://docs.ultralytics.com/zh/modes/train
