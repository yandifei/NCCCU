# Ultralytics 分类预测参数参考（通用参数完整版 + Classify 专项）

> 版本：2026-11-07。**本文件首先完整保留通用 Predict 模式参数，再单独列出分类预测接口和代码**。同一套 `model.predict()` 参数覆盖多个任务，但检测框/NMS 参数不等于分类概率阈值。

## 官方和原始资料

- [通用预测模式](https://docs.ultralytics.com/zh/modes/predict)
- [分类任务说明](https://docs.ultralytics.com/zh/tasks/classify)
- [通用训练模式](https://docs.ultralytics.com/zh/modes/train)
- 本文「完整通用推理参数」「完整通用可视化参数」「输入源」及「Results/plot 接口」以你上传的《Ultralytics预测参数参考.md》为基础保留原表的参数名称、类型、默认值、说明，并新增分类适用性标注。`tile` 是对照官网补充的条目。
- `✅ 通用可用`：分类可使用；`⚠️ 条件适用/需验证`：对分类效果可能无意义或依版本/后端；`— 其他任务/分类不使用`：主要为检测框、分割、姿态等任务。

## 1. 完整通用推理参数表（`model.predict()`）

**共 27 项**：保留原文件全部推理参数，另增官网 `tile`。注意：此表不是仅对 `classify` 生效的参数集合。

| 参数 | 类型 | 默认值 | 描述 | 分类适用性 |
| --- | --- | --- | --- | --- |
| `source` | `str` 或 `int` 或 `None` | `None` | 指定推理的数据源。可以是图像路径、视频文件、目录、URL 或实时流的设备 ID。如果省略，系统会记录警告，并将模型回退到内置演示资源（`ultralytics/assets`，或 OBB 的演示 URL）。支持多种格式和数据源，可灵活应用于[不同类型的输入](https://docs.ultralytics.com/modes/predict#inference-sources)。 | ✅ 通用可用 |
| `conf` | `float` | `0.25` | 设置检测的最低置信度阈值。置信度低于此阈值的检测对象将被忽略。调整此值有助于减少误报。 | — 其他任务/分类不使用 |
| `iou` | `float` | `0.7` | [交并比](https://www.ultralytics.com/glossary/intersection-over-union-iou) (IoU) 阈值，用于非极大值抑制 (NMS)。较低的值会通过消除重叠框来减少检测结果，有助于减少重复检测。 | — 其他任务/分类不使用 |
| `imgsz` | `int` 或 `tuple` | `640` | Letterbox 目标尺寸。整数值表示正方形 `N×N`；元组表示 `(height, width)`。使用 `rect=True` 时，由于最小矩形填充，实际张量可能小于此目标尺寸。使用 `rect=False` 可指定固定尺寸。参见[固定形状与最小矩形](https://docs.ultralytics.com/modes/predict#fixed-shape-vs-minimum-rectangle-rect)。 | ✅ 通用可用 |
| `rect` | `bool` | `True` | 如果为 `True`，则在可能时使用最小矩形填充（相同形状的批次和受支持的后端）。如果为 `False`，则始终填充到完整的 `imgsz`。参见[固定形状与最小矩形](https://docs.ultralytics.com/modes/predict#fixed-shape-vs-minimum-rectangle-rect)。 | ⚠️ 条件适用/需验证 |
| `quantize` | `int` 或 `str` | `None` | 推理精度：`16`/`"fp16"` 与 `32`/`"fp32"`/未设置为 PyTorch 和 TorchScript 模型选择 FP16 或 FP32 计算；其他格式则在其构件与运行时所选的精度下进行计算。在 `16` 下，OpenVINO 仍在客户端将输入进行 FP16 舍入并将其展宽回 FP32，但这不会改变运行时实际的计算精度。INT8/PTQ 量化在[导出](https://docs.ultralytics.com/modes/export#quantization-options)期间进行配置，然后通过加载导出的模型来使用。该参数替代了已弃用的 `half` 标志。 | ✅ 通用可用 |
| `device` | `str` | `None` | 指定推理设备（例如 `cpu`、`cuda:0`、`0`、`npu` 或 `npu:0`）。允许你在 CPU、特定 GPU、华为 Ascend NPU 或其他计算设备之间进行选择，以执行模型推理。 | ✅ 通用可用 |
| `dnn` | `bool` | `False` | 如果为 `True`，则在 ONNX 模型推理中使用 [OpenCV](https://www.ultralytics.com/glossary/opencv) DNN 模块，而不是 ONNX Runtime。 | ✅ 通用可用 |
| `data` | `str` | `None` | 数据集 YAML 的路径（例如 `coco8.yaml`），仅读取其中的 `names`，且仅当加载的模型不包含自身的类别名称时使用：例如第三方导出模型，或与其附带元数据分离的 Ultralytics 导出模型。否则，此类模型会报告 `class0`、`class1` 等。 | ✅ 通用可用 |
| `batch` | `int` | `1` | 指定推理的批次大小（仅当源是[目录、视频文件或 `.txt` 文件](https://docs.ultralytics.com/modes/predict#inference-sources)时有效）。较大的批次大小可以提供更高的吞吐量，从而缩短推理所需的总时间。 | ✅ 通用可用 |
| `tile` | `bool` 或 `int` | `False` | 固定大小切片推理并把结果映射回原图坐标；用于高分辨率目标任务。2026-11-07 官网有此参数，原上传表未含。 | ⚠️ 条件适用/需验证 |
| `max_det` | `int` | `300` | 每张图像允许的最大检测数量。限制模型在单次推理中可以检测的对象总数，防止密集场景产生过多输出。 | — 其他任务/分类不使用 |
| `vid_stride` | `int` | `1` | 视频输入的帧步长。允许跳过视频帧以加快处理速度，但会降低时间分辨率。值为 1 时处理每一帧，值更高时跳过部分帧。 | ✅ 通用可用 |
| `stream_buffer` | `bool` | `False` | 决定是否为视频流排队接收到的帧。如果为 `False`，则会丢弃旧帧以容纳新帧（针对实时应用进行了优化）。如果为 `True`，则会将新帧加入缓冲区，确保不跳过任何帧，但当推理 FPS 低于流 FPS 时会产生延迟。 | ✅ 通用可用 |
| `visualize` | `bool` | `False` | 在每个预测结果旁保存类别激活热力图，显示哪些像素提高了预测类别的分数。遵循 `conf` 和 `classes`，因此 `classes=[0]` 仅映射该类别。仅适用于 Ultralytics PyTorch 模型。 | ⚠️ 条件适用/需验证 |
| `augment` | `bool` | `False` | 为预测启用测试时增强 (TTA)，可能提升检测鲁棒性，但会降低推理速度。仅适用于 Ultralytics PyTorch 模型。 | ⚠️ 条件适用/需验证 |
| `agnostic_nms` | `bool` | `False` | 启用类无关的 NMS，压制不同类别中得分较低的重叠框，而不仅仅是同一类别内。在类别重叠常见的多类检测场景中很有用。使用无 NMS 推理（在 YOLO26 或 YOLOv10 上为 `nms=False`）时，这仅防止同一检测结果出现多个类别标签（IoU=1.0 重复），而不会在不同框之间执行基于 IoU 阈值的压制。 | — 其他任务/分类不使用 |
| `classes` | `list[int]` | `None` | 将预测结果过滤为一组类别 ID。仅返回属于指定类别的检测结果。适用于在多类别检测任务中聚焦相关对象。 | ⚠️ 条件适用/需验证 |
| `retina_masks` | `bool` | `False` | 返回高分辨率分割掩码。启用后，返回的掩码（`masks.data`）将匹配原始图像尺寸。禁用后，掩码将使用推理期间的图像尺寸。 | — 其他任务/分类不使用 |
| `embed` | `list[int]` | `None` | 指定用于提取特征向量或[嵌入](https://www.ultralytics.com/glossary/embeddings)的层。使用 `model.embed(source)` 获取倒数第二层嵌入，或使用 `model.predict(source, embed=[layer])` 选择特定层。适用于聚类或相似性搜索等下游任务。仅适用于 Ultralytics PyTorch 模型。 | ✅ 通用可用 |
| `project` | `str` | `None` | 项目目录的名称；启用 `save` 后，预测输出将保存到该目录。 | ✅ 通用可用 |
| `name` | `str` | `None` | 预测运行的名称。用于在项目文件夹中创建子目录；启用 `save` 后，预测输出将保存到该子目录。 | ✅ 通用可用 |
| `stream` | `bool` | `False` | 通过返回 Results 对象生成器，而不是一次性将所有帧加载到内存中，为长视频或大量图像启用节省内存的处理方式。 | ✅ 通用可用 |
| `verbose` | `bool` | `True` | 控制是否在终端中显示详细的推理日志，为预测过程提供实时反馈。 | ✅ 通用可用 |
| `compile` | `bool` 或 `str` | `False` | 使用 `backend='inductor'` 启用 PyTorch 2.x 的 `torch.compile` 图编译。接受 `True` → `"default"`、`False` → 禁用，或使用 `"default"`、`"reduce-overhead"`、`"max-autotune-no-cudagraphs"` 等字符串模式。不支持时会发出警告并回退到 eager 模式。 | ✅ 通用可用 |
| `channels_last` | `bool` | `None` | 对原生 PyTorch 推理使用 channels_last (NHWC) 内存格式。`None` 会在启用 oneDNN 的 Linux 和 Windows x86 CPU 上、PyTorch 1.13 或更高版本中自动启用该格式，`False` 会禁用该格式，`True` 会在受支持的 x86 CPU 或 CUDA 设备上请求该格式。ARM64、MPS、较旧的 PyTorch 版本、不支持 oneDNN 的 CPU，以及 TensorRT 和 ONNX 等导出格式均保持不变。 | ✅ 通用可用 |
| `nms` | `bool`，可选 | `None` | 默认运行带 NMS 的一对多推理（`None` 或 `True`）。设置 `False` 以在可用时使用无 NMS 的一对一（one-to-one）头。详见[端到端检测指南](https://docs.ultralytics.com/guides/end2end-detection)。 | — 其他任务/分类不使用 |

### 1.1 通用 `rect` / `imgsz` 预处理说明

- 通用预测模式的 `rect=True` 描述的是可能应用的**最小矩形 Letterbox**；当图像批次尺寸一致且后端允许时可能得到小于 `imgsz` 的最终张量。
- 通用 `imgsz=640` 是默认输入尺寸参数，并非分类任务建议尺寸。YOLO26 分类预训练模型通常使用 `imgsz=224` 进行评估。
- **分类任务额外限制**：默认分类预测变换通常包含 `CenterCrop`，不能认为把 `rect=False` 就能保留完整图片；若有边缘缺陷/长条图像，应统一训练与预测的自定义变换。
- 长视频或海量图片建议开启 `stream=True`，以迭代方式读取结果，降低内存压力。

## 2. 完整通用预测可视化参数表

**共 10 项**：完整保留原文件的预测可视化设置，检测专用项也不删。

| 参数 | 类型 | 默认值 | 描述 | 分类适用性 |
| --- | --- | --- | --- | --- |
| `show` | `bool` | `False` | 如果为 `True`，则会在窗口中显示带标注的图像或视频。适用于开发或测试期间获取即时视觉反馈。 | ✅ 通用可用 |
| `save` | `bool` | `False or True` | 启用后将带注释的图像或视频保存到文件。适用于文档记录、进一步分析或分享结果。使用 CLI 时默认为 True，使用 Python 时默认为 False。 | ✅ 通用可用 |
| `save_frames` | `bool` | `False` | 处理视频时，将单独的帧保存为图像。适用于提取特定帧或进行详细的逐帧分析。 | ✅ 通用可用 |
| `save_txt` | `bool` | `False` | 将检测结果按照 `[class] [x_center] [y_center] [width] [height] [confidence]` 格式保存到文本文件中。适用于与其他分析工具集成。 | ⚠️ 条件适用/需验证 |
| `save_conf` | `bool` | `False` | 在保存的文本文件中包含置信度分数。增加可用于后处理和分析的详细信息。 | ⚠️ 条件适用/需验证 |
| `save_crop` | `bool` | `False` | 保存检测对象的裁剪图像。适用于数据集增强、分析，或为特定对象创建专用数据集。 | — 其他任务/分类不使用 |
| `show_labels` | `bool` | `True` | 在可视化输出中显示每个检测结果的标签，帮助你立即了解检测到的对象。 | ✅ 通用可用 |
| `show_conf` | `bool` | `True` | 在每个检测结果旁显示其置信度分数，让你了解模型对每个检测结果的确定程度。 | ✅ 通用可用 |
| `show_boxes` | `bool` | `True` | 在检测到的对象周围绘制边界框。对于在图像或视频帧中直观识别和定位对象至关重要。 | — 其他任务/分类不使用 |
| `line_width` | `int or None` | `None` | 指定边界框的线宽。如果为 `None`，则会根据图像尺寸自动调整线宽。可自定义视觉效果以提高清晰度。 | — 其他任务/分类不使用 |

## 3. 完整通用推理源（`source`）表

| 来源 | 示例 | 类型 | 备注 |
|---|---|---|---|
| 图像 | `'image.jpg'` | `str` 或 `Path` | 单个图像文件。 |
| URL | `'https://ultralytics.com/images/bus.jpg'` | `str` | 图像 URL。 |
| 屏幕截图 | `'screen'` | `str` | 截取屏幕截图。 |
| PIL | `Image.open('image.jpg')` | `PIL.Image` | 包含 RGB 通道的 HWC 格式。 |
| [OpenCV](https://www.ultralytics.com/glossary/opencv) | `cv2.imread('image.jpg')` | `np.ndarray` | 包含 BGR 通道的 HWC 格式 `uint8 (0-255)`。 |
| NumPy | `np.zeros((640,1280,3))` | `np.ndarray` | 包含 BGR 通道的 HWC 格式 `uint8 (0-255)`。 |
| torch | `torch.zeros(16,3,320,640)` | `torch.Tensor` | 包含 RGB 通道的 BCHW 格式 `float32 (0.0-1.0)`。 |
| CSV | `'sources.csv'` | `str` 或 `Path` | 包含图像、视频或目录路径的 CSV 文件。 |
| 视频 ✅ | `'video.mp4'` | `str` 或 `Path` | MP4、AVI 等格式的视频文件。 |
| 目录 ✅ | `'path/'` | `str` 或 `Path` | 包含图像或视频的目录路径。 |
| glob ✅ | `'path/*.jpg'` | `str` | 用于匹配多个文件的 Glob 模式。使用 `*` 字符作为通配符。 |
| YouTube ✅ | `'https://youtu.be/LNwODJXcvt4'` | `str` | YouTube 视频 URL。 |
| 流 ✅ | `'rtsp://example.com/media.mp4'` | `str` | 用于 RTSP、RTMP、TCP 等流式协议的 URL，或 IP 地址。 |
| 多流 ✅ | `'list.streams'` | `str` 或 `Path` | `*.streams` 文本文件，每行包含一个流 URL，即 8 个流将以批次大小 8 运行。 |
| 摄像头 ✅ | `0` | `int` | 要执行推理的已连接摄像头设备索引。 |

> `stream=True` 是否适用由输入源类型决定，`stream` 控制的是结果返回形式而不是模型任务。视频、摄像头、目录等数据源可以逐帧/逐图分类。

## 4. `Results` 通用返回属性（所有任务共用接口）

| 属性 | 类型 | 描述 |
|---|---|---|
| `orig_img` | `np.ndarray` | 作为 NumPy 数组的原始图像。 |
| `orig_shape` | `tuple` | 采用 (height, width) 格式的原始图像形状。 |
| `boxes` | `Boxes, optional` | 包含检测边界框的 Boxes 对象。 |
| `masks` | `Masks, optional` | 包含检测掩码的 Masks 对象。 |
| `probs` | `Probs, optional` | 包含分类任务中每个类别概率的 Probs 对象。 |
| `keypoints` | `Keypoints, optional` | 包含每个对象检测到的关键点的 Keypoints 对象。 |
| `obb` | `OBB, optional` | 包含定向边界框的 OBB 对象。 |
| `semantic_mask` | `SemanticMask, optional` | 包含密集逐像素类别映射的 SemanticMask 对象。 |
| `speed` | `dict` | 一个字典，记录预处理、推理和后处理速度，单位为每张图像的毫秒数。 |
| `names` | `dict` | 将类别索引映射到类别名称的字典。 |
| `path` | `str` | 图像文件的路径。 |
| `save_dir` | `str, optional` | 用于保存结果的目录。 |

> `boxes`、`masks`、`keypoints`、`obb` 等在通用结果对象里有字段，但普通整图分类的核心输出是 `result.probs`。

## 5. `Results.plot()` 通用绘制参数完整表

| 参数 | 类型 | 描述 | 默认值 | 分类适用性 |
| --- | --- | --- | --- | --- |
| `conf` | `bool` | 包含检测置信度分数。 | `True` | ✅ 通用可用 |
| `line_width` | `float` | 边界框的线宽。如果为 `None`，则会随图像尺寸缩放。 | `None` | — 非分类重点 |
| `font_size` | `float` | 文本字体大小。如果为 `None`，则会随图像尺寸缩放。 | `None` | ✅ 通用可用 |
| `font` | `str` | 文本标注的字体名称。 | `'Arial.ttf'` | ✅ 通用可用 |
| `pil` | `bool` | 将图像作为 PIL Image 对象返回。 | `False` | ✅ 通用可用 |
| `img` | `np.ndarray \| torch.Tensor` | 备用图像。张量必须是连续的 HWC BGR uint8。 | `None` | ✅ 通用可用 |
| `kpt_radius` | `int` | 绘制关键点的半径。 | `5` | — 非分类重点 |
| `kpt_line` | `bool` | 使用线条连接关键点。 | `True` | — 非分类重点 |
| `labels` | `bool` | 在标注中包含类别标签。 | `True` | ✅ 通用可用 |
| `boxes` | `bool` | 在图像上叠加边界框。 | `True` | — 非分类重点 |
| `masks` | `bool` | 在图像上叠加掩码。 | `True` | — 非分类重点 |
| `probs` | `bool` | 包含分类概率。 | `True` | ⭐ 分类结果绘制关键项 |
| `show` | `bool` | 使用默认图像查看器直接显示标注后的图像。 | `False` | ✅ 通用可用 |
| `save` | `bool` | 将标注后的图像保存到 `filename` 指定的文件。 | `False` | ✅ 通用可用 |
| `filename` | `str` | 当 `save` 为 `True` 时，用于保存标注后图像的文件路径和名称。 | `None` | ✅ 通用可用 |
| `color_mode` | `str` | 指定颜色模式，例如“instance”或“class”。 | `'class'` | ✅ 通用可用 |
| `txt_color` | `tuple[int, int, int]` | 边界框和图像分类标签的 BGR 文本颜色。 | `(255, 255, 255)` | ✅ 通用可用 |

> `plot(probs=True)` 是分类概率显示重点；`plot(boxes=True)` 是检测框相关，与分类无关。它们属于 `Results.plot()` 的参数，不能与 `model.predict()` 的同名参数混用。

## 6. 分类专用预测参数与输出接口（单独保留）

**区别于训练**：官方 `tasks/classify` 并没有再定义一套独立于通用 `model.predict()` 的全新关键词参数表；分类的专用部分主要体现为 `-cls` 模型、中心裁剪预处理、`Results.probs` 输出和 Top-1/Top-5 的读取方式。

### 6.1 分类预测最常使用的通用参数

| `predict()` 参数 | 通用默认值 | 分类时的作用 / 推荐写法 |
|---|---|---|
| `source` | `None` | 单图、目录、视频流等；分类返回整图类别概率。 |
| `imgsz` | `640` | 分类基线常显式指定 `224` 或训练时使用的尺寸。 |
| `batch` | `1` | 批量目录/视频推理适当提高吞吐。 |
| `device` | `None` | 用 GPU、CPU、MPS 等。 |
| `stream` | `False` | 图片很多时用 `True` 返回生成器。 |
| `quantize` | `None` | 原生权重 FP16/FP32 运行配置，依设备及导出格式。 |
| `save` | Python `False` / CLI `True` | 需要可视化图片时手动打开。 |
| `project` / `name` | `None` / `None` | 管理输出目录。 |
| `show` / `show_labels` / `show_conf` | `False` / `True` / `True` | 控制显示和结果文字；取概率值仍通过 `result.probs`。 |
| `conf` | `0.25` | **这是检测阈值**；不能用来筛选分类 Top-1。若要拒识，后处理比较 `top1conf`。 |
| `iou` / `nms` / `agnostic_nms` | `0.7` / `None` / `False` | 检测框/NMS 相关；整图分类不需要。 |

### 6.2 分类模型专用输出字段（`result.probs`）

| 属性 | 类型 | 形状 | 意义 |
|---|---|---|---|
| `result.probs` | `Probs` | `(C,)` | 当前图片的类别概率对象。 |
| `result.probs.data` | `torch.float32`（低精度运行可能是 FP16） | `(C,)` | 对应全部 `C` 个类别的概率向量。 |
| `result.probs.top1` | `int` | 标量 | 最大概率的类别 ID。 |
| `result.probs.top1conf` | Tensor | 标量 | 最大概率的类别分数。 |
| `result.probs.top5` | `list[int]` | `(<=5)` | 前五类别索引。 |
| `result.probs.top5conf` | Tensor | `(<=5)` | Probs 通用接口提供的前五分数。 |
| `result.names` | `dict` | 类别 ID → 名称 | 查出类别名称。 |

### 6.3 `Probs` 通用方法和属性完整表

| 名称 | 类型 | 描述 |
|---|---|---|
| `cpu()` | 方法 | 返回位于 CPU 内存中的 probs 张量副本。 |
| `numpy()` | 方法 | 以 NumPy 数组形式返回 probs 张量副本。 |
| `cuda()` | 方法 | 返回位于 GPU 内存中的 probs 张量副本。 |
| `to()` | 方法 | 返回位于指定设备上并具有指定 dtype 的 probs 张量副本。 |
| `top1` | 属性（`int`） | 排名第 1 的类别索引。 |
| `top5` | 属性（`list[int]`） | 排名前 5 的类别索引。 |
| `top1conf` | 属性（`torch.Tensor`） | 排名第 1 的类别置信度。 |
| `top5conf` | 属性（`torch.Tensor`） | 排名前 5 的类别置信度。 |

### 6.4 最短可运行分类预测代码

```python
from ultralytics import YOLO

model = YOLO("runs/classify/baseline/weights/best.pt")
results = model.predict(source="test.jpg", imgsz=224)
for r in results:
    print(r.path, r.names[r.probs.top1], float(r.probs.top1conf))
```

```bash
yolo classify predict model=runs/classify/baseline/weights/best.pt source=test.jpg imgsz=224
```

---

## 后续：分类预测、批量处理、CSV、验证与导出操作说明

以下延续先前的分类预测版，包含输入源示例、分类输出字段及操作、流式推理、阈值后处理、CSV 导出、ONNX 导出等。前面的**通用参数表不因为下面的分类专项总结而省略任何行**。

## 7. 核心概念和最短示例

- 通用 `modes/predict` 适用于多种模型任务；**使用 `-cls` 权重才能获得分类 `result.probs`**。
- 一张图对应一个 `Results`，`probs` 是长度为类别数 `C` 的概率向量；`top1` 是最高概率的类别编号，`top1conf` 是该类别的置信度。
- 与 Detect 不同：整图分类没有需要消费的 `result.boxes.xyxy` 等边界框字段。

```python
from ultralytics import YOLO

model = YOLO("runs/classify/exp1/weights/best.pt")
results = model.predict(source="test.jpg", imgsz=224)
for result in results:
    label_id = result.probs.top1
    label_name = result.names[label_id]
    score = float(result.probs.top1conf)
    print(result.path, label_name, round(score, 4))
```

```bash
yolo classify predict model=runs/classify/exp1/weights/best.pt source=test.jpg imgsz=224
```

## 8. `source` 输入类型及示例（通用推理模式；分类模型按帧/图输出概率）

| 输入 | `source` 示例 | 说明 |
| --- | --- | --- |
| 单张图片 | `"image.jpg"` | 一次返回一个 Results |
| HTTP(S) 图片 | `"https://.../image.jpg"` | 远程图片 URL |
| 图片目录 | `"dataset/test"` | 对目录下可处理的图像逐张预测；推荐 stream=True |
| glob 匹配 | `"images/*.jpg"` | 按通配符匹配图片 |
| PIL 图像 | `Image.open("image.jpg")` | PIL RGB 图片 |
| OpenCV | `cv2.imread("image.jpg")` | NumPy HWC、BGR、uint8 |
| NumPy | `np.zeros((224,224,3),dtype=np.uint8)` | HWC、BGR、uint8 |
| PyTorch Tensor | `torch.zeros(1,3,224,224)` | BCHW、RGB、float32、取值 0~1 |
| 视频 | `"video.mp4"` | 逐帧分类；是否值得分类取决于业务 |
| 摄像头 | `0` | 实时逐帧分类 |
| RTSP/其他流 | `"rtsp://..."` | 视频流分类 |
| CSV | `"sources.csv"` | 从路径列表读取数据源 |
| `.streams` | `"cameras.streams"` | 多视频流 |
| 屏幕 | `"screen"` | 持续屏幕截图 |
| YouTube | `"https://youtu.be/..."` | 视频在线来源，依赖环境/工具 |

> 内存提示：多张图片、文件夹及长视频优先用 `stream=True`；不使用流时，预测对象会被收集到列表，可能占用较多内存。官方支持的图像扩展名包括 `avif/bmp/dng/heic/heif/jp2/jpeg/jpg/mpo/png/tif/tiff/webp`，视频包括 `avi/mp4/mov/mkv/webm` 等。文件格式是否可读还取决于环境安装的解码依赖。

## 9. 通用预测参数中，分类任务常用/可考虑的项

默认值来自官网 `modes/predict`，后面是分类任务解释。**分类模型的图像预处理通常是 CenterCrop；不能把检测模型的 `rect/Letterbox` 描述自动套在分类模型上。**

| 参数 | 类型 | 通用默认值 | 对分类任务的意义 |
| --- | --- | --- | --- |
| `source` | `str/int/None` | `None` | 预测图片、目录、视频、流、摄像头等。 |
| `imgsz` | `int/tuple` | `640` | 模式页通用默认；分类模型推荐明确设置训练采用的分辨率，如 224。 |
| `device` | `str/None` | `None` | `0`、`cpu`、`mps` 等，按运行环境设置。 |
| `quantize` | `int/str` | `None` | 原生 PyTorch/TorchScript 的 FP16/FP32 精度选择；导出的 INT8 在导出阶段配置。 |
| `dnn` | `bool` | `False` | 使用 OpenCV DNN 运行 ONNX；仅在 ONNX 场景有意义。 |
| `batch` | `int` | `1` | 对目录/视频/文本文件等部分来源生效；通常增大可提高吞吐。 |
| `vid_stride` | `int` | `1` | 跳帧：如 2 表示间隔采样，适用于视频分类。 |
| `stream_buffer` | `bool` | `False` | 视频流缓冲；False 允许丢帧保实时，True 保留帧但可能积压。 |
| `embed` | `list[int]` | `None` | 提取中间特征；适用于相似度检索/特征分析（官方限制原生 PyTorch 模型）。 |
| `project` | `str` | `None` | `save=True` 时用于控制输出文件夹。 |
| `name` | `str` | `None` | 本次预测的输出子目录。 |
| `stream` | `bool` | `False` | True 返回生成器，适合大量文件和持续视频。 |
| `verbose` | `bool` | `True` | 控制日志。 |
| `compile` | `bool/str` | `False` | 可选 `torch.compile`，不支持时回退。 |
| `channels_last` | `bool` | `None` | 原生 PyTorch 内存格式性能选项，须按设备验证。 |

### 9.1 在通用表中出现，但不要直接按检测语义配置的参数

| 参数 | 通用默认值 | 分类中的处理原则 |
| --- | --- | --- |
| `conf` | `0.25` | 最低检测框置信度阈值；**不能指望它过滤分类的 Top-1**。如需拒识，在 `result.probs.top1conf` 上自行判断。 |
| `iou` | `0.7` | NMS 框重叠阈值，非整图分类。 |
| `rect` | `True` | 文档描述针对检测的 Letterbox/最小矩形；分类默认使用中心裁剪，设置该值不能保证分类保留全图。 |
| `data` | `None` | 常用于缺失类别名元数据的导出检测模型；正常 `-cls` 权重通常无需另传。 |
| `tile` | `False` | 高分辨率切片检测相关；不要假定分类会按 tile 做聚合分类。 |
| `max_det` | `300` | 最大检测目标数，分类无框。 |
| `visualize` | `False` | 通用模式的类别激活热图可视化选项；并非通用分类结果接口，需在具体 PyTorch 版本实测。 |
| `augment` | `False` | 预测时增强 TTA；官网主要以检测鲁棒性描述，分类是否产生预期增强请实测。 |
| `agnostic_nms` | `False` | 类别无关 NMS，非分类。 |
| `classes` | `None` | 通用文档描述为筛选检测结果；分类 Top-1 不应靠它筛选。 |
| `retina_masks` | `False` | 分割掩码分辨率，非分类。 |
| `nms` | `None` | 检测头选择，非分类。 |

## 10. 预测结果可视化与保存选项

| 参数 | 通用默认 | 分类场景建议 |
|---|---|---|
| `show` | `False` | 可在有图形界面的环境显示可视化；服务器一般关闭。 |
| `save` | Python `False` / CLI `True` | 若需保存带分类结果的图片，设置 `save=True`。 |
| `save_frames` | `False` | 视频逐帧保存；大量帧会占存储。 |
| `show_labels` | `True` | 显示标签的通用可视化选项，实际绘制方式依模型任务。 |
| `show_conf` | `True` | 显示分数的通用可视化选项。 |
| `save_txt` | `False` | 官方通用页描述的是**检测框 TXT 格式**，分类结果建议自行写 CSV/JSON。 |
| `save_conf` | `False` | 通用 `save_txt` 相关，不代表自动导出分类概率列表。 |
| `save_crop` | `False` | 导出检测裁剪目标，整图分类通常不需要。 |
| `show_boxes` | `True` | 检测框绘制开关，非分类。 |
| `line_width` | `None` | 检测框线宽，非分类。 |

## 11. 分类输出接口：`Results` 和 `Probs`

### 11.1 核心分类字段（官网 classify 页面）

| 属性 | 类型 | 形态 | 含义 |
| --- | --- | --- | --- |
| `result.probs` | Probs | (C,) | 本图所有类别的概率对象 |
| `result.probs.data` | Tensor | (C,) | 类别概率序列；一般 FP32，低精度执行可为 FP16 |
| `result.probs.top1` | int | 标量 | 排名第一的类别索引 |
| `result.probs.top1conf` | Tensor | 标量 | 排名第一的置信度 |
| `result.probs.top5` | list[int] | 最多 5 个 | 按概率从高到低排列的前五个类别索引 |
| `result.probs.top5conf` | Tensor | 最多 5 个 | 前五名对应分数（来自通用 Probs API，若目标版本提供） |
| `result.names` | dict | ID→名称 | 类别名称映射 |
| `result.path` | str | 文件路径 | 输入图像路径 |
| `result.orig_img` | NumPy 数组 | 原始图像 | 原图像素 |
| `result.orig_shape` | tuple | (H,W) | 原图尺寸 |
| `result.speed` | dict | 毫秒 | 预处理、推理和后处理各阶段耗时 |
| `result.save_dir` | str/None | 目录 | 输出保存目录（可能不存在） |

> 注意：官网 **classify 任务页**明确列出 `probs`、`data`、`top1`、`top1conf`、`top5`；`top5conf` 为 Probs 通用 API 的补充项，如果运行的版本没有此属性，可用 `result.probs.data[result.probs.top5]` 取值。

### 11.2 读取 Top-1、Top-5 和完整概率向量

```python
from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")
result = model.predict("test.jpg", imgsz=224)[0]
probs = result.probs

print("Top-1 ID:", probs.top1)
print("Top-1 label:", result.names[probs.top1])
print("Top-1 confidence:", float(probs.top1conf))

for i in probs.top5:
    print(f"{i}: {result.names[i]} -> {float(probs.data[i]):.4f}")

all_scores = probs.data.cpu().tolist()  # 顺序对应模型的 class ID
```

### 11.3 批量/流式分类（图片文件夹）

```python
from ultralytics import YOLO

model = YOLO("runs/classify/exp1/weights/best.pt")
for r in model.predict(source="images/", imgsz=224, batch=16, stream=True):
    print(r.path, r.names[r.probs.top1], float(r.probs.top1conf))
```

### 11.4 类别置信度低时拒识：自己实现阈值

```python
from ultralytics import YOLO

model = YOLO("runs/classify/exp1/weights/best.pt")
threshold = 0.75
for r in model.predict(source="test_images/", imgsz=224, stream=True):
    p = float(r.probs.top1conf)
    predicted = r.names[r.probs.top1] if p >= threshold else "unknown"
    print(r.path, predicted, p)
```

**重要**：这里的 `unknown` 是业务后处理，而不是模型增加了训练类别。若需要“其他/未知”类有监督学习，应在数据集中准备对应训练样本。

### 11.5 分类结果导出 CSV（比检测 TXT 更合适）

```python
import csv
from ultralytics import YOLO

model = YOLO("runs/classify/exp1/weights/best.pt")
with open("classify_predictions.csv", "w", encoding="utf-8-sig", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["path", "class_id", "label", "confidence"])
    for r in model.predict(source="images/", stream=True, imgsz=224):
        idx = r.probs.top1
        writer.writerow([r.path, idx, r.names[idx], float(r.probs.top1conf)])
```

### 11.6 `Results` 常用方法

| 方法 | 作用 |
|---|---|
| `show()` | 显示结果图片；需要 GUI 环境。 |
| `save(filename=...)` | 保存可视化结果。 |
| `plot()` | 返回带任务相关标注的 BGR NumPy 图片。 |
| `cpu()` / `cuda()` | 把结果数据搬到 CPU / CUDA。 |
| `numpy()` | 将结果张量转成 NumPy。 |
| `to(device=...)` | 按设备/数据类型转换。 |
| `summary()` | 创建结构化摘要（输出结构随版本、任务改变）。 |
| `to_json()` / `to_csv()` / `to_df()` | 把结果格式化输出（可用性依当前版本）。 |
| `save_txt()` / `save_crop()` | 主要面向检测/裁剪场景，不作为分类结果首选。 |

### 11.7 `Results.plot()` 参数（通用接口中与分类展示相关的配置）

官方通用 Predict 页面还有 `plot()` 的细粒度显示设置，这与 `predict()` 参数表是两个不同层次：

| 参数 | 默认值 | 分类中用途 |
|---|---|---|
| `probs` | `True` | 控制是否绘出分类概率信息，**分类展示的重要开关**。 |
| `labels` | `True` | 绘制类别文本标签。 |
| `conf` | `True` | 绘制置信度文本；不是检测阈值。 |
| `font_size` | `None` | 标注文字大小。 |
| `font` | `Arial.ttf` | 文本字体的名称（运行环境要有对应字体）。 |
| `txt_color` | `(255,255,255)` | 文本 BGR 颜色。 |
| `pil` | `False` | 选择返回 PIL Image（否则常为 BGR NumPy）。 |
| `img` | `None` | 可传入另一张图像用于绘制。 |
| `show` | `False` | 绘制时额外显示图片。 |
| `save` | `False` | 绘制时直接保存图片。 |
| `filename` | `None` | 搭配 `save=True` 的输出文件名。 |
| `boxes` / `masks` | `True` | 检测/分割绘图相关，分类无需设置。 |
| `line_width` | `None` | 检测框宽度，分类无需设置。 |
| `kpt_radius` / `kpt_line` | `5` / `True` | 姿态关键点相关，分类无需设置。 |
| `color_mode` | `'class'` | 通用可视化颜色模式。 |

```python
from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")
result = model.predict("test.jpg", imgsz=224)[0]
annotated = result.plot(probs=True, labels=True, conf=True, font_size=18)
# annotated 是 BGR 图像，可以用 cv2.imwrite 保存
import cv2
cv2.imwrite("classification_visualized.jpg", annotated)
```

## 12. 验证与预测有什么区别？

- `model.predict(...)`：只对给定图片/流计算分类结果，不要求真值标签。
- `model.val(data="dataset", split="val")`：读取有标签的验证集，计算 Top-1 与 Top-5 Accuracy 等指标。

```python
from ultralytics import YOLO

model = YOLO("runs/classify/exp1/weights/best.pt")
metric = model.val(data="dataset", split="val", imgsz=224)
print(metric.top1, metric.top5)
```

## 13. 分类模型导出（classify 页面完整格式目录）

分类页面给出的可导出格式如下。**表中参数是官网导出格式能力的通用清单，`nms`、`conf`、`iou` 等出现在导出格式参数栏，并不代表这些配置对分类输出有 NMS 意义**。需要核对目标平台是否支持所选的具体 `-cls` 模型。

| 格式 | `format` | 官网模型名/目录 | 元数据 | 官网所列参数 |
| --- | --- | --- | --- | --- |
| PyTorch | `—` | `yolo26n-cls.pt` | ✅ | — |
| TorchScript | `torchscript` | `yolo26n-cls.torchscript` | ✅ | `imgsz`, `quantize`, `dynamic`, `nms`, `batch`, `device` |
| ONNX | `onnx` | `yolo26n-cls.onnx` | ✅ | `imgsz`, `quantize`, `dynamic`, `simplify`, `opset`, `nms`, `batch`, `data`, `fraction`, `device` |
| OpenVINO | `openvino` | `yolo26n-cls_openvino_model/` | ✅ | `imgsz`, `quantize`, `dynamic`, `nms`, `batch`, `data`, `fraction`, `device` |
| TensorRT | `engine` | `yolo26n-cls.engine` | ✅ | `imgsz`, `quantize`, `dynamic`, `simplify`, `opset`, `workspace`, `nms`, `batch`, `data`, `fraction`, `device` |
| CoreML | `coreml` | `yolo26n-cls.mlpackage` | ✅ | `imgsz`, `dynamic`, `quantize`, `nms`, `batch`, `device` |
| Apple Core AI | `coreai` | `yolo26n-cls.aimodel` | ✅ | `imgsz`, `batch`, `quantize` |
| TF SavedModel | `saved_model` | `yolo26n-cls_saved_model/` | ✅ | `imgsz`, `quantize`, `opset`, `nms`, `batch`, `data`, `fraction`, `device` |
| TF GraphDef | `pb` | `yolo26n-cls.pb` | ❌ | `imgsz`, `opset`, `batch`, `device` |
| TF Edge TPU | `edgetpu` | `yolo26n-cls_edgetpu.tflite` | ✅ | `imgsz`, `quantize`, `opset`, `data`, `fraction`, `device` |
| LiteRT | `litert` | `yolo26n-cls.tflite` | ✅ | `imgsz`, `quantize`, `batch`, `data`, `fraction`, `device` |
| PaddlePaddle | `paddle` | `yolo26n-cls_paddle_model/` | ✅ | `imgsz`, `batch`, `device` |
| MNN | `mnn` | `yolo26n-cls.mnn` | ✅ | `imgsz`, `batch`, `dynamic`, `quantize`, `simplify`, `opset`, `nms`, `device` |
| NCNN | `ncnn` | `yolo26n-cls_ncnn_model/` | ✅ | `imgsz`, `quantize`, `batch`, `device` |
| IMX500 | `imx` | `yolo11n_imx_model/` | ✅ | `imgsz`, `quantize`, `data`, `fraction`, `nms`, `device` |
| RKNN | `rknn` | `yolo26n-cls_rknn_model/` | ✅ | `imgsz`, `batch`, `name`, `quantize`, `simplify`, `opset`, `data`, `fraction`, `device` |
| ExecuTorch | `executorch` | `yolo26n-cls_executorch_model/` | ✅ | `imgsz`, `batch`, `device` |
| Axelera | `axelera` | `yolo26n-cls_axelera_model/` | ✅ | `imgsz`, `batch`, `quantize`, `data`, `fraction`, `device` |
| DEEPX | `deepx` | `yolo26n-cls_deepx_model/` | ✅ | `imgsz`, `quantize`, `simplify`, `opset`, `data`, `optimize`, `device` |
| Qualcomm QNN | `qnn` | `yolo26n-cls_qnn.onnx` | ✅ | `imgsz`, `batch`, `name`, `quantize`, `simplify`, `opset`, `data`, `fraction`, `device` |
| Hailo | `hailo` | `yolo26n-cls_hailo_model/` | ✅ | `imgsz`, `name`, `quantize`, `data`, `fraction`, `simplify`, `conf`, `iou`, `device` |
| Huawei Ascend | `ascend` | `yolo26n-cls_ascend_model/` | ✅ | `imgsz`, `batch`, `name`, `quantize`, `opset`, `simplify`, `nms`, `device` |
| AMD Xilinx | `xilinx` | `yolo26n-cls_xilinx_model/` | ✅ | `imgsz`, `name`, `quantize`, `data`, `fraction`, `opset`, `simplify`, `device` |
| NXP | `nxp` | `yolo26n-cls_nxp.tflite` | ✅ | `imgsz`, `quantize`, `data`, `fraction`, `device` |

> 官网分类导出表的 IMX500 行使用了 `yolo11n_imx_model/`，与该页其它 `yolo26n-cls...` 命名不一致；这里**按网页展示值记录而未自行修正**，实际导出文件名以命令输出为准。

### 13.1 PyTorch 分类模型导出 ONNX

```python
from ultralytics import YOLO

model = YOLO("runs/classify/exp1/weights/best.pt")
model.export(format="onnx", imgsz=224, dynamic=False)
```

```bash
yolo export model=runs/classify/exp1/weights/best.pt format=onnx imgsz=224
```

### 13.2 加载 ONNX 进行分类预测

```python
from ultralytics import YOLO

model = YOLO("runs/classify/exp1/weights/best.onnx", task="classify")
for r in model.predict(source="test.jpg", imgsz=224):
    print(r.names[r.probs.top1], float(r.probs.top1conf))
```

### 13.3 常见导出选项含义（按部署需求选择）

| 参数 | 主要含义 |
|---|---|
| `format` | `onnx`、`engine`、`openvino`、`coreml` 等目标格式。 |
| `imgsz` | 导出时输入尺寸；应与训练/部署预处理协调。 |
| `batch` | 导出时的批次支持/固定批大小，依格式有所不同。 |
| `dynamic` | 是否允许动态输入维度，依格式支持。 |
| `quantize` | 导出精度/量化配置；是否能用 INT8、需要何种校准数据取决于格式。 |
| `simplify` | 对导出图做简化（如 ONNX）。 |
| `opset` | ONNX 算子集版本。 |
| `device` | 执行导出使用的设备。 |
| `data` / `fraction` | 某些量化/校准路径需要的数据及抽样比例。 |
| `workspace` | 某些 TensorRT 路径的构建工作区配置。 |
| `nms` / `conf` / `iou` | 官网通用导出格式表有列出；**整图分类无检测框 NMS 流程**。 |

## 14. 与检测预测的区别总结

| 维度 | Detect | Classify |
|---|---|---|
| 模型名 | `yolo26n.pt` | `yolo26n-cls.pt` |
| 预测核心输出 | `result.boxes` | `result.probs` |
| 主要内容 | 多个框、类别与分数 | 整图一个最可能类别和整组类别概率 |
| 阈值策略 | `conf` 可过滤检测框 | 自行判断 `top1conf` 并进行拒识 |
| 后处理 | 可能有 NMS | 不使用检测框 NMS |
| 验证指标 | mAP50-95 等 | Top-1/Top-5 Accuracy |

## 15. 常见问题与检查项

- 结果是 `boxes` 而不是 `probs`：先检查是否加载了检测权重。
- 分类预测分辨率不一致：明确传入 `imgsz=224`（或实际训练尺寸）。
- 图像边缘重要特征被裁掉：分类默认 CenterCrop；需要与训练配套地定制预处理。
- 分类分数总是很高但错得多：检查数据划分、预训练类别映射及校准；不要把 Top-1 softmax 分数等同于真实正确率。
- 希望每图出现多个类别：默认 classify 是单标签分类，需重新定义任务或采用别的架构。
- 想导出概率明细：建议使用 `result.probs.data` 自行写 CSV/JSON，而不是照抄 Detect 的 `save_txt`。

## 16. 相关文档

- 分类任务：https://docs.ultralytics.com/zh/tasks/classify
- 通用预测：https://docs.ultralytics.com/zh/modes/predict
- 分类数据集：https://docs.ultralytics.com/zh/datasets/classify
- 通用训练：https://docs.ultralytics.com/zh/modes/train
- 分类训练参数：`Ultralytics分类训练参数参考.md`
