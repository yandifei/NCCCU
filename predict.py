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

    # 以下逐项列出参考文档的参数，数值为文档默认值。
    # 上方实参控制当前运行；下方注释用于查阅，标为未接入的项目需要实现后才能启用。
    # 原生 API 各自接收不同参数，不能把 plot/export 的参数直接传给训练或预测函数。

    # model.predict() 参数（27 项）
    # source=None,  # 当前使用 data 和 split 读取官方 CSV；指定推理的数据源
    # conf=0.25,  # 原生任务范围不含分类，当前未接入；设置检测的最低置信度阈值
    # iou=0.7,  # 原生任务范围不含分类，当前未接入；交并比 (IoU) 阈值，用于非极大值抑制 (NMS)
    # imgsz=640,  # 当前从训练权重读取输入尺寸；Letterbox 目标尺寸
    # rect=True,  # 当前使用与训练相同的缩放填充处理；如果为 True，则在可能时使用最小矩形填充（相同形状的批次和受支持的后端）
    # quantize=None,  # 当前使用 FP32 推理；推理精度：16/"fp16" 与 32/"fp32"/未设置为 PyTorch 和 TorchScript 模型选择 FP16 或 FP32 计算；其他格式则在其构件与运行时所选的精度下进行计算
    # device=None,  # 部分接入，本项目使用单设备编号或 cpu；指定推理设备（例如 cpu、cuda:0、0、npu 或 npu:0）
    # dnn=False,  # 当前未接入；如果为 True，则在 ONNX 模型推理中使用 OpenCV DNN 模块，而不是 ONNX Runtime
    # data=None,  # 已接入，上方使用本项目配置；数据集 YAML 的路径（例如 coco8.yaml），仅读取其中的 names，且仅当加载的模型不包含自身的类别名称时使用：例如第三方导出模型，或与其附带元数据分离的 Ultralytics 导出模型
    # batch=1,  # 已接入，上方使用本项目配置；指定推理的批次大小（仅当源是目录、视频文件或 .txt 文件时有效）
    # tile=False,  # 当前未接入；固定大小切片推理并把结果映射回原图坐标；用于高分辨率目标任务
    # max_det=300,  # 原生任务范围不含分类，当前未接入；每张图像允许的最大检测数量
    # vid_stride=1,  # 当前未接入；视频输入的帧步长
    # stream_buffer=False,  # 当前未接入；决定是否为视频流排队接收到的帧
    # visualize=False,  # 当前未接入；在每个预测结果旁保存类别激活热力图，显示哪些像素提高了预测类别的分数
    # augment=False,  # 当前未接入；为预测启用测试时增强 (TTA)，可能提升检测鲁棒性，但会降低推理速度
    # agnostic_nms=False,  # 原生任务范围不含分类，当前未接入；启用类无关的 NMS，压制不同类别中得分较低的重叠框，而不仅仅是同一类别内
    # classes=None,  # 本赛输出全部 10 类；将预测结果过滤为一组类别 ID
    # retina_masks=False,  # 原生任务范围不含分类，当前未接入；返回高分辨率分割掩码
    # embed=None,  # 当前未接入；指定用于提取特征向量或嵌入的层
    # project=None,  # 当前固定 outputs/predict；项目目录的名称；启用 save 后，预测输出将保存到该目录
    # name=None,  # 已接入，上方使用本项目配置；预测运行的名称
    # stream=False,  # 当前按批次计算并汇总检查级概率；通过返回 Results 对象生成器，而不是一次性将所有帧加载到内存中，为长视频或大量图像启用节省内存的处理方式
    # verbose=True,  # 当前固定输出完成摘要；控制是否在终端中显示详细的推理日志，为预测过程提供实时反馈
    # compile=False,  # 当前未接入；使用 backend='inductor' 启用 PyTorch 2.x 的 torch.compile 图编译
    # channels_last=None,  # 当前未接入；对原生 PyTorch 推理使用 channels_last (NHWC) 内存格式
    # nms=None,  # 原生任务范围不含分类，当前未接入；默认运行带 NMS 的一对多推理（None 或 True）

    # 预测可视化 参数（10 项）
    # show=False,  # 当前未接入；如果为 True，则会在窗口中显示带标注的图像或视频
    # save=False,  # 文档 Python 默认 False、CLI 默认 True；当前固定保存赛事 CSV，图片可视化未接入；启用后将带注释的图像或视频保存到文件
    # save_frames=False,  # 当前未接入；处理视频时，将单独的帧保存为图像
    # save_txt=False,  # 当前未接入；将检测结果按照 [class] [x_center] [y_center] [width] [height] [confidence] 格式保存到文本文件中
    # save_conf=False,  # 当前未接入；在保存的文本文件中包含置信度分数
    # save_crop=False,  # 原生任务范围不含分类，当前未接入；保存检测对象的裁剪图像
    # show_labels=True,  # 当前未接入；在可视化输出中显示每个检测结果的标签，帮助你立即了解检测到的对象
    # show_conf=True,  # 当前未接入；在每个检测结果旁显示其置信度分数，让你了解模型对每个检测结果的确定程度
    # show_boxes=True,  # 原生任务范围不含分类，当前未接入；在检测到的对象周围绘制边界框
    # line_width=None,  # 原生任务范围不含分类，当前未接入；指定边界框的线宽

    # Results.plot() 参数（17 项）
    # conf=True,  # 绘制接口，当前未接入；包含检测置信度分数
    # line_width=None,  # 绘制接口，当前未接入；边界框的线宽
    # font_size=None,  # 绘制接口，当前未接入；文本字体大小
    # font='Arial.ttf',  # 绘制接口，当前未接入；文本标注的字体名称
    # pil=False,  # 绘制接口，当前未接入；将图像作为 PIL Image 对象返回
    # img=None,  # 绘制接口，当前未接入；备用图像
    # kpt_radius=5,  # 绘制接口，当前未接入；绘制关键点的半径
    # kpt_line=True,  # 绘制接口，当前未接入；使用线条连接关键点
    # labels=True,  # 绘制接口，当前未接入；在标注中包含类别标签
    # boxes=True,  # 绘制接口，当前未接入；在图像上叠加边界框
    # masks=True,  # 绘制接口，当前未接入；在图像上叠加掩码
    # probs=True,  # 绘制接口，当前未接入；包含分类概率
    # show=False,  # 绘制接口，当前未接入；使用默认图像查看器直接显示标注后的图像
    # save=False,  # 绘制接口，当前未接入；将标注后的图像保存到 filename 指定的文件
    # filename=None,  # 绘制接口，当前未接入；当 save 为 True 时，用于保存标注后图像的文件路径和名称
    # color_mode='class',  # 绘制接口，当前未接入；指定颜色模式，例如“instance”或“class”
    # txt_color=(255, 255, 255),  # 绘制接口，当前未接入；边界框和图像分类标签的 BGR 文本颜色

    # model.export() 导出参数（文档未统一给出默认值，以下为示例值）
    # 当前多标签模型导出尚未接入，导出时必须保留独立 sigmoid 输出。
    # format='onnx',  # 导出接口，当前未接入；目标格式，如 onnx、engine
    # imgsz=224,  # 导出接口，当前未接入；导出尺寸，应匹配训练预处理
    # batch=1,  # 导出接口，当前未接入；导出批量大小
    # dynamic=False,  # 导出接口，当前未接入；动态输入维度
    # quantize=None,  # 导出接口，当前未接入；量化精度，按目标格式选择
    # simplify=True,  # 导出接口，当前未接入；简化导出图
    # opset=None,  # 导出接口，当前未接入；ONNX 算子集版本
    # device=0,  # 导出接口，当前未接入；导出设备
    # data=None,  # 导出接口，当前未接入；量化校准数据配置
    # fraction=1.0,  # 导出接口，当前未接入；校准数据比例，测试集不得用于校准
    # workspace=None,  # 导出接口，当前未接入；TensorRT 工作区大小
    # nms=None,  # 导出接口，当前未接入；检测 NMS，当前多标签分类无检测框
    # conf=0.25,  # 导出接口，当前未接入；检测置信度阈值
    # iou=0.7,  # 导出接口，当前未接入；检测 IoU 阈值
    # name=None,  # 导出接口，当前未接入；部分导出格式的产物名称
    # optimize=False,  # 导出接口，当前未接入；部分后端的导出优化

    # 其他文档示例接口（示例值）
    # task="classify",  # YOLO 构造函数的任务选择，当前从项目检查点恢复模型
    # split="val",  # model.val 验证划分；当前预测 split 实参使用官方 valid 命名
