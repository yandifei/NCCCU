from ncccu.inference import predict_multilabel

# 关键点：保留入口保护，避免 Windows 创建进程时递归执行
if __name__ == '__main__':
    predict_multilabel(
        model="./outputs/train/ncccu_cls/weights/best.pt",  # 训练得到的最佳权重
        split="test",  # 官方 CSV 划分，修改时同步修改 source
        workers=3,  # 数据加载进程数
        gate=0.5,  # 多标签概率阈值
        deterministic=True,  # 严格确定性算法
        seed=0,  # 固定随机种子
        save_dir=None,  # 默认使用 project/name，也可指定 outputs 内的完整目录
        smoke_test=False,  # 冒烟权重仅用于小批流程验证
        # 预测参数
        source='./datasets/test.csv',  # 显式指定 data/split 对应的官方 CSV
        conf=None,  # 固定约束，设置其他值会报错；设置检测的最低置信度阈值
        iou=None,  # 固定约束，设置其他值会报错；交并比 (IoU) 阈值，用于非极大值抑制 (NMS)
        imgsz=224,  # 必须与训练检查点的输入尺寸相同
        rect=False,  # 固定约束，设置其他值会报错；如果为 True，则在可能时使用最小矩形填充（相同形状的批次和受支持的后端）
        quantize=32,  # 32 使用 FP32；16 在 CUDA 上使用 FP16
        device=0,  # 指定推理设备（例如 cpu、cuda:0、0、npu 或 npu:0）
        dnn=False,  # 固定约束，设置其他值会报错；如果为 True，则在 ONNX 模型推理中使用 OpenCV DNN 模块，而不是 ONNX Runtime
        data='./datasets',  # 本项目读取官方数据根目录
        batch=16,  # 指定推理的批次大小（仅当源是目录、视频文件或 .txt 文件时有效）
        tile=False,  # 固定约束，设置其他值会报错；固定大小切片推理并把结果映射回原图坐标；用于高分辨率目标任务
        max_det=0,  # 固定约束，设置其他值会报错；每张图像允许的最大检测数量
        vid_stride=1,  # 固定约束，设置其他值会报错；视频输入的帧步长
        stream_buffer=False,  # 固定约束，设置其他值会报错；决定是否为视频流排队接收到的帧
        visualize=False,  # 固定约束，设置其他值会报错；在每个预测结果旁保存类别激活热力图，显示哪些像素提高了预测类别的分数
        augment=False,  # True 使用原图与水平翻转图的 sigmoid 概率均值
        agnostic_nms=False,  # 固定约束，设置其他值会报错；启用类无关的 NMS，压制不同类别中得分较低的重叠框，而不仅仅是同一类别内
        classes=None,  # 固定约束，设置其他值会报错；将预测结果过滤为一组类别 ID
        retina_masks=False,  # 固定约束，设置其他值会报错；返回高分辨率分割掩码
        embed=None,  # 固定约束，设置其他值会报错；指定用于提取特征向量或嵌入的层
        project='outputs/predict',  # 项目目录的名称；启用 save 后，预测输出将保存到该目录
        name='ncccu_cls_test',  # 预测运行的名称
        stream=False,  # 固定约束，设置其他值会报错；通过返回 Results 对象生成器，而不是一次性将所有帧加载到内存中，为长视频或大量图像启用节省内存的处理方式
        verbose=True,  # 控制是否在终端中显示详细的推理日志，为预测过程提供实时反馈
        compile=False,  # 固定约束，设置其他值会报错；使用 backend='inductor' 启用 PyTorch 2.x 的 torch.compile 图编译
        channels_last=False,  # 对原生 PyTorch 推理使用 channels_last (NHWC) 内存格式
        nms=False,  # 固定约束，设置其他值会报错；默认运行带 NMS 的一对多推理（None 或 True）
        # 图片和文本结果保存
        show=False,  # 如果为 True，则会在窗口中显示带标注的图像或视频
        save=True,  # 保存带全部 10 类概率的图片，赛事 CSV 始终输出
        save_frames=False,  # 固定约束，设置其他值会报错；处理视频时，将单独的帧保存为图像
        save_txt=True,  # 额外保存逐图达到 gate 阈值的官方类别 ID
        save_conf=True,  # 逐图 TXT 同时写入各类别概率
        save_crop=False,  # 固定约束，设置其他值会报错；保存检测对象的裁剪图像
        show_labels=True,  # 在可视化输出中显示每个检测结果的标签，帮助你立即了解检测到的对象
        show_conf=True,  # 在每个检测结果旁显示其置信度分数，让你了解模型对每个检测结果的确定程度
        show_boxes=False,  # 固定约束，设置其他值会报错；在检测到的对象周围绘制边界框
        line_width=None,  # 固定约束，设置其他值会报错；指定边界框的线宽
        plot_options={  # 实际绘图参数，单独传入以区分同名参数
            "conf": True,  # 显示概率值
            "line_width": None,  # 分类图无边界框
            "font_size": 16,  # 文字字号
            "font": 'default',  # 文字字体
            "pil": True,  # 使用 PIL 绘制
            "img": None,  # 使用当前原图
            "kpt_radius": 0,  # 禁用关键点
            "kpt_line": False,  # 禁用骨架连线
            "labels": True,  # 显示类别名称
            "boxes": False,  # 禁用检测框
            "masks": False,  # 禁用分割掩码
            "probs": True,  # 显示全部类别概率
            "show": False,  # 预览窗口还需要外层 show=True
            "save": True,  # 保存标注图片
            "filename": None,  # 按影像 ID 自动生成 outputs 内的文件名
            "color_mode": 'class',  # 类别文本模式
            "txt_color": (255, 255, 255),  # 文字 BGR 颜色
        },
        export_options={  # 实际导出，当前支持固定输入的 FP32 TorchScript
            "enabled": True,  # 执行导出
            "format": 'torchscript',  # 当前支持 torchscript
            "imgsz": 224,  # 匹配训练尺寸
            "batch": 1,  # 导出批量
            "dynamic": False,  # 固定形状
            "quantize": 32,  # 当前支持 FP32
            "simplify": False,  # TorchScript 无 ONNX 简化
            "opset": None,  # TorchScript 无 ONNX 算子集
            "device": 'cuda:0',  # 与当前预测设备一致
            "data": None,  # 无需校准数据
            "fraction": 1.0,  # 保持完整数据约定
            "workspace": None,  # TorchScript 无 TensorRT 工作区
            "nms": False,  # 多标签无检测框 NMS
            "conf": None,  # 无检测置信度阈值
            "iou": None,  # 无检测框 IoU
            "name": 'model',  # outputs/predict/name 下的导出文件名
            "optimize": False,  # 关闭后端专项优化
        },
        task="classify",  # 分类模型任务
    )
