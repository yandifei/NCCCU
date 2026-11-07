import torch
print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
print("GPU 设备数量:", torch.cuda.device_count())
print("当前 GPU 名称:", torch.cuda.get_device_name(0))
