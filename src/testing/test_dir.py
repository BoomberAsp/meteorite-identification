import os


current_path = os.path.dirname(os.path.abspath(__file__))
# 获取当前脚本所在的项目根目录
root_path = os.path.dirname(current_path)

TEST_DATA_DIR = os.path.join(root_path, "data/test_images")
print(TEST_DATA_DIR)

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    # 指定使用系统分配给你的 nvidia2 (在 torch 里通常是设备 0)
    print(f"Current device name: {torch.cuda.get_device_name(0)}")
    
    # 尝试做一个简单的张量计算，测试计算节点是否真的通了
    x = torch.tensor([1.0, 2.0]).cuda()
    y = x * 2
    print(f"Calculation result on GPU: {y}")

import torch
import ctypes

print(f"PyTorch Version: {torch.__version__}")
print(f"PyTorch built with CUDA: {torch.version.cuda}")

# 检查 libcuda 是否能被 12.8 的环境加载
try:
    ctypes.CDLL("libcuda.so.1")
    print("✅ Driver Library (libcuda.so.1) Loaded Successfully")
except Exception as e:
    print(f"❌ Driver Library Load Failed: {e}")

print(f"Is CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    # 打印具体的错误原因（如果有）
    try:
        print(f"CUDA Error Code: {torch.cuda.get_device_properties(0)}")
    except Exception as e:
        print(f"Detailed Error: {e}")