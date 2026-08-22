"""
工具函数模块
"""
import sys
import torch


def check_environment():
    """检查 Python 和 PyTorch 版本"""
    print("=" * 40)
    print("【环境检查】")
    print("=" * 40)
    print(f"Python 版本: {sys.version.split()[0]}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    else:
        print("运行模式: CPU")
    print("=" * 40)


def get_device():
    """获取当前可用设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")