import sys
import torch
import numpy as np
import h5py
import open3d as o3d
import pytorch3d
from accelerate import Accelerator

def check_environment():
    print("="*30)
    print("正在检查环境依赖...")
    print("="*30)

    # 1. 检查 Python 版本
    print(f"[Python]: {sys.version.split()[0]}")

    # 2. 检查 PyTorch 和 CUDA
    print(f"[PyTorch]: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"[CUDA]: 可用 (版本: {torch.version.cuda})")
        print(f"[GPU]: {torch.cuda.get_device_name(0)}")
        # 测试简单的 Tensor 运算
        try:
            x = torch.tensor([1.0]).cuda()
            print("[Tensor Test]: GPU 张量创建成功")
        except Exception as e:
            print(f"[Tensor Test]: 失败 - {e}")
    else:
        print("[CUDA]: 不可用 (请检查显卡驱动或 PyTorch 安装版本)")

    # 3. 检查其他关键库
    try:
        print(f"[NumPy]: {np.__version__}")
    except ImportError:
        print("[NumPy]: 未安装")

    try:
        print(f"[h5py]: {h5py.__version__}")
    except ImportError:
        print("[h5py]: 未安装")

    try:
        print(f"[Open3D]: {o3d.__version__}")
    except ImportError:
        print("[Open3D]: 未安装")

    try:
        print(f"[PyTorch3D]: {pytorch3d.__version__}")
    except ImportError:
        print("[PyTorch3D]: 未安装 (这是常见问题点，请重点关注)")

    try:
        acc = Accelerator()
        print(f"[Accelerate]: {acc.__class__.__name__} 初始化成功")
    except Exception as e:
        print(f"[Accelerate]: 初始化失败 - {e}")

    print("="*30)
    print("环境检查完成。如果没有报错且 CUDA 可用，则环境安装正确。")
    print("="*30)

if __name__ == "__main__":
    check_environment()