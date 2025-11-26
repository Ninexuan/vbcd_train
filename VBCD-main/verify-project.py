import torch
import numpy as np
import sys
import os
import gc

# 1. 尝试导入项目模块
try:
    from models.crownmvm2 import CrownMVM
    from models.loss import curvature_and_margine_penalty_loss
    from dentaldataset import create_voxelwithnormal_grid 
    print("[√] 模块导入成功！")
except ImportError as e:
    print(f"[X] 模块导入失败: {e}")
    sys.exit(1)

def cleanup_memory():
    """安全地清理显存"""
    gc.collect()
    torch.cuda.empty_cache()

def test_voxel_function():
    print("\n--- 测试 1: 验证数据生成 (AI补充代码) ---")
    # ...保持原有的测试逻辑不变...
    N_points = 1000
    coords = np.random.uniform(40, 60, (N_points, 3)) 
    normals = np.random.uniform(-1, 1, (N_points, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    curvatures = np.random.uniform(0, 1, (N_points, 1))
    point_cloud = np.hstack((coords, normals, curvatures))
    
    center = np.array([50.0, 50.0, 50.0])
    crop_size = np.array([2.0, 2.0, 2.0])
    voxel_size = np.array([0.15625, 0.15625, 0.15625])
    half_crop = (crop_size * 10) / 2 
    min_bound = center - half_crop
    max_bound = center + half_crop
    
    try:
        grid = create_voxelwithnormal_grid(point_cloud, min_bound, max_bound, voxel_size)
        if grid.shape == (4, 128, 128, 128):
            print(f"[√] 生成网格成功: {grid.shape}")
            return torch.from_numpy(grid).float().unsqueeze(0)
        else:
            print(f"[!] 尺寸不对: {grid.shape}")
            return torch.from_numpy(grid).float().unsqueeze(0)
    except Exception as e:
        print(f"[X] 数据生成崩溃: {e}")
        sys.exit(1)

def run_model_at_resolution(model, input_tensor, prompt_idx, resolution_name):
    """封装单次模型运行逻辑"""
    print(f"\n>>> 正在尝试 {resolution_name} 分辨率...")
    try:
        # 使用混合精度 (fp16)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            voxel_ind, refined_pos_with_normal, batch_x = model(input_tensor, prompt_idx)
        
        print(f"[√] 成功！{resolution_name} 分辨率跑通了！")
        print(f"   输出 Voxel 形状: {voxel_ind.shape}")
        print(f"   输出 Points 形状: {refined_pos_with_normal.shape}")
        return True
    except torch.OutOfMemoryError:
        print(f"[X] 失败: {resolution_name} 显存不足 (OOM)。")
        return False
    except Exception as e:
        print(f"[X] 失败: {resolution_name} 发生未知错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_resilient(input_voxel):
    print("\n--- 测试 2: 验证模型前向传播 (各种分辨率) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    cleanup_memory()
    
    # 初始化模型
    # 注意：虽然我们可能输入64，但模型参数 scale=128 暂时不用改，
    # 因为 U-Net 是全卷积网络，能适应不同尺寸输入。
    model = CrownMVM(scale=128, in_channels=1, out_channels=4).to(device)
    prompt_idx = torch.tensor([0]).to(device)

    # === 尝试 1: 128^3 (大概率失败，但为了验证流程) ===
    input_128 = input_voxel[:, :1, ...].to(device)
    success = run_model_at_resolution(model, input_128, prompt_idx, "128^3")
    
    # 清理变量和显存
    del input_128
    cleanup_memory()
    
    if success:
        print("\n恭喜！你的显卡竟然跑通了 128^3！")
        return

    # === 尝试 2: 64^3 (必须成功，否则无法训练) ===
    print("\n[!] 自动降级：准备尝试 64^3 分辨率...")
    
    # 1. 手动把 128 的数据缩小到 64
    input_64_cpu = torch.nn.functional.interpolate(input_voxel[:, :1, ...], size=(64,64,64))
    input_64 = input_64_cpu.to(device)
    
    success_64 = run_model_at_resolution(model, input_64, prompt_idx, "64^3")
    
    del input_64
    cleanup_memory()
    
    if success_64:
        print("\n[总结] 环境验证通过！")
        print("虽然 128^3 跑不动，但 64^3 可以运行。")
        print("接下来的策略：我们将修改训练配置，在 64x64x64 分辨率下训练模型。")
    else:
        print("\n[崩溃] 64^3 也跑不动，或者有其他代码错误。请检查上方报错信息。")

if __name__ == "__main__":
    # 1. 生成数据
    voxel_tensor = test_voxel_function()
    
    # 2. 弹性测试模型
    if voxel_tensor is not None:
        test_model_resilient(voxel_tensor)