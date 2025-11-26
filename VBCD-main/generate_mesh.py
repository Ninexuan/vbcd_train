import torch
import numpy as np
import open3d as o3d
import os
import glob
from torch.utils.data import DataLoader
from models.crownmvm2 import CrownMVM
from dentaldataset import IOS_Datasetv2

# ================= 配置 =================
# 你的模型路径
CHECKPOINT_PATH = "./checkpoints/real_train_step_2000.pth"
DATA_ROOT = "./train_data"
OUTPUT_DIR = "./debug_visuals"
TARGET_TOOTH = "16" # 我们专门诊断 16 号牙
# ========================================

def save_colored_ply(points, filename, color):
    """保存带颜色的点云"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # 给点云上色 (R,G,B)
    pcd.paint_uniform_color(color)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"  -> 已保存: {filename}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载模型
    print("加载模型...")
    if not os.path.exists(CHECKPOINT_PATH):
        # 自动寻找
        pths = glob.glob("./checkpoints/*/*.pth")
        if pths:
            ckpt_path = max(pths, key=os.path.getctime)
            print(f"使用自动找到的模型: {ckpt_path}")
        else:
            print("错误: 找不到 .pth 文件")
            return
    else:
        ckpt_path = CHECKPOINT_PATH
        
    model = CrownMVM(in_channels=4, out_channels=4).to(device)
    state = torch.load(ckpt_path, map_location=device)
    
    # 权重键名修复
    new_state = {}
    for k, v in state.items():
        name = k.replace('module.', '')
        if 'basic_conv' in name: name = name.replace('basic_conv', 'basic_module')
        new_state[name] = v
        
    model.load_state_dict(new_state, strict=False)
    model.eval()

    # 2. 加载数据 (修复: 去掉不支持的 subdirs 参数)
    print("加载数据...")
    dataset = IOS_Datasetv2(DATA_ROOT, is_train=False) # <--- 修正了这里
    loader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)
    
    print(f"在测试集中寻找牙位 {TARGET_TOOTH} ...")
    
    target_batch = None
    
    # 循环寻找目标牙位
    for batch in loader:
        inputs, targets, pointcloud_inform, batch_y, min_bound_crop, file_dirs = batch
        current_path = file_dirs[0]
        
        # 检查路径里是否包含 '16'
        if f"/{TARGET_TOOTH}/" in current_path or f"\\{TARGET_TOOTH}\\" in current_path:
            print(f"找到目标数据: {current_path}")
            target_batch = batch
            break
    
    if target_batch is None:
        print(f"未找到 {TARGET_TOOTH} 的测试数据，将使用第一个可用数据。")
        target_batch = next(iter(loader))

    # 开始诊断
    with torch.no_grad():
        inputs, targets, pointcloud_inform, batch_y, min_bound_crop, file_dirs = target_batch
        inputs = inputs.to(device)
        
        # 构造 Prompt (16号牙对应的 ID 是 5)
        # 这里简单写死为 5，或者你可以复用之前的映射逻辑
        prompt = torch.tensor([5], dtype=torch.long).to(device) 
        
        # === 3. 模型推理 ===
        outputs = model(inputs, prompt)
        if len(outputs) == 4:
            voxel_ind, voxel_normal, refined_pos_with_normal, batch_x = outputs
        else:
            voxel_ind, refined_pos_with_normal, batch_x = outputs

        # === 4. 分析粗糙体素 (Coarse) ===
        # 提取概率 > 0.5 的格子的中心点
        prob = torch.sigmoid(voxel_ind)
        mask = (prob > 0.5).squeeze()
        indices = torch.nonzero(mask) # (N, 3)
        
        # 获取精修点 (Refined)
        refined_points = refined_pos_with_normal[:, :3]
        
        print(f"\n【数值诊断】")
        print(f"1. 粗糙点数量 (Voxel): {indices.shape[0]}")
        print(f"2. 精修点数量 (Refined): {refined_points.shape[0]}")
        
        # 计算 Offset (仅当数量一致时)
        if indices.shape[0] == refined_points.shape[0]:
            voxel_size = 0.3125
            origin = min_bound_crop[0].cpu().numpy()
            
            # 还原粗糙坐标
            coarse_points_np = origin + indices.cpu().numpy() * voxel_size
            refined_points_np = refined_points.cpu().numpy()
            
            # 计算偏移
            offsets = refined_points_np - coarse_points_np
            avg_offset = np.mean(np.abs(offsets))
            
            print(f"3. 偏移量统计:")
            print(f"   最大偏移: {np.max(np.abs(offsets)):.4f} mm")
            print(f"   平均偏移: {avg_offset:.4f} mm")
            
            if avg_offset > 2.0:
                print("   [!!! 警报 !!!] 平均偏移量过大！点被炸飞了！")
            
            # === 5. 保存对比文件 ===
            save_colored_ply(coarse_points_np, os.path.join(OUTPUT_DIR, "1_Coarse_Green.ply"), [0, 1, 0]) # 绿色
            save_colored_ply(refined_points_np, os.path.join(OUTPUT_DIR, "2_Refined_Red.ply"), [1, 0, 0]) # 红色
            
            print(f"\n诊断完成！请在 MeshLab 中打开 {OUTPUT_DIR} 文件夹下的两个 PLY 文件。")
            print("绿色 = 粗糙体素中心 (应该像马赛克牙齿)")
            print("红色 = 精修后点云 (如果它是乱的，说明精修层炸了)")
        else:
            print("粗糙点和精修点数量不一致，无法直接计算 Offset，请检查模型输出逻辑。")

if __name__ == "__main__":
    main()