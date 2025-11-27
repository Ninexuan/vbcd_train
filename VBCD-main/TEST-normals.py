import torch
import numpy as np
import open3d as o3d
import os
import glob
from torch.utils.data import DataLoader
from models.crownmvm2 import CrownMVM
from dentaldataset import IOS_Datasetv2

# ================= 配置 =================
# 必须指向你那个“跑了2000步”的权重文件
CHECKPOINT_PATH = "./checkpoints/real_train_step_2000.pth" 
DATA_ROOT = "./train_data"
OUTPUT_DIR = "./debug_normals_check"
# ========================================

def main():
    if not os.path.exists(CHECKPOINT_PATH):
        # 尝试自动寻找 step_2000 或 best
        pths = glob.glob("./checkpoints/*/*step_2000.pth") + glob.glob("./checkpoints/*/*best.pth")
        if pths:
            ckpt_path = pths[0]
            print(f"自动定位模型: {ckpt_path}")
        else:
            print("找不到模型文件！")
            return
    else:
        ckpt_path = CHECKPOINT_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载模型
    print("加载模型中...")
    model = CrownMVM(in_channels=4, out_channels=4).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    new_state = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '').replace('basic_conv', 'basic_module')
        new_state[name] = v
    model.load_state_dict(new_state, strict=False)
    model.eval()

    # 2. 加载数据 (随便找一个测试样本)
    dataset = IOS_Datasetv2(DATA_ROOT, is_train=False)
    loader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)
    
    print("正在导出精修点云及法向量...")

    with torch.no_grad():
        # 只看前 3 个样本
        for i, batch in enumerate(loader):
            if i >= 3: break
            
            inputs, targets, _, _, _, file_dirs = batch
            inputs = inputs.to(device)
            
            # 解析名字
            path_parts = os.path.normpath(file_dirs[0]).split(os.sep)
            case_name = f"{path_parts[-4]}_{path_parts[-2]}" # e.g. 16_case_7
            
            # 伪造 prompt (或者用你的 mapping)
            prompt = torch.zeros(1, dtype=torch.long).to(device)

            # 推理
            outputs = model(inputs, prompt)
            if len(outputs) == 4:
                _, _, refined_pos_with_normal, _ = outputs
            else:
                _, refined_pos_with_normal, _ = outputs
            
            # === 关键步骤：提取数据 ===
            data = refined_pos_with_normal.cpu().numpy()
            points = data[:, :3]
            normals = data[:, 3:6]
            
            # 检查法向量数值是否正常
            norm_lengths = np.linalg.norm(normals, axis=1)
            print(f"\n[{case_name}]")
            print(f"  点数: {len(points)}")
            print(f"  法向量长度均值: {np.mean(norm_lengths):.4f} (正常应接近1.0)")
            print(f"  法向量是否存在全0情况: {(norm_lengths == 0).any()}")

            # === 保存为带法向的 PLY ===
            # 这样你才能在 MeshLab 里看到“刺猬”
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            # 统一把法向归一化，排除长度干扰，只看方向
            pcd.normalize_normals()
            
            save_path = os.path.join(OUTPUT_DIR, f"Debug_{case_name}_refined.ply")
            o3d.io.write_point_cloud(save_path, pcd)
            print(f"  已保存诊断文件: {save_path}")
            print("  -> 请务必用 MeshLab 打开，开启 'Render -> Show Normal' 查看！")

if __name__ == "__main__":
    main()