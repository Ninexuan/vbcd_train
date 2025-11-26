import open3d as o3d
import numpy as np
import h5py
import os

# === 1. 曲率计算函数 (基于 PCA 协方差分析) ===
def compute_curvature(pcd, radius=2.0):
    """
    计算点云的表面变化率 (Surface Variation) 作为曲率的近似值。
    用于 VBCD 项目的 CMPL Loss 权重。
    """
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    num_points = len(points)
    curvatures = []

    print(f"    正在计算 {num_points} 个点的曲率 (搜索半径={radius}mm)...")
    
    for i in range(num_points):
        # 找邻居
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius)
        
        # 孤立点处理
        if k < 4:
            curvatures.append(0.0)
            continue
        
        # PCA 分析
        neighbors = points[idx, :]
        cov = np.cov(neighbors.T)
        eigenvalues, _ = np.linalg.eigh(cov) # 特征值分解
        
        # 计算表面变化率: λ1 / (λ1+λ2+λ3)
        # 最小特征值占比越大，说明越弯曲
        surface_variation = eigenvalues[0] / (np.sum(eigenvalues) + 1e-6)
        curvatures.append(surface_variation)

    return np.array(curvatures)

# === 2. 核心转换流程 ===
def stl_to_h5_pipeline(input_meshes, output_path, num_samples, is_gt=False):
    """
    input_meshes: 一个列表，包含要合并的 open3d mesh 对象
    output_path: 保存的 .h5 路径
    is_gt: 是否是 Ground Truth (如果是，需要对曲率做数值放大，方便 Loss 计算)
    """
    print(f"[-] 开始处理目标: {os.path.basename(output_path)}")

    # 2.1 合并网格 (Merge)
    # Open3D 中，网格相加就是合并
    combined_mesh = input_meshes[0]
    for i in range(1, len(input_meshes)):
        combined_mesh += input_meshes[i]
    
    # 2.2 采样 (Sampling)
    # 将面片变成点云，使用泊松盘采样保证均匀
    print(f"    正在合并并采样为 {num_samples} 个点...")
    pcd = combined_mesh.sample_points_poisson_disk(number_of_points=num_samples)
    
    # 2.3 计算法向量 (Normals)
    # 如果没法向就算，有法向就统一方向
    if not pcd.has_normals():
        pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(k=10)
    
    # 2.4 计算曲率 (Curvatures)
    # 搜索半径设为 1.0mm ~ 2.0mm 比较适合牙齿特征
    curv_raw = compute_curvature(pcd, radius=1.5)
    
    if is_gt:
        # GT 数据需要把曲率放大，作为 Loss 的权重
        # 经验值：放大 30 倍，并截断到 0-10 之间，防止 Loss 爆炸
        curvatures = np.clip(curv_raw * 30.0, 0, 10.0)
    else:
        # 输入数据不需要曲率，填 0 即可
        curvatures = np.zeros_like(curv_raw)

    # 2.5 数据整形与保存
    vertices = np.asarray(pcd.points).astype(np.float32)
    normals = np.asarray(pcd.normals).astype(np.float32)
    curvatures = curvatures.reshape(-1, 1).astype(np.float32)

    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # 键名必须严格对应 dentaldataset.py
        f.create_dataset('vertices', data=vertices)
        f.create_dataset('normals', data=normals)
        f.create_dataset('curvatures', data=curvatures)
    
    print(f"[√] 保存成功: {output_path}")
    print(f"    数据形状: V{vertices.shape} N{normals.shape} C{curvatures.shape}")


# === 主程序 ===
if __name__ == "__main__":
    # 配置路径 (请根据你的实际文件名修改)
    raw_dir = "raw_data"  # 假设你把stl放在这个文件夹下
    
    # 文件名 (根据你的截图)
    file_jaw = os.path.join(raw_dir, "down_tooth.stl")
    file_ant = os.path.join(raw_dir, "up_tooth.stl")
    file_crown = os.path.join(raw_dir, "crown.stl")

    # 输出路径 (符合 VBCD 训练目录结构: 牙位/train/病例名)
    # 你的牙是 16 号
    output_case_dir = "train_data/16/train/case_001"
    
    # 1. 检查文件是否存在
    if not all(os.path.exists(p) for p in [file_jaw, file_ant, file_crown]):
        print("错误：请检查 raw_data 文件夹下的文件名是否完全一致！")
        exit()

    # 2. 读取 STL
    print(">>>正在读取 STL 文件...")
    mesh_jaw = o3d.io.read_triangle_mesh(file_jaw)
    mesh_ant = o3d.io.read_triangle_mesh(file_ant)
    mesh_crown = o3d.io.read_triangle_mesh(file_crown)

    # 3. 制作输入数据 (pna_crop.h5)
    # 逻辑：模型底板 + 对颌牙 = 完整的输入环境
    stl_to_h5_pipeline(
        input_meshes=[mesh_jaw, mesh_ant], 
        output_path=os.path.join(output_case_dir, "pna_crop.h5"), 
        num_samples=30000, # 输入点云通常密集一些
        is_gt=False
    )

    # 4. 制作标准答案 (crown.h5)
    # 逻辑：单独的修复体
    stl_to_h5_pipeline(
        input_meshes=[mesh_crown], 
        output_path=os.path.join(output_case_dir, "crown.h5"), 
        num_samples=10000, # GT 点云，1万点足够
        is_gt=True         # 开启曲率增强
    )

    print("\n[完成] 所有数据转换完毕！可以直接用于 VBCD 训练了。")