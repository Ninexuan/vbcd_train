import os
import glob
import subprocess
import re
import numpy as np
import open3d as o3d
import h5py
import shutil
from scipy.spatial import cKDTree

# ========================== 核心配置区 ==========================
# 1. 原始数据集根目录 (已更新为英文路径)
RAW_DATA_ROOT = r"D:\vbcd\Dataset"

# 2. 导师工具的绝对路径 (请确认此路径是否正确，根据你之前的截图配置)
TOOL_EXE_PATH = r"D:\VBCD_TOOLS\MODELTOOLS\GenerateByVertex.exe"

# 3. 输出 H5 文件的保存路径
OUTPUT_ROOT = r"train_data"

# 4. 采样点数设置
NUM_POINTS_INPUT = 30000  # 输入数据
NUM_POINTS_GT = 10000     # 修复体
# ===============================================================

def read_mesh_safe(filepath):
    """
    【安全读取】解决 Open3D 无法读取含中文文件名的问题。
    原理：把文件拷贝到当前运行目录下的临时英文文件，读完再删除。
    """
    temp_name = "temp_read_buffer.stl"
    # 使用绝对路径
    temp_path = os.path.abspath(temp_name)
    
    try:
        # shutil 支持中文路径复制
        shutil.copyfile(filepath, temp_path)
        # 读取纯英文路径的临时文件
        mesh = o3d.io.read_triangle_mesh(temp_path)
        
        if not mesh.has_vertices():
            print(f"      [Error] 读取为空，可能文件损坏: {os.path.basename(filepath)}")
            return None
        return mesh
        
    except Exception as e:
        print(f"      [Error] 文件读取失败: {e}")
        return None
        
    finally:
        # 清理垃圾
        if os.path.exists(temp_path):
            os.remove(temp_path)

def parse_vec_file(vec_path):
    """
    解析 .vec 文件 (格式: X Y Z NX NY NZ C，带表头)
    """
    try:
        # skiprows=1 跳过表头
        data = np.loadtxt(vec_path, skiprows=1)
        
        if data.ndim == 1 or data.shape[1] != 7:
            print(f"      [Error] .vec 格式异常: {data.shape}")
            return None, None

        # 提取原始顶点 (前3列) 和 曲率 (第7列)
        original_vertices = data[:, 0:3]
        raw_curv = data[:, 6]

        # 数值保护: 取绝对值并截断到 0-10
        curvatures = np.abs(raw_curv)
        curvatures = np.clip(curvatures, 0, 10.0)
        
        return original_vertices, curvatures.reshape(-1, 1)

    except Exception as e:
        print(f"      [Error] 解析 .vec 失败: {e}")
        return None, None

def run_curvature_tool(obj_path):
    """
    调用外部工具，并设置工作目录以加载 DLL
    """
    # 1. 【新增】严格检查工具是否存在
    if not os.path.exists(TOOL_EXE_PATH):
        print(f"      [Fatal Error] 找不到工具文件！\n      路径: {TOOL_EXE_PATH}")
        return None, None

    tool_dir = os.path.dirname(TOOL_EXE_PATH)
    
    # 2. 【核心修复】cmd 使用绝对路径，而不是仅文件名
    # 原来是: cmd = [os.path.basename(TOOL_EXE_PATH), obj_path]
    # 现在改用: TOOL_EXE_PATH (D:\...\GenerateByVertex.exe)
    cmd = [TOOL_EXE_PATH, obj_path]
    
    print(f"      [Tool] 调用工具计算曲率...")
    try:
        # cwd=tool_dir 依然必须保留，为了让工具找到同级目录的 DLL
        subprocess.run(cmd, cwd=tool_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"      [Error] 工具运行崩溃 (Exit Code {e.returncode})。")
        return None, None
    except FileNotFoundError:
        print(f"      [Error] 系统无法执行文件，请检查路径是否完全正确！")
        return None, None

    # 3. 寻找 .vec
    vec_path = obj_path.replace(".obj", ".vec")
    if not os.path.exists(vec_path):
        vec_path = obj_path + ".vec"
        
    if os.path.exists(vec_path):
        return parse_vec_file(vec_path)
    else:
        print(f"      [Error] 未生成 .vec 文件")
        return None, None

def identify_files(file_list):
    crown, scan, ant = None, None, None
    for f in file_list:
        name = os.path.basename(f)
        if "修复体" in name: crown = f
        elif "对颌牙" in name: ant = f
        elif "模型底板" in name or "扫描模型" in name: scan = f
    return crown, scan, ant

def extract_tooth_number(filename):
    # 从文件名提取牙位，如 "牙齿_16" -> "16"
    match = re.search(r"牙齿_(\d+)", filename)
    if match: return match.group(1)
    return "unknown"

def process_case_folder(case_name, file_list):
    print(f"\n>>> 正在处理病例: {case_name}")
    
    file_crown, file_scan, file_ant = identify_files(file_list)
    if not (file_crown and file_scan and file_ant):
        print(f"    [跳过] 文件不全。Crown:{bool(file_crown)}, Scan:{bool(file_scan)}, Ant:{bool(file_ant)}")
        return

    # 确定输出目录
    tooth_num = extract_tooth_number(os.path.basename(file_crown))
    save_dir = os.path.join(OUTPUT_ROOT, tooth_num, "train", f"case_{case_name}")
    os.makedirs(save_dir, exist_ok=True)

    # ================= 处理 GT (Crown) =================
    print(f"    [1/2] 制作 GT (牙位 {tooth_num})...")
    
    # 使用安全读取函数
    mesh_crown = read_mesh_safe(file_crown)
    if mesh_crown is None: return

    # A. 转 OBJ 供工具使用
    temp_obj = os.path.abspath(os.path.join(save_dir, "temp_gt.obj"))
    o3d.io.write_triangle_mesh(temp_obj, mesh_crown)
    
    # B. 获取高精度曲率
    vec_verts, vec_curv = run_curvature_tool(temp_obj)
    
    # C. 采样
    pcd_gt = mesh_crown.sample_points_poisson_disk(NUM_POINTS_GT)
    
    # D. 映射曲率
    if vec_verts is not None:
        tree = cKDTree(vec_verts)
        _, idxs = tree.query(np.asarray(pcd_gt.points))
        sampled_curv = vec_curv[idxs]
    else:
        print("      [警告] 使用全0曲率 (工具失败)")
        sampled_curv = np.zeros((NUM_POINTS_GT, 1))

    # E. 保存 GT
    if not pcd_gt.has_normals(): pcd_gt.estimate_normals()
    pcd_gt.orient_normals_consistent_tangent_plane(10)
    
    with h5py.File(os.path.join(save_dir, "crown.h5"), 'w') as f:
        f.create_dataset('vertices', data=np.asarray(pcd_gt.points).astype(np.float32))
        f.create_dataset('normals', data=np.asarray(pcd_gt.normals).astype(np.float32))
        f.create_dataset('curvatures', data=sampled_curv.astype(np.float32))

    # ================= 处理 Input (Merge) =================
    print("    [2/2] 制作 Input (合并)...")
    mesh_scan = read_mesh_safe(file_scan)
    mesh_ant = read_mesh_safe(file_ant)
    
    if mesh_scan is None or mesh_ant is None: return
    
    # 合并
    mesh_input = mesh_scan + mesh_ant
    
    # 采样
    pcd_input = mesh_input.sample_points_poisson_disk(NUM_POINTS_INPUT)
    if not pcd_input.has_normals(): pcd_input.estimate_normals()
    pcd_input.orient_normals_consistent_tangent_plane(10)
    
    # 保存 Input
    with h5py.File(os.path.join(save_dir, "pna_crop.h5"), 'w') as f:
        f.create_dataset('vertices', data=np.asarray(pcd_input.points).astype(np.float32))
        f.create_dataset('normals', data=np.asarray(pcd_input.normals).astype(np.float32))
        f.create_dataset('curvatures', data=np.zeros((NUM_POINTS_INPUT, 1)).astype(np.float32))

    # 清理
    try:
        if os.path.exists(temp_obj): os.remove(temp_obj)
        temp_vec = temp_obj.replace(".obj", ".vec")
        if os.path.exists(temp_vec): os.remove(temp_vec)
    except: pass
    
    print(f"    [√] 成功: {save_dir}")

def main():
    # 遍历 D:\vbcd\Dataset 下的所有子文件夹
    if not os.path.exists(RAW_DATA_ROOT):
        print(f"错误: 找不到数据目录 {RAW_DATA_ROOT}")
        return

    subfolders = [f.path for f in os.scandir(RAW_DATA_ROOT) if f.is_dir()]
    print(f"发现 {len(subfolders)} 个病例文件夹...")
    
    for folder in subfolders:
        case_id = os.path.basename(folder)
        stl_files = glob.glob(os.path.join(folder, "*.stl"))
        process_case_folder(case_id, stl_files)

if __name__ == "__main__":
    main()