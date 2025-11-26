import h5py
import numpy as np
import os

# ================= 配置区 =================
# 请把这里改成你想要检查的 H5 文件路径
# 例如检查刚才生成的 16号牙 case_1 的 GT 数据
H5_FILE_PATH = r"train_data\16\train\case_1\pna_crop.h5"
# =========================================

def print_h5_as_vec(file_path, num_lines=20):
    if not os.path.exists(file_path):
        print(f"[错误] 找不到文件: {file_path}")
        print("请在脚本的 '配置区' 修改 H5_FILE_PATH 为你实际生成的路径。")
        return

    print(f"正在读取: {file_path}\n")

    try:
        with h5py.File(file_path, 'r') as f:
            # 1. 读取三个数据集
            # H5里它们是分开存的，我们需要读出来
            vertices = np.array(f['vertices'])   # (N, 3) -> X Y Z
            normals = np.array(f['normals'])     # (N, 3) -> NX NY NZ
            curvatures = np.array(f['curvatures']) # (N, 1) -> C

            # 2. 检查行数是否对齐
            if not (len(vertices) == len(normals) == len(curvatures)):
                print("[严重错误] 数据行数不匹配！文件可能已损坏。")
                return

            # 3. 打印表头 (模拟 .vec 格式)
            print(f"{'X':>10} {'Y':>10} {'Z':>10} {'NX':>10} {'NY':>10} {'NZ':>10} {'C':>10}")
            print("-" * 80)

            # 4. 循环打印前 N 行
            # 我们把三个数组的数据拼在一起显示
            for i in range(min(num_lines, len(vertices))):
                v = vertices[i]
                n = normals[i]
                c = curvatures[i]
                
                # 格式化输出：保留6位小数，和你的截图类似
                print(f"{v[0]:10.6f} {v[1]:10.6f} {v[2]:10.6f} "
                      f"{n[0]:10.6f} {n[1]:10.6f} {n[2]:10.6f} "
                      f"{c[0]:10.6f}")

            print("-" * 80)
            print(f"\n总行数: {len(vertices)}")
            print("注意：由于 H5 数据经过了 '泊松盘采样' (Sampling)，")
            print("这里的坐标点与原始 .vec 文件中的坐标点位置不会完全一一对应（行号不对应），")
            print("但数值范围、物体形状和曲率特征应该是一致的。")

    except Exception as e:
        print(f"读取出错: {e}")

if __name__ == "__main__":
    print_h5_as_vec(H5_FILE_PATH, num_lines=20)