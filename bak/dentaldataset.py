import os
from pathlib import Path
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

class IOS_Datasetv2(Dataset):
    def __init__(self, root_dir, is_train=True, crop_size=(2.00, 2.00, 2.00), voxel_size=(0.15625, 0.15625, 0.15625)):
        #root_dir这个参数的作用是指定数据集的根目录
        #is_train=True意味着加载训练集数据，否则加载测试集数据
        #crop_size设定为(2.00, 2.00, 2.00)表示裁剪区域的大小，单位为厘米，设计这个大小的目的是为了将关键区域包含在内，同时减少计算量
        self.root_dir = Path(root_dir)#Path函数的作用是将字符串路径转换为Path对象，这个功能强大
        self.crop_size = crop_size
        self.voxel_size = voxel_size
#         self.subdirs = ['11', '12', '13','14', '15', '16', '17', 
#  '21', '22','23', '24', '25', '26', '27', 
#  '31', '32','33','34', '35', '36', '37', 
#  '41', '42', '43','44', '45', '46', '47']
        self.subdirs = ['16']
        self.data_paths = []
        for subdir in self.subdirs:
            subdir_path = self.root_dir / subdir
            if is_train:#根据is_train参数决定加载训练集还是测试集
                case_dir = subdir_path / 'train' #如果是训练集，则路径指向train子目录
            else:
                case_dir = subdir_path / 'test' #如果是测试集，则路径指向test子目录
            for case in os.listdir(case_dir):#遍历训练集或者测试集目录下的每个病例文件夹，case的值为病例文件夹的名称，比如
                abs_case = os.path.join(case_dir, case)
                crown_file = os.path.join(abs_case, 'crown.h5')
                pna_crop_file = os.path.join(abs_case, 'pna_crop.h5')
                self.data_paths.append((abs_case, crown_file, pna_crop_file))
                #形成类似与这种格式的元组:self.data_paths.append((
                #     '/home/data/dental_scans/46/train/patient_A_case_001',   # abs_case
                #     '/home/data/dental_scans/46/train/patient_A_case_001/crown.h5',  # crown_file (GT)
                #     '/home/data/dental_scans/46/train/patient_A_case_001/pna_crop.h5' # pna_crop_file (Input)
                # ))

    def crop_mesh(self, mesh, center, crop_size):#这个函数的作用是裁剪点云数据
        #mesh:输入的点云数据，通常是一个包含点坐标和法向量的数组，例如这种形式:[[x1, y1, z1, nx1, ny1, nz1], [x2, y2, z2, nx2, ny2, nz2], ...]
        points = np.asarray(mesh)#将输入的点云数据转换为NumPy数组，方便后续处理

        positions = points[:, :3]#提取点云中所有点的坐标
        normals = points[:, 3:]#提取点云中所有点的法向量

        half_crop_size = 10*np.array(crop_size) / 2#计算裁剪区域的一半大小，单位转换为毫米
        min_bound = center - half_crop_size
        max_bound = center + half_crop_size
        #上面这三行代码的作用是依据中心点为中心，获得裁剪区域的最小和最大边界坐标，比如中心点为[50, 50, 50]，裁剪大小为(2.00, 2.00, 2.00)，那么乘以10除以2之后，min_bound为[40, 40, 40]，max_bound为[60, 60, 60]
        #所以只要有点的坐标在这个范围内，就会被保留下来
        mask = np.all((positions >= min_bound) & (positions <= max_bound), axis=1)#axis =1的作用是对每一行进行逻辑与操作，确保点的x、y、z坐标都在指定范围内
        cropped_positions = positions[mask]
        cropped_normals = normals[mask]
        #上面这两行代码的作用是根据掩码筛选出在裁剪区域内的点的坐标和法向量
        cropped_points = np.hstack((cropped_positions, cropped_normals))#hstack的作用是水平堆叠数组,还原成点云的标注形式：[[x1, y1, z1, nx1, ny1, nz1], [x2, y2, z2, nx2, ny2, nz2], ...]
        return cropped_points
    
    def __len__(self):  #返回数据集的大小
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        #这个代码根据索引idx加载对应路径的文件，每一个idx对应一个病例文件夹
        dirpath, crown_file, pna_crop_file = self.data_paths[idx]
        with h5py.File(crown_file, 'r') as f:
            crown_vertices = np.array(f['vertices'])
            crown_normals = np.array(f['normals'])
            curvatures = np.array(f['curvatures'])
            #上述的代码的作用是加载真实牙冠的点云数据，包括顶点坐标、法向量和曲率
        with h5py.File(pna_crop_file, 'r') as f:
            pna_crop_vertices = np.array(f['vertices'])
            pna_crop_normals = np.array(f['normals'])
        crown_min_bound = crown_vertices.min(axis=0)
        crown_max_bound = crown_vertices.max(axis=0)
        crown_center = (crown_min_bound + crown_max_bound) / 2#通过计算最小和最大边界的中点，获得牙冠点云的中心位置

        half_crop_size = np.array(self.crop_size) * 10 / 2
        min_bound_crop = crown_center - half_crop_size
        max_bound_crop = crown_center + half_crop_size
        point_cloud_full_inform = np.concatenate([crown_vertices,crown_normals,curvatures.reshape(-1,1)],axis=1)#将真实牙冠的顶点坐标、法向量和曲率信息拼接在一起，形成完整的点云信息，形状为(N, 7)，其中N是点的数量，7表示每个点包含3个坐标、3个法向量分量和1个曲率值，其中axis=1表示在列方向上进行拼接
        crown_voxel_grid = create_voxelwithnormal_grid(point_cloud_full_inform, min_bound_crop, max_bound_crop, self.voxel_size)
        #上述代码的作用是将裁剪区域内的真实牙冠点云转换为体素网格表示，方便后续的体素处理和模型输入
        crown_tensor = torch.from_numpy(crown_voxel_grid).float()#将体素网格数据转换为PyTorch张量，并指定数据类型为浮点型
        pna_crop = self.crop_mesh(np.concatenate([pna_crop_vertices,pna_crop_normals],axis=1),crown_center, self.crop_size)
        #上述代码是在操作pna_crop也就是待补牙的点云数据，从真实牙冠中心点位置裁剪周围区域，将邻牙、对颌牙以及预备牙等重要信息包裹在内
        pna_crop_tensor = torch.tensor(pna_crop, dtype=torch.float32)#将裁剪后的待补牙点云数据转换为PyTorch张量，并指定数据类型为浮点型
        pna_crop_tensor = self.normalize_point_cloud(pna_crop_tensor,cropsize=self.crop_size)#对裁剪后的待补牙点云数据进行归一化处理，使其坐标范围适合神经网络输入
        point_cloud_crown_inform = torch.tensor(point_cloud_full_inform,dtype=torch.float32)#将包含真实牙冠完整信息的点云数据转换为PyTorch张量，并指定数据类型为浮点型
        return pna_crop_tensor[:,:3], crown_tensor, point_cloud_crown_inform, torch.tensor(min_bound_crop,dtype=torch.float32),dirpath
    
    def collate_fn(self, batch):
        pna_crop_tensor,crown_tensor,point_cloud_crown_tensor,min_bound_crop,dirpath = zip(*batch)#zip函数的作用是将batch中的每个元素按照位置进行分组，形成多个元组，每个元组包含对应位置的所有元素
        point_cloud_crown_tensor = [pc for pc in point_cloud_crown_tensor]#取出每个样本的真实牙冠点云数据，形成一个列表
        combined_point_cloud = torch.cat(point_cloud_crown_tensor, dim=0)

        batch_sizes = [pc.shape[0] for pc in point_cloud_crown_tensor]
        batch_indices = torch.cat([torch.full((size,), i, dtype=torch.long) for i, size in enumerate(batch_sizes)])

        return pna_crop_tensor,torch.stack(crown_tensor),combined_point_cloud,batch_indices,torch.stack(min_bound_crop),dirpath
    
    def normalize_point_cloud(self, point_cloud,cropsize):
        if not isinstance(point_cloud, torch.Tensor):
            point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
        if point_cloud.shape[1] != 6:
            raise ValueError("Point cloud should have shape (num_points, 6)")
        
        positions = point_cloud[:, :3]
        normals = point_cloud[:, 3:]

        point_cloud_center = (torch.min(positions, dim=0)[0] + torch.max(positions, dim=0)[0]) / 2
        crop_center = 10*torch.tensor(cropsize, dtype=torch.float32) / 2
        crop_scale = 10*torch.tensor(cropsize, dtype=torch.float32)

        normalized_positions = (positions - point_cloud_center + crop_center) / crop_scale
        normalized_positions = (normalized_positions - 0.5) * 2

        normalized_point_cloud = torch.cat((normalized_positions, normals), dim=1)

        return normalized_point_cloud
    
    # ...existing code...
def create_voxelwithnormal_grid(point_cloud, min_bound, max_bound, voxel_size):
    """
    将带法向与曲率的牙冠点云转换为体素网格 (占据 + 平均法向).
    参数:
        point_cloud: np.ndarray (N,7) [x,y,z,nx,ny,nz,curv]
        min_bound: np.ndarray/list (3,) 裁剪区域最小坐标 (与 __getitem__ 中的 min_bound_crop 对应)
        max_bound: np.ndarray/list (3,) 裁剪区域最大坐标
        voxel_size: tuple/list (3,) 每个维度体素尺寸 (与 self.voxel_size 一致)
    返回:
        grid: np.ndarray (4, D, H, W)
              [0] 占据, [1:4] 平均法向 (nx,ny,nz)
    """
    if point_cloud.size == 0:
        return np.zeros((4, 1, 1, 1), dtype=np.float32)

    min_bound = np.asarray(min_bound, dtype=np.float32)
    max_bound = np.asarray(max_bound, dtype=np.float32)
    voxel_size = np.asarray(voxel_size, dtype=np.float32)

    # 计算体素网格尺寸(向上取整)，保证至少1
    spans = max_bound - min_bound
    grid_size = np.maximum(1, np.ceil(spans / voxel_size).astype(int))
    D, H, W = grid_size.tolist()

    # 分离坐标与法向
    coords = point_cloud[:, :3].astype(np.float32)
    normals = point_cloud[:, 3:6].astype(np.float32)

    # 计算每个点落入体素索引
    indices = np.floor((coords - min_bound) / voxel_size).astype(int)

    # 过滤越界
    in_bounds = np.all((indices >= 0) & (indices < grid_size), axis=1)
    indices = indices[in_bounds]
    normals = normals[in_bounds]

    # 预分配
    occ = np.zeros((D, H, W), dtype=np.float32)
    nsum = np.zeros((3, D, H, W), dtype=np.float32)
    count = np.zeros((D, H, W), dtype=np.int32)

    # 累加
    for (x, y, z), n in zip(indices, normals):
        occ[x, y, z] = 1.0
        nsum[:, x, y, z] += n
        count[x, y, z] += 1

    # 计算平均法向并归一化
    mask = count > 0
    avg_normals = np.zeros_like(nsum)
    avg_normals[:, mask] = nsum[:, mask] / count[mask]

    # 防止除0与长度接近0
    lengths = np.linalg.norm(avg_normals, axis=0, keepdims=True)
    nz_mask = lengths > 1e-6
    avg_normals[:, nz_mask[0]] /= lengths[:, nz_mask[0]]

    # 组装输出
    grid = np.zeros((4, D, H, W), dtype=np.float32)
    grid[0] = occ
    grid[1:4] = avg_normals
    return grid
# ...existing code...