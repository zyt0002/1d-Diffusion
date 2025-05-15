import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class ECGDataset(Dataset):
    """
    用于加载ECG数据的数据集类
    """
    def __init__(self, data_path, segment_length=1000, transform=None, normalize=True):
        """
        初始化ECG数据集
        
        Args:
            data_path (str): 数据文件路径
            segment_length (int): 每个样本的长度（默认为1秒，即1000个点）
            transform (callable, optional): 可选的数据变换
            normalize (bool): 是否归一化数据到[-1, 1]区间
        """
        self.data = np.load(data_path)
        self.segment_length = segment_length
        self.transform = transform
          # 如果需要归一化
        if normalize:
            # 使用Z-score标准化，使数据均值为0，标准差为1
            self.mean = np.mean(self.data)
            self.std = np.std(self.data)
            self.data = (self.data - self.mean) / self.std
            # 保存归一化参数，以便后续可能的反归一化
            self.normalization_params = {'mean': self.mean, 'std': self.std}
        
        # 计算可以生成的segment数量
        self.num_segments = len(self.data) // self.segment_length
        
    def __len__(self):
        return self.num_segments
    
    def denormalize(self, data):
        """
        将标准化的数据转换回原始范围
        
        Args:
            data (torch.Tensor or numpy.ndarray): 标准化后的数据
            
        Returns:
            torch.Tensor or numpy.ndarray: 反标准化后的数据
        """
        if hasattr(self, 'normalization_params'):
            if isinstance(data, torch.Tensor):
                # 如果输入是PyTorch张量
                return data * self.normalization_params['std'] + self.normalization_params['mean']
            else:
                # 如果输入是NumPy数组
                return data * self.normalization_params['std'] + self.normalization_params['mean']
        else:
            # 如果没有进行标准化，则直接返回
            return data
    
    def __getitem__(self, idx):
        """
        获取指定索引的ECG段
        
        Args:
            idx (int): 索引
            
        Returns:
            torch.Tensor: 形状为[1, segment_length]的ECG信号段
        """
        start_idx = idx * self.segment_length
        end_idx = start_idx + self.segment_length
        
        # 提取信号段
        segment = self.data[start_idx:end_idx]
        
        # 转为Tensor
        segment = torch.FloatTensor(segment).view(1, -1)  # [1, segment_length]
        
        # 应用变换（如果有）
        if self.transform:
            segment = self.transform(segment)
            
        return segment


def get_dataloader(data_path, batch_size=64, segment_length=1000, 
                   num_workers=4, shuffle=True, pin_memory=True, train_val_split=0.9):
    """
    创建数据加载器
    
    Args:
        data_path (str): 数据文件路径
        batch_size (int): 批次大小
        segment_length (int): 每个样本的长度
        num_workers (int): 数据加载的工作线程数
        shuffle (bool): 是否打乱数据
        pin_memory (bool): 是否将数据固定到内存中（对GPU训练有帮助）
        train_val_split (float): 训练集占比
        
    Returns:
        tuple: (train_loader, val_loader, dataset) - 返回训练加载器、验证加载器和原始数据集（用于反归一化）
    """
    dataset = ECGDataset(data_path, segment_length)
    
    # 计算划分点
    train_size = int(len(dataset) * train_val_split)
    val_size = len(dataset) - train_size
    
    # 划分数据集
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # 测试数据加载
    print("===== 测试数据加载模块 =====")
    data_path = "data/ecg_all.npy"
    
    try:
        print(f"正在加载数据: {data_path}")
        dataset = ECGDataset(data_path, segment_length=1000)
        print(f"数据集大小: {len(dataset)} 样本")
        
        # 获取并显示一个样本
        sample = dataset[0]
        print(f"样本形状: {sample.shape}")
        print(f"样本最小值: {sample.min().item()}, 最大值: {sample.max().item()}")
        
        # 获取数据加载器
        print("正在创建数据加载器...")
        train_loader, val_loader = get_dataloader(data_path, batch_size=32)
        
        # 打印数据集信息
        print(f"训练集大小: {len(train_loader.dataset)} 样本, 批次数: {len(train_loader)}")
        print(f"验证集大小: {len(val_loader.dataset)} 样本, 批次数: {len(val_loader)}")
        
        # 获取一个批次并检查形状
        print("读取一个批次数据...")
        for batch in train_loader:
            print(f"批次形状: {batch.shape}")
            print(f"批次最小值: {batch.min().item()}, 最大值: {batch.max().item()}")
            break
            
        print("数据加载模块测试成功！")
    except Exception as e:
        print(f"数据加载测试出错: {e}")
