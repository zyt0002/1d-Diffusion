import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm

def convert_ecg_to_npy():
    """
    将CSV格式的ECG数据转换为NPY格式，方便后续使用
    采样率：1000Hz
    """
    # 设置数据路径（根据您的实际路径调整）
    csv_path = "../Dataset Files/Kensas/.pats/*pred.csv"
    output_dir = "./data"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有的CSV文件
    csv_files = glob.glob(csv_path)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found at path: {csv_path}")
    
    # 初始化数据列表
    all_ecg_data = []
    
    # 遍历每个文件提取ECG数据
    for file in tqdm(csv_files, desc="Processing files"):
        try:
            data = pd.read_csv(file)
            if 'ECG' not in data.columns:
                continue
            all_ecg_data.append(data['ECG'].values)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not all_ecg_data:
        raise ValueError("No valid ECG data loaded.")
    
    # 保存训练、验证和测试数据
    all_ecg_combined = []
    
    for i, ecg_sample in enumerate(all_ecg_data):
        total_len = len(ecg_sample)
        
        # 划分数据集
        train_idx = int(0.7 * total_len)
        val_idx = int(0.8 * total_len)
        
        # 添加到相应的数据集
        all_ecg_combined.extend(ecg_sample)
    
    # 将数据转换为numpy数组
    all_ecg_np = np.array(all_ecg_combined)
    
    # 保存为npy文件
    np.save(os.path.join(output_dir, "ecg_all.npy"), all_ecg_np)
    
    print(f"Data conversion completed. Total samples: {len(all_ecg_np)}")
    print(f"Data saved to {output_dir}/ecg_all.npy")
    
    return os.path.join(output_dir, "ecg_all.npy")

if __name__ == "__main__":
    convert_ecg_to_npy()
