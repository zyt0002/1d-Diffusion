import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import MLPDiffusionModel, DeepMLPDiffusionModel
from diffusion import GaussianDiffusionSampler
from data_module import ECGDataset
from utils import load_checkpoint, set_seed, get_device, plot_ecg_samples, plot_diffusion_process


def parse_args():
    parser = argparse.ArgumentParser(description='生成心电图信号')
    
    # 模型参数
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                        help='模型检查点路径')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='要生成的样本数量')
    parser.add_argument('--save_dir', type=str, default='generated_samples',
                        help='保存生成样本的目录')
    parser.add_argument('--save_npy', action='store_true',
                        help='是否将生成的样本保存为NPY文件')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save_process', action='store_true',
                        help='是否保存整个生成过程')
    parser.add_argument('--process_steps', type=int, default=1000,
                        help='要保存的扩散过程步数')
    
    return parser.parse_args()


def generate_samples(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载检查点
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    checkpoint_args = checkpoint['args']
    
    # 创建模型
    if checkpoint_args.get('model_type', 'mlp') == 'mlp':
        hidden_dims = [int(dim) for dim in checkpoint_args.get('hidden_dims', '1024,2048,1024').split(',')]
        model = MLPDiffusionModel(
            input_dim=checkpoint_args.get('segment_length', 1000),
            hidden_dims=hidden_dims,
            time_dim=checkpoint_args.get('time_dim', 128),
            dropout=checkpoint_args.get('dropout', 0.1)
        )
        print(f"使用MLP模型，隐藏层维度: {hidden_dims}")
    else:
        model = DeepMLPDiffusionModel(
            input_dim=checkpoint_args.get('segment_length', 1000),
            hidden_dim=checkpoint_args.get('hidden_dim', 1024),
            depth=checkpoint_args.get('depth', 8),
            time_dim=checkpoint_args.get('time_dim', 128),
            dropout=checkpoint_args.get('dropout', 0.1)
        )
        print(f"使用深层MLP模型，隐藏层维度: {checkpoint_args.get('hidden_dim', 1024)}，深度: {checkpoint_args.get('depth', 8)}")
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 获取其他参数
    beta_1 = checkpoint_args.get('beta_1', 1e-4)
    beta_T = checkpoint_args.get('beta_T', 0.02)
    num_timesteps = checkpoint_args.get('num_timesteps', 1000)
    segment_length = checkpoint_args.get('segment_length', 1000)
    
    # 创建扩散采样器
    sampler = GaussianDiffusionSampler(
        model,
        beta_1,
        beta_T,
        num_timesteps,
        segment_length,
        mean_type='epsilon',
        var_type='fixedlarge'
    )
    sampler = sampler.to(device)
    
    # 生成噪声
    z = torch.randn(args.num_samples, 1, segment_length, device=device)
    
    # 如果需要保存生成过程
    if args.save_process:
        # 修改采样器的forward方法以跟踪中间结果
        def generate_with_intermediates(self, x_T):
            intermediates = []
            x_t = x_T
            intermediates.append(x_t.clone())
            
            for time_step in tqdm(reversed(range(self.T)), desc="生成样本"):
                t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
                
                # 在t=0时没有噪声
                if time_step > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0
                    
                x_t = mean + torch.exp(0.5 * log_var) * noise
                
                # 每隔一定步数保存中间结果
                if time_step % (num_timesteps // args.process_steps) == 0 or time_step == 0:
                    intermediates.append(x_t.clone())
            
            return torch.clip(x_t, -1, 1), intermediates
        
        # 保存原始forward方法
        original_forward = sampler.forward
        
        # 替换forward方法
        import types
        sampler.forward = types.MethodType(generate_with_intermediates, sampler)
        
        # 生成样本和中间结果
        print(f"生成 {args.num_samples} 个样本及其中间过程...")
        with torch.no_grad():
            samples, intermediates = sampler(z)
            
        # 恢复原始forward方法
        sampler.forward = original_forward
        
        # 可视化每个样本的生成过程
        for i in range(args.num_samples):
            sample_process = [inter[i:i+1] for inter in intermediates]
            sample_process = torch.cat(sample_process, dim=0)
            
            # 绘制扩散过程
            process_path = os.path.join(args.save_dir, f"process_sample_{i+1}.png")
            plot_diffusion_process(sample_process, save_path=process_path, show=False)
            print(f"样本 {i+1} 的生成过程保存至: {process_path}")
            
    else:
        # 直接生成样本
        print(f"生成 {args.num_samples} 个样本...")
        with torch.no_grad():
            samples = sampler(z)
    
    # 将样本移动到CPU并转换为NumPy数组
    samples_np = samples.cpu().numpy()
    
    # 绘制生成的样本
    samples_path = os.path.join(args.save_dir, "generated_samples.png")
    plot_ecg_samples(samples, save_path=samples_path, show=False)
    print(f"生成的样本保存至: {samples_path}")
    
    # 如果需要保存为NPY文件
    if args.save_npy:
        npy_path = os.path.join(args.save_dir, "generated_samples.npy")
        np.save(npy_path, samples_np)
        print(f"生成的样本数据保存至: {npy_path}")
        
    return samples_np


if __name__ == "__main__":
    args = parse_args()
    generate_samples(args)
