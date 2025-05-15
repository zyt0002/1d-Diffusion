import os
import argparse
import torch
import numpy as np

from utils import set_seed
from data_module import get_dataloader
from models import MLPDiffusionModel, DeepMLPDiffusionModel
from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
import train as train_module
import generate as generate_module


def parse_args():
    parser = argparse.ArgumentParser(description='1D扩散模型生成心电信号')
    
    # 主要操作模式
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'generate', 'all'],
                        help='操作模式：训练、生成或全部')
    
    # 训练参数
    parser.add_argument('--data_path', type=str, default='data/ecg_all.npy', 
                        help='心电图数据的路径')
    parser.add_argument('--segment_length', type=int, default=1000, 
                        help='心电图段的长度（默认1秒 = 1000点）')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='批次大小')
    parser.add_argument('--model_type', type=str, default='mlp', choices=['mlp', 'deep_mlp'],
                        help='使用的模型类型')
    parser.add_argument('--hidden_dims', type=str, default='1024,2048,1024',
                        help='MLP隐藏层维度，用逗号分隔')
    parser.add_argument('--hidden_dim', type=int, default=1024,
                        help='深层MLP的隐藏层维度')
    parser.add_argument('--depth', type=int, default=8,
                        help='深层MLP的残差块数量')
    parser.add_argument('--time_dim', type=int, default=128,
                        help='时间嵌入的维度')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout概率')
    parser.add_argument('--num_timesteps', type=int, default=10000,
                        help='扩散步数T')
    parser.add_argument('--beta_1', type=float, default=1e-4,
                        help='β1，初始噪声水平')
    parser.add_argument('--beta_T', type=float, default=0.02,
                        help='βT，最终噪声水平')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='权重衰减')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='保存模型的目录')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='每隔多少轮保存一次模型')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='每隔多少批次记录一次日志')
    parser.add_argument('--sample_interval', type=int, default=5,
                        help='每隔多少轮生成一次样本图像')
    
    # 生成参数
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='模型检查点路径')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='要生成的样本数量')
    parser.add_argument('--generated_dir', type=str, default='generated_samples',
                        help='保存生成样本的目录')
    parser.add_argument('--save_npy', action='store_true',
                        help='是否将生成的样本保存为NPY文件')
    parser.add_argument('--save_process', action='store_true',
                        help='是否保存整个生成过程')
    parser.add_argument('--process_steps', type=int, default=10,
                        help='要保存的扩散过程步数')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载的工作线程数')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    if args.mode == 'train' or args.mode == 'all':
        print("===== 开始训练模型 =====")
        # 调用训练模块
        train_args = train_module.parse_args([
            '--data_path', args.data_path,
            '--segment_length', str(args.segment_length),
            '--batch_size', str(args.batch_size),
            '--model_type', args.model_type,
            '--hidden_dims', args.hidden_dims,
            '--hidden_dim', str(args.hidden_dim),
            '--depth', str(args.depth),
            '--time_dim', str(args.time_dim),
            '--dropout', str(args.dropout),
            '--num_timesteps', str(args.num_timesteps),
            '--beta_1', str(args.beta_1),
            '--beta_T', str(args.beta_T),
            '--num_epochs', str(args.num_epochs),
            '--lr', str(args.lr),
            '--weight_decay', str(args.weight_decay),
            '--save_dir', args.save_dir,
            '--save_interval', str(args.save_interval),
            '--log_interval', str(args.log_interval),
            '--sample_interval', str(args.sample_interval),
            '--num_samples', str(args.num_samples),
            '--seed', str(args.seed),
            '--num_workers', str(args.num_workers),
        ])
        train_module.train(train_args)
        
        # 如果模式为all，使用最佳模型进行生成
        if args.mode == 'all':
            args.checkpoint_path = os.path.join(args.save_dir, 'model_best.pt')
    
    if args.mode == 'generate' or args.mode == 'all':
        print("===== 开始生成样本 =====")
        # 确保检查点路径存在
        if args.checkpoint_path is None:
            # 尝试找到最佳模型
            if os.path.exists(os.path.join(args.save_dir, 'model_best.pt')):
                args.checkpoint_path = os.path.join(args.save_dir, 'model_best.pt')
            else:
                # 尝试找到最新的模型
                checkpoints = [f for f in os.listdir(args.save_dir) if f.startswith('model_epoch_') and f.endswith('.pt')]
                if checkpoints:
                    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
                    args.checkpoint_path = os.path.join(args.save_dir, checkpoints[0])
                else:
                    raise ValueError("未指定检查点路径，且无法找到已有的模型")
          # 调用生成模块
        generate_args = generate_module.parse_args([
            '--checkpoint_path', args.checkpoint_path,
            '--num_samples', str(args.num_samples),
            '--save_dir', args.generated_dir,
            '--seed', str(args.seed),
            '--save_process', '' if args.save_process else '--no-save_process',
            '--process_steps', str(args.process_steps),
            '--save_npy', '' if args.save_npy else '--no-save_npy',
            '--data_path', args.data_path,
        ])
        generate_module.generate_samples(generate_args)


if __name__ == "__main__":
    main()
