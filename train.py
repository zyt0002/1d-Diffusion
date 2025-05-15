import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 添加这一行
from tqdm import tqdm
import time
import json

from models import MLPDiffusionModel, DeepMLPDiffusionModel
from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from data_module import get_dataloader
from utils import save_checkpoint, setup_logging, get_device


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description='训练1D扩散模型生成心电信号')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='data/ecg_all.npy', 
                        help='心电图数据的路径')
    parser.add_argument('--segment_length', type=int, default=1000, 
                        help='心电图段的长度（默认1秒 = 1000点）')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='批次大小')
    
    # 模型参数
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
    
    # 扩散模型参数
    parser.add_argument('--num_timesteps', type=int, default=10000,
                        help='扩散步数T')
    parser.add_argument('--beta_1', type=float, default=1e-4,
                        help='β1，初始噪声水平')
    parser.add_argument('--beta_T', type=float, default=0.02,
                        help='βT，最终噪声水平')
    
    # 训练参数
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
    parser.add_argument('--num_samples', type=int, default=5,
                        help='生成样本的数量')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载的工作线程数')
    
    return parser.parse_args(args_list)


def train(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(args.save_dir)
    logger.info(f"参数: {args}")
    
    # 保存参数
    with open(os.path.join(args.save_dir, 'params.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # 获取数据加载器
    train_loader, val_loader = get_dataloader(
        args.data_path, 
        batch_size=args.batch_size,
        segment_length=args.segment_length,
        num_workers=args.num_workers
    )
    logger.info(f"训练集: {len(train_loader.dataset)} 样本，验证集: {len(val_loader.dataset)} 样本")
    
    # 创建模型
    if args.model_type == 'mlp':
        hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
        model = MLPDiffusionModel(
            input_dim=args.segment_length,
            hidden_dims=hidden_dims,
            time_dim=args.time_dim,
            dropout=args.dropout
        )
        logger.info(f"使用MLP模型，隐藏层维度: {hidden_dims}")
    else:
        model = DeepMLPDiffusionModel(
            input_dim=args.segment_length,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            time_dim=args.time_dim,
            dropout=args.dropout
        )
        logger.info(f"使用深层MLP模型，隐藏层维度: {args.hidden_dim}，深度: {args.depth}")
    
    # 创建扩散训练器和采样器
    diffusion_trainer = GaussianDiffusionTrainer(
        model, 
        args.beta_1, 
        args.beta_T, 
        args.num_timesteps
    )
    
    diffusion_sampler = GaussianDiffusionSampler(
        model,
        args.beta_1,
        args.beta_T,
        args.num_timesteps,
        args.segment_length,
        mean_type='epsilon',
        var_type='fixedlarge'
    )
    
    # 将模型移动到设备
    diffusion_trainer = diffusion_trainer.to(device)
    diffusion_sampler = diffusion_sampler.to(device)
    
    # 打印模型参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {num_params:,}")
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练循环
    global_step = 0
    best_val_loss = float('inf')
    
    # 为可视化准备固定的噪声
    fixed_noise = torch.randn(args.num_samples, 1, args.segment_length, device=device)
    
    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"Epoch {epoch}/{args.num_epochs}")
        
        # 训练阶段
        diffusion_trainer.train()
        train_loss = 0.0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"训练 Epoch {epoch}")
        for batch_idx, data in enumerate(progress_bar):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # 计算损失
            loss = diffusion_trainer(data).mean()
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 累计损失
            train_loss += loss.item()
            train_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({"loss": loss.item()})
            
            # 记录日志
            if batch_idx % args.log_interval == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
            
            global_step += 1
        
        # 计算平均训练损失
        avg_train_loss = train_loss / train_batches
        logger.info(f"平均训练损失: {avg_train_loss:.6f}")
        
        # 验证阶段
        diffusion_trainer.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"验证 Epoch {epoch}"):
                data = data.to(device)
                loss = diffusion_trainer(data).mean()
                val_loss += loss.item()
                val_batches += 1
        
        # 计算平均验证损失
        avg_val_loss = val_loss / val_batches
        logger.info(f"平均验证损失: {avg_val_loss:.6f}")
        
        # 学习率调度器
        scheduler.step(avg_val_loss)
        
        # 保存检查点
        if epoch % args.save_interval == 0 or epoch == args.num_epochs:
            checkpoint_path = os.path.join(args.save_dir, f"model_epoch_{epoch}.pt")
            save_checkpoint(
                checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                avg_train_loss,
                avg_val_loss,
                args
            )
            logger.info(f"模型保存至: {checkpoint_path}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint_path = os.path.join(args.save_dir, "model_best.pt")
            save_checkpoint(
                best_checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                avg_train_loss,
                avg_val_loss,
                args
            )
            logger.info(f"最佳模型保存至: {best_checkpoint_path}")
        
        # 生成样本
        if epoch % args.sample_interval == 0 or epoch == args.num_epochs:
            diffusion_sampler.eval()
            with torch.no_grad():
                # 从固定噪声生成样本
                samples = diffusion_sampler(fixed_noise)
                samples = samples.cpu().numpy()
                
                # 创建图形目录
                samples_dir = os.path.join(args.save_dir, "samples")
                os.makedirs(samples_dir, exist_ok=True)
                
                # 保存图像
                plt.figure(figsize=(12, 8))
                for i in range(min(args.num_samples, samples.shape[0])):
                    plt.subplot(args.num_samples, 1, i+1)
                    plt.plot(samples[i, 0])
                    plt.title(f"Generated ECG Sample {i+1}")
                    plt.ylim(-1.2, 1.2)
                
                plt.tight_layout()
                plt.savefig(os.path.join(samples_dir, f"samples_epoch_{epoch}.png"))
                plt.close()
                logger.info(f"样本保存至: {samples_dir}/samples_epoch_{epoch}.png")
    
    logger.info("训练完成！")
    

if __name__ == "__main__":
    args = parse_args()
    train(args)
