import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def get_device():
    """
    获取可用的设备（GPU或CPU）
    
    Returns:
        torch.device: 可用的设备
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def setup_logging(log_dir):
    """
    设置日志记录
    
    Args:
        log_dir: 日志保存目录
        
    Returns:
        logging.Logger: 日志记录器
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # 配置日志记录器
    logger = logging.getLogger("ecg_diffusion")
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"日志文件: {log_file}")
    
    return logger


def save_checkpoint(path, epoch, model, optimizer, scheduler, train_loss, val_loss, args):
    """
    保存模型检查点
    
    Args:
        path: 保存路径
        epoch: 当前轮数
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        train_loss: 训练损失
        val_loss: 验证损失
        args: 训练参数
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'args': vars(args)
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """
    加载模型检查点
    
    Args:
        path: 检查点路径
        model: 模型
        optimizer: 优化器 (可选)
        scheduler: 学习率调度器 (可选)
        
    Returns:
        dict: 检查点信息
    """
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def set_seed(seed):
    """
    设置随机种子以确保可重复性
    
    Args:
        seed: 随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_ecg_samples(samples, save_path=None, show=True, titles=None):
    """
    绘制心电图样本
    
    Args:
        samples: 形状为 [batch_size, 1, segment_length] 的样本
        save_path: 保存路径 (可选)
        show: 是否显示图形
        titles: 子图标题列表 (可选)
    """
    batch_size = samples.shape[0]
    
    plt.figure(figsize=(12, 2 * batch_size))
    
    for i in range(batch_size):
        plt.subplot(batch_size, 1, i+1)
        plt.plot(samples[i, 0].detach().cpu().numpy())
        
        if titles is not None and i < len(titles):
            plt.title(titles[i])
        else:
            plt.title(f"ECG Sample {i+1}")
            
        plt.ylim(-1.2, 1.2)
        plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
        
    if show:
        plt.show()
    else:
        plt.close()
        

def plot_diffusion_process(samples, save_path=None, show=True, num_timesteps=5):
    """
    绘制扩散过程
    
    Args:
        samples: 形状为 [num_timesteps, 1, segment_length] 的样本
        save_path: 保存路径 (可选)
        show: 是否显示图形
        num_timesteps: 要显示的时间步数量
    """
    n_steps = min(samples.shape[0], num_timesteps)
    indices = np.linspace(0, samples.shape[0]-1, n_steps).astype(int)
    
    plt.figure(figsize=(12, 2 * n_steps))
    
    for i, idx in enumerate(indices):
        plt.subplot(n_steps, 1, i+1)
        plt.plot(samples[idx, 0].detach().cpu().numpy())
        
        if i == 0:
            plt.title("Final Sample (t=0)")
        elif i == n_steps - 1:
            plt.title("Initial Noise (t=T)")
        else:
            plt.title(f"Intermediate Step (t={samples.shape[0]-idx})")
            
        plt.ylim(-1.5, 1.5)
        plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
        
    if show:
        plt.show()
    else:
        plt.close()
