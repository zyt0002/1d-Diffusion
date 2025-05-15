import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
from scipy import signal

from data_module import ECGDataset
from utils import set_seed

# matplotlib设置中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

def parse_args():
    parser = argparse.ArgumentParser(description='可视化ECG数据和生成的样本')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='data/ecg_all.npy',
                        help='原始心电图数据的路径')
    parser.add_argument('--generated_path', type=str, default=None,
                        help='生成的心电图样本的路径(NPY文件)')
    parser.add_argument('--segment_length', type=int, default=1000,
                        help='心电图段的长度（默认1秒 = 1000点）')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='要显示的样本数量')
    parser.add_argument('--save_dir', type=str, default='plots',
                        help='保存图像的目录')
    parser.add_argument('--create_animation', action='store_true',
                        help='是否创建动画')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    return parser.parse_args()


def plot_sample_comparison(real_samples, generated_samples, save_path=None, show=True):
    """
    绘制真实样本和生成样本的对比图
    
    Args:
        real_samples: 真实样本，形状为 [batch_size, 1, segment_length]
        generated_samples: 生成样本，形状为 [batch_size, 1, segment_length]
        save_path: 保存路径 (可选)
        show: 是否显示图形
    """
    num_samples = min(real_samples.shape[0], generated_samples.shape[0])
    
    plt.figure(figsize=(15, 3 * num_samples))
    
    for i in range(num_samples):
        # 绘制真实样本
        plt.subplot(num_samples, 2, 2*i+1)
        plt.plot(real_samples[i, 0])
        plt.title(f"真实心电图样本 {i+1}")
        plt.ylim(-1.2, 1.2)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 绘制生成样本
        plt.subplot(num_samples, 2, 2*i+2)
        plt.plot(generated_samples[i, 0])
        plt.title(f"生成心电图样本 {i+1}")
        plt.ylim(-1.2, 1.2)
        plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
        
    if show:
        plt.show()
    else:
        plt.close()


def plot_psd_comparison(real_samples, generated_samples, fs=1000, save_path=None, show=True):
    """
    绘制功率谱密度对比图
    
    Args:
        real_samples: 真实样本，形状为 [batch_size, 1, segment_length]
        generated_samples: 生成样本，形状为 [batch_size, 1, segment_length]
        fs: 采样率 (Hz)
        save_path: 保存路径 (可选)
        show: 是否显示图形
    """
    num_samples = min(real_samples.shape[0], generated_samples.shape[0])
    
    plt.figure(figsize=(15, 3 * num_samples))
    
    for i in range(num_samples):
        # 计算真实样本的PSD
        f_real, psd_real = signal.welch(real_samples[i, 0], fs=fs, nperseg=256)
        
        # 计算生成样本的PSD
        f_gen, psd_gen = signal.welch(generated_samples[i, 0], fs=fs, nperseg=256)
        
        # 绘制真实样本的PSD
        plt.subplot(num_samples, 2, 2*i+1)
        plt.semilogy(f_real, psd_real)
        plt.title(f"真实心电图样本 {i+1} 的功率谱密度")
        plt.xlabel('频率 (Hz)')
        plt.ylabel('功率/频率 (dB/Hz)')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 绘制生成样本的PSD
        plt.subplot(num_samples, 2, 2*i+2)
        plt.semilogy(f_gen, psd_gen)
        plt.title(f"生成心电图样本 {i+1} 的功率谱密度")
        plt.xlabel('频率 (Hz)')
        plt.ylabel('功率/频率 (dB/Hz)')
        plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
        
    if show:
        plt.show()
    else:
        plt.close()


def create_ecg_animation(samples, save_path=None, interval=20):
    """
    创建心电图的动画
    
    Args:
        samples: 样本，形状为 [batch_size, 1, segment_length]
        save_path: 保存路径 (可选)
        interval: 帧间隔 (毫秒)
    """
    num_samples = samples.shape[0]
    segment_length = samples.shape[2]
    
    fig, axs = plt.subplots(num_samples, 1, figsize=(10, 2 * num_samples))
    if num_samples == 1:
        axs = [axs]
    
    lines = []
    for i in range(num_samples):
        line, = axs[i].plot([], [])
        axs[i].set_xlim(0, segment_length)
        axs[i].set_ylim(-1.2, 1.2)
        axs[i].set_title(f"ECG Sample {i+1}")
        axs[i].grid(True, linestyle='--', alpha=0.5)
        lines.append(line)
    
    plt.tight_layout()
    
    def init():
        for line in lines:
            line.set_data([], [])
        return lines
    
    def animate(frame):
        # 显示300个点，并随着时间滚动
        window_size = 300
        for i, line in enumerate(lines):
            x = np.arange(max(0, frame - window_size + 1), frame + 1)
            y = samples[i, 0, max(0, frame - window_size + 1):frame + 1]
            line.set_data(x, y)
        return lines
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=segment_length,
                         interval=interval, blit=True)
    
    if save_path is not None:
        # 保存为GIF (需要安装imagemagick)
        anim.save(save_path, writer='pillow', fps=30)
        print(f"动画保存至: {save_path}")
    else:
        plt.show()
        
    plt.close()


def main(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载真实数据
    dataset = ECGDataset(args.data_path, segment_length=args.segment_length)
    
    # 随机选择真实样本
    indices = np.random.choice(len(dataset), args.num_samples, replace=False)
    real_samples = np.array([dataset[i].numpy() for i in indices])
    
    # 加载生成的样本（如果提供）
    if args.generated_path is not None:
        if args.generated_path.endswith('.npy'):
            generated_samples = np.load(args.generated_path)
            if len(generated_samples.shape) == 2:
                # 转换形状到 [batch_size, 1, segment_length]
                generated_samples = generated_samples.reshape(-1, 1, args.segment_length)
            
            # 限制样本数量
            generated_samples = generated_samples[:args.num_samples]
            
            # 绘制真实样本和生成样本的对比图
            comparison_path = os.path.join(args.save_dir, "sample_comparison.png")
            plot_sample_comparison(real_samples, generated_samples, save_path=comparison_path, show=False)
            print(f"样本对比图保存至: {comparison_path}")
            
            # 绘制功率谱密度对比图
            psd_path = os.path.join(args.save_dir, "psd_comparison.png")
            plot_psd_comparison(real_samples, generated_samples, save_path=psd_path, show=False)
            print(f"功率谱密度对比图保存至: {psd_path}")
    else:
        # 只绘制真实样本
        plt.figure(figsize=(12, 2 * args.num_samples))
        for i in range(args.num_samples):
            plt.subplot(args.num_samples, 1, i+1)
            plt.plot(real_samples[i, 0])
            plt.title(f"真实心电图样本 {i+1}")
            plt.ylim(-1.2, 1.2)
            plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        real_samples_path = os.path.join(args.save_dir, "real_samples.png")
        plt.savefig(real_samples_path)
        plt.close()
        print(f"真实样本图保存至: {real_samples_path}")
    
    # 创建动画
    if args.create_animation:
        if args.generated_path is not None:
            # 创建生成样本的动画
            gen_animation_path = os.path.join(args.save_dir, "generated_ecg_animation.gif")
            create_ecg_animation(generated_samples, save_path=gen_animation_path)
        
        # 创建真实样本的动画
        real_animation_path = os.path.join(args.save_dir, "real_ecg_animation.gif")
        create_ecg_animation(real_samples, save_path=real_animation_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
