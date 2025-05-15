# 1D扩散模型生成心电信号项目 - 总结与使用指南

## 项目概述

该项目实现了基于扩散模型（DDPM）的1D心电信号生成系统。项目利用多层感知机（MLP）作为基础网络，通过学习心电信号的噪声分布，能够从随机噪声生成新的、高质量的心电信号样本。

## 项目结构

```
├── data/                     # 数据目录
│   └── ecg_all.npy          # 原始心电图数据
├── checkpoints/              # 模型保存目录
├── generated_samples/        # 生成样本保存目录
├── plots/                    # 可视化图表保存目录
├── convert_data.py           # 数据转换脚本
├── data_module.py            # 数据加载和预处理
├── diffusion.py              # 扩散模型算法实现
├── generate.py               # 生成新的心电图信号
├── main.py                   # 主程序入口
├── models.py                 # MLP模型架构定义
├── plot.py                   # 可视化工具
├── train.py                  # 训练扩散模型
└── utils.py                  # 辅助函数
```

## 核心模块说明

### 1. 数据处理模块 (data_module.py)

这个模块负责从原始心电图数据（ecg_all.npy）读取并处理数据。主要功能：
- 将连续心电图数据分割成固定长度（默认1000点，即1秒）的片段
- 归一化数据到[-1, 1]范围
- 创建PyTorch数据集和数据加载器

### 2. 模型定义模块 (models.py)

提供两种基于MLP的网络结构：
- `MLPDiffusionModel`：基础MLP模型，支持配置多层隐藏层
- `DeepMLPDiffusionModel`：深层MLP模型，使用残差连接提高性能

两种模型都包含时间步嵌入，用于条件化扩散过程中的不同噪声级别。

### 3. 扩散模型实现 (diffusion.py)

实现了DDPM算法的核心组件：
- `GaussianDiffusionTrainer`：实现前向扩散过程的训练
- `GaussianDiffusionSampler`：实现反向去噪采样过程

### 4. 训练模块 (train.py)

管理模型的训练过程，包括：
- 损失计算
- 优化器配置
- 学习率调度
- 模型保存和验证
- 训练中间结果可视化

### 5. 生成模块 (generate.py)

使用训练好的模型生成新的心电图样本，支持：
- 批量生成多个样本
- 保存生成过程的中间状态
- 导出为不同的格式

### 6. 可视化工具 (plot.py)

提供丰富的可视化功能：
- 真实与生成样本对比
- 功率谱密度（PSD）分析
- 心电图动画生成
- 扩散过程可视化

## 使用指南

### 环境准备

首先安装必要的依赖包：

```powershell
pip install torch numpy matplotlib tqdm scipy
```

### 训练模型

可以通过主程序或直接使用训练脚本：

```powershell
# 使用主程序训练
python main.py --mode train --model_type mlp --num_epochs 100 --batch_size 64

# 或直接使用训练脚本
python train.py --model_type mlp --num_epochs 100 --batch_size 64
```

#### 主要训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_type` | 模型类型 (mlp/deep_mlp) | mlp |
| `--hidden_dims` | MLP隐藏层维度 | 1024,2048,1024 |
| `--hidden_dim` | 深层MLP的隐藏层维度 | 1024 |
| `--depth` | 深层MLP的残差块数量 | 8 |
| `--num_timesteps` | 扩散步数 | 1000 |
| `--beta_1` | 初始噪声水平 | 1e-4 |
| `--beta_T` | 最终噪声水平 | 0.02 |
| `--batch_size` | 批次大小 | 64 |
| `--lr` | 学习率 | 2e-4 |
| `--num_epochs` | 训练轮数 | 100 |

### 生成样本

使用训练好的模型生成新的心电图样本：

```powershell
# 使用主程序生成
python main.py --mode generate --checkpoint_path checkpoints/model_best.pt --num_samples 10

# 或直接使用生成脚本
python generate.py --checkpoint_path checkpoints/model_best.pt --num_samples 10 --save_npy --save_process
```

#### 主要生成参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--checkpoint_path` | 模型检查点路径 | 必填 |
| `--num_samples` | 生成样本数量 | 10 |
| `--save_npy` | 是否保存为NPY文件 | False |
| `--save_process` | 是否保存生成过程 | False |
| `--process_steps` | 保存的生成过程步数 | 10 |

### 可视化

可视化真实样本和生成样本的对比：

```powershell
python plot.py --data_path data/ecg_all.npy --generated_path generated_samples/generated_samples.npy --create_animation
```

## 训练和生成流程示例

### 完整训练流程

```powershell
# 1. 训练基础MLP模型，50轮
python train.py --model_type mlp --num_epochs 50 --batch_size 32 --save_interval 5

# 2. 使用训练好的模型生成样本
python generate.py --checkpoint_path checkpoints/model_best.pt --num_samples 20 --save_npy --save_process

# 3. 可视化对比结果
python plot.py --data_path data/ecg_all.npy --generated_path generated_samples/generated_samples.npy
```

### 快速测试

对于快速测试系统是否正常工作：

```powershell
# 使用较少的轮数训练
python train.py --model_type mlp --num_epochs 5 --batch_size 16 --hidden_dims 512,512

# 生成少量样本查看效果
python generate.py --checkpoint_path checkpoints/model_epoch_5.pt --num_samples 3
```

## 模型选择建议

1. **基础MLP模型**：
   - 训练速度快，参数量较少
   - 适合快速实验和迭代
   - 对硬件要求较低

2. **深层MLP模型**：
   - 拥有更强的表达能力，可以生成更高质量的样本
   - 训练时间更长，需要更多计算资源
   - 适合追求更高质量生成结果的场景

## 扩展与自定义

本项目易于扩展和自定义：

1. **调整网络结构**：
   - 在models.py中修改现有模型或添加新的模型结构
   - 可以尝试不同的激活函数、更深的网络等

2. **使用其他数据**：
   - 修改data_module.py以支持其他形式的1D信号数据
   - 可根据需要调整预处理和归一化方法

3. **调整扩散过程**：
   - 在diffusion.py中修改噪声调度或采样策略
   - 可以尝试不同的$\beta$值分布

## 故障排除

1. **训练损失不收敛**：
   - 尝试降低学习率
   - 减小批次大小
   - 检查数据归一化是否正确

2. **生成样本质量差**：
   - 增加训练轮数
   - 尝试使用更深的模型
   - 调整扩散步数（num_timesteps）

3. **内存不足**：
   - 减小批次大小
   - 使用基础MLP而非深层MLP
   - 减小隐藏层维度

4. **训练速度慢**：
   - 减小模型规模
   - 减少扩散步数
   - 如果有GPU，确保模型在GPU上运行

通过本项目，您可以深入了解扩散模型的原理和实现，并将其应用于1D心电信号的生成。该框架也可以扩展到其他类型的1D时间序列信号生成任务。