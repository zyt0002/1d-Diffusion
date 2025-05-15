import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeEmbedding(nn.Module):
    """
    时间步嵌入模块，将时间步转换为高维向量表示
    """
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, t):
        """
        Args:
            t: 时间步，形状为 [batch_size]
            
        Returns:
            时间嵌入，形状为 [batch_size, dim]
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding


class MLPDiffusionModel(nn.Module):
    """
    用于扩散模型的MLP网络，预测给定噪声图像的噪声
    """
    def __init__(self, input_dim=1000, hidden_dims=[1024, 2048, 1024], time_dim=128, dropout=0.1):
        """
        初始化MLP扩散模型
        
        Args:
            input_dim (int): 输入信号的长度
            hidden_dims (list): 隐藏层维度列表
            time_dim (int): 时间嵌入的维度
            dropout (float): Dropout概率
        """
        super().__init__()
        
        # 时间嵌入
        self.time_embedding = TimeEmbedding(time_dim)
        
        # 时间嵌入的线性投影
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )
        
        # 输入层
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        # 隐藏层
        self.layers = nn.ModuleList([])
        
        in_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            self.layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout)
                )
            )
            in_dim = h_dim
            
        # 输出层
        self.output_proj = nn.Linear(hidden_dims[-1], input_dim)
        
    def forward(self, x, t):
        """
        前向传播
        
        Args:
            x: 输入噪声图像 [batch_size, 1, input_dim]
            t: 时间步 [batch_size]
            
        Returns:
            预测的噪声 [batch_size, 1, input_dim]
        """
        # 打平输入
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)  # [batch_size, input_dim]
        
        # 获取时间嵌入
        t_emb = self.time_embedding(t)  # [batch_size, time_dim]
        t_emb = self.time_proj(t_emb)  # [batch_size, hidden_dims[0]]
        
        # 输入投影
        h = self.input_proj(x_flat)  # [batch_size, hidden_dims[0]]
        
        # 加入时间信息
        h = h + t_emb
        
        # 通过隐藏层
        for layer in self.layers:
            h = layer(h)
            
        # 输出投影
        output = self.output_proj(h)  # [batch_size, input_dim]
        
        # 恢复原始形状
        output = output.view(batch_size, 1, -1)  # [batch_size, 1, input_dim]
        
        return output


class ResidualMLPBlock(nn.Module):
    """
    残差MLP块，用于构建更深的网络
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return x + self.block(x)


class DeepMLPDiffusionModel(nn.Module):
    """
    深层MLP扩散模型，使用残差连接
    """
    def __init__(self, input_dim=1000, hidden_dim=1024, depth=8, time_dim=128, dropout=0.1):
        """
        初始化深层MLP扩散模型
        
        Args:
            input_dim (int): 输入信号的长度
            hidden_dim (int): 隐藏层维度
            depth (int): 残差块数量
            time_dim (int): 时间嵌入的维度
            dropout (float): Dropout概率
        """
        super().__init__()
        
        # 时间嵌入
        self.time_embedding = TimeEmbedding(time_dim)
        
        # 时间嵌入的线性投影
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 输入层
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 残差块
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, dropout) for _ in range(depth)
        ])
        
        # 输出层
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, t):
        """
        前向传播
        
        Args:
            x: 输入噪声图像 [batch_size, 1, input_dim]
            t: 时间步 [batch_size]
            
        Returns:
            预测的噪声 [batch_size, 1, input_dim]
        """
        # 打平输入
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)  # [batch_size, input_dim]
        
        # 获取时间嵌入
        t_emb = self.time_embedding(t)  # [batch_size, time_dim]
        t_emb = self.time_proj(t_emb)  # [batch_size, hidden_dim]
        
        # 输入投影
        h = self.input_proj(x_flat)  # [batch_size, hidden_dim]
        
        # 加入时间信息
        h = h + t_emb
        
        # 通过残差块
        for block in self.blocks:
            h = block(h)
            
        # 输出投影
        output = self.output_proj(h)  # [batch_size, input_dim]
        
        # 恢复原始形状
        output = output.view(batch_size, 1, -1)  # [batch_size, 1, input_dim]
        
        return output


if __name__ == "__main__":
    # 测试模型
    print("===== 测试模型模块 =====")
    
    batch_size = 4
    seq_length = 1000
    
    print(f"创建测试数据，批次大小: {batch_size}, 序列长度: {seq_length}")
    # 创建样本数据
    x = torch.randn(batch_size, 1, seq_length)
    t = torch.randint(0, 1000, (batch_size,))
    
    print("初始化基础MLP模型...")
    # 初始化模型
    model = MLPDiffusionModel(input_dim=seq_length)
    
    print("前向传播...")
    # 前向传播
    output = model(x, t)
    
    print(f"输入形状: {x.shape}")
    print(f"MLP输出形状: {output.shape}")
    
    print("初始化深层MLP模型...")
    deep_model = DeepMLPDiffusionModel(input_dim=seq_length)
    
    print("前向传播...")
    deep_output = deep_model(x, t)
    print(f"深层MLP输出形状: {deep_output.shape}")
    
    # 检查参数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"MLP参数量: {count_parameters(model):,}")
    print(f"深层MLP参数量: {count_parameters(deep_model):,}")
    
    print("模型模块测试成功！")
