"""
图卷积神经网络（GCN）模型定义
"""
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class GCN(torch.nn.Module):
    """
    4层图卷积网络（GCN）
    用于分子溶解度预测（回归任务）
    """
    def __init__(self, in_channels, embedding_size=64):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        
        # 4层图卷积
        self.initial_conv = GCNConv(in_channels, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        
        # 输出层：回归任务输出1维
        self.out = Linear(embedding_size * 2, 1)
    
    def forward(self, x, edge_index, batch_index):
        """
        前向传播
        
        Args:
            x: 节点特征矩阵 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            batch_index: 批次索引 [num_nodes]
        
        Returns:
            out: 预测值 [batch_size, 1]
            hidden: 池化后的特征向量 [batch_size, embedding_size*2]
        """
        # 第1层
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)
        
        # 第2~4层
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden)
        
        # 全局池化拼接：最大池化 + 平均池化
        hidden = torch.cat([
            global_max_pool(hidden, batch_index),
            global_mean_pool(hidden, batch_index)
        ], dim=1)
        
        # 输出层
        out = self.out(hidden)
        return out, hidden