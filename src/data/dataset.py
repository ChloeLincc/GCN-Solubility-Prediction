"""
数据加载模块
负责从 MoleculeNet 下载 ESOL 数据集，并划分为训练集和测试集
"""
import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import DataLoader


def load_esol_dataset(root=".", train_ratio=0.8, batch_size=64):
    """
    加载 ESOL 数据集，并返回训练和测试的 DataLoader

    Args:
        root: 数据存储路径
        train_ratio: 训练集比例
        batch_size: 批次大小

    Returns:
        train_loader, test_loader, dataset
    """
    print("正在加载 ESOL 数据集...")
    dataset = MoleculeNet(root=root, name="ESOL")
    print(f"数据加载成功！共 {len(dataset)} 个样本")
    print(f"节点特征维度: {dataset.num_node_features}")

    # 划分数据集
    data_size = len(dataset)
    train_dataset = dataset[:int(data_size * train_ratio)]
    test_dataset = dataset[int(data_size * train_ratio):]

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    return train_loader, test_loader, dataset


def get_dataset_info(dataset):
    """获取数据集的基本信息"""
    return {
        "num_samples": len(dataset),
        "num_node_features": dataset.num_node_features,
        "num_edge_features": dataset.num_edge_features,
    }