"""
项目运行入口脚本
"""
import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from src import (
    check_environment,
    get_device,
    load_esol_dataset,
    GCN,
    Trainer,
    Evaluator,
    get_dataset_info
)
from src.utils.visualize import plot_results
import warnings
warnings.filterwarnings("ignore")


def main():
    """主函数"""
    # 1. 环境检查
    check_environment()
    
    # 2. 加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 3. 加载数据
    train_loader, test_loader, dataset = load_esol_dataset(
        root=".",
        train_ratio=config['train_ratio'],
        batch_size=config['batch_size']
    )
    
    # 4. 创建模型
    print("【定义模型结构】")
    model = GCN(
        in_channels=dataset.num_node_features,
        embedding_size=config['embedding_size']
    )
    print(model)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    print("=" * 40)
    
    # 5. 训练准备
    print("【训练准备】")
    device = get_device()
    print(f"当前运行设备: {device}")
    model = model.to(device)
    
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 6. 训练
    trainer = Trainer(model, train_loader, loss_fn, optimizer, device)
    losses = trainer.train(
        num_epochs=config['num_epochs'],
        print_every=100
    )
    
    # 7. 评估
    evaluator = Evaluator(model, test_loader, loss_fn, device)
    metrics, all_preds, all_trues = evaluator.evaluate()
    evaluator.inspect_batch(num_samples=10)
    
    # 8. 可视化
    print("【生成可视化图表】")
    plot_results(losses, all_preds, all_trues, metrics['rmse'])
    
    print("=" * 40)
    print("🎉 全流程完成！")
    print("=" * 40)


if __name__ == "__main__":
    main()