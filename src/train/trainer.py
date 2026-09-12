"""
模型训练模块
"""
import torch
import warnings
warnings.filterwarnings("ignore")


class Trainer:
    """
    训练器类：封装训练循环
    """
    def __init__(self, model, train_loader, loss_fn, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.losses = []  # 记录每个epoch的平均损失
    
    def train_epoch(self):
        """
        训练一个epoch
        
        Returns:
            avg_loss: 该epoch的平均损失
            last_batch_loss: 最后一个批次的损失（用于波动监测）
        """
        self.model.train()
        total_loss = 0.0
        last_batch_loss = 0.0
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            pred, _ = self.model(batch.x.float(), batch.edge_index, batch.batch)
            loss = self.loss_fn(pred, batch.y)
            loss.backward()
            self.optimizer.step()
            
            last_batch_loss = loss.item()
            total_loss += loss.item() * batch.num_graphs
        
        avg_loss = total_loss / len(self.train_loader.dataset)
        return avg_loss, last_batch_loss
    
    def train(self, num_epochs, print_every=100):
        """
        完整训练流程
        
        Args:
            num_epochs: 训练轮数
            print_every: 每隔多少轮打印一次损失
        """
        print("=" * 40)
        print("【开始训练】")
        print(f"训练 {num_epochs} 个 Epoch，每 {print_every} 步打印一次...")
        print("=" * 40)
        
        self.losses = []
        for epoch in range(num_epochs):
            avg_loss, batch_loss = self.train_epoch()
            self.losses.append(avg_loss)
            
            if epoch % print_every == 0:
                print(f"Epoch {epoch:4d} | 平均损失(趋势): {avg_loss:.6f} | 最后批次损失(波动): {batch_loss:.6f}")
        
        print("=" * 40)
        print("训练完成！")
        print(f"最终平均训练损失: {self.losses[-1]:.6f}")
        print("=" * 40)
        return self.losses