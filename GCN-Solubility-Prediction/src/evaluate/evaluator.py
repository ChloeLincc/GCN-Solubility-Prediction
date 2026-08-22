"""
模型评估模块
"""
import torch
import numpy as np
import pandas as pd


class Evaluator:
    """
    评估器类：负责在测试集上评估模型
    """
    def __init__(self, model, test_loader, loss_fn, device):
        self.model = model
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.device = device
    
    def evaluate(self):
        """
        在测试集上评估模型
        
        Returns:
            metrics: 包含 MSE、RMSE 的字典
            all_preds: 所有预测值
            all_trues: 所有真实值
        """
        self.model.eval()
        total_test_loss = 0.0
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                pred, _ = self.model(batch.x.float(), batch.edge_index, batch.batch)
                loss = self.loss_fn(pred, batch.y)
                total_test_loss += loss.item() * batch.num_graphs
                
                all_preds.append(pred.cpu())
                all_trues.append(batch.y.cpu())
        
        avg_test_loss = total_test_loss / len(self.test_loader.dataset)
        
        # 计算 RMSE
        all_preds = torch.cat(all_preds, dim=0).numpy().flatten()
        all_trues = torch.cat(all_trues, dim=0).numpy().flatten()
        rmse = np.sqrt(np.mean((all_preds - all_trues) ** 2))
        
        metrics = {
            "mse": avg_test_loss,
            "rmse": rmse
        }
        
        print("=" * 40)
        print("【测试/预测阶段】")
        print("=" * 40)
        print(f"测试集平均损失 (MSE): {avg_test_loss:.6f}")
        print(f"测试集 RMSE: {rmse:.4f}")
        print("=" * 40)
        
        return metrics, all_preds, all_trues
    
    def inspect_batch(self, num_samples=10):
        """
        抽查一个批次，展示预测值 vs 真实值
        
        Args:
            num_samples: 展示的样本数量
        """
        self.model.eval()
        test_batch = next(iter(self.test_loader))
        
        with torch.no_grad():
            test_batch = test_batch.to(self.device)
            pred, _ = self.model(test_batch.x.float(), test_batch.edge_index, test_batch.batch)
            
            df = pd.DataFrame()
            df["y_real"] = test_batch.y.cpu().tolist()
            df["y_pred"] = pred.cpu().tolist()
            df["y_real"] = df["y_real"].apply(lambda row: row[0])
            df["y_pred"] = df["y_pred"].apply(lambda row: row[0])
        
        print("\n--- 第一个测试 batch 的预测 vs 真实值（前 10 个样本）---")
        print(df.head(num_samples))
        return df