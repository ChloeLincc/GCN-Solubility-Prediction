"""
可视化模块
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_results(losses, all_preds, all_trues, rmse):
    """
    绘制训练损失曲线和预测散点图

    Args:
        losses: 训练损失历史
        all_preds: 所有预测值
        all_trues: 所有真实值
        rmse: RMSE值
    """
    sns.set_style("whitegrid")

    plt.figure(figsize=(14, 5))

    # 左图：训练损失曲线
    plt.subplot(1, 2, 1)
    losses_float = [float(l) for l in losses]
    sns.lineplot(x=range(len(losses_float)), y=losses_float, color="blue")
    plt.xlabel("训练轮次 (Epoch)")
    plt.ylabel("平均训练损失 (MSE)")
    plt.title("模型训练收敛曲线")
    plt.grid(True, alpha=0.3)

    # 右图：预测散点图
    plt.subplot(1, 2, 2)
    df_test = pd.DataFrame({
        "y_real": all_trues,
        "y_pred": all_preds
    })
    sns.scatterplot(data=df_test, x="y_real", y="y_pred", alpha=0.6, color="green")
    plt.plot([-7, 2], [-7, 2], 'r--', linewidth=2, label="完美预测 (y=x)")
    plt.xlim(-7, 2)
    plt.ylim(-7, 2)
    plt.xlabel("真实溶解度 (LogS)")
    plt.ylabel("预测溶解度 (LogS)")
    plt.title(f"测试集预测效果 (RMSE: {rmse:.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    print("可视化图表已展示。")