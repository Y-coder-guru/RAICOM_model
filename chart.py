import pandas as pd
import matplotlib.pyplot as plt
import os

# 请修改为你的 results.csv 实际路径
csv_file = r"D:\y_rtxq\RAICOM_yolov8\runs\detect\train2\results.csv"

# 读取数据
df = pd.read_csv(csv_file)
# 删除第一列（通常是 epoch 序号，已经包含在 index 里）
df = df.drop(columns=df.columns[0], errors='ignore')
# 去掉列名的首尾空格
df.columns = df.columns.str.strip()

# 准备画图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('YOLOv8 训练曲线', fontsize=16)

# ---- 第一张图：Loss 曲线 ----
loss_cols = [col for col in df.columns if 'loss' in col.lower()]
if loss_cols:
    for col in loss_cols:
        axes[0].plot(df[col], label=col)
    axes[0].set_title('Loss 曲线')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
else:
    axes[0].text(0.5, 0.5, '未找到 Loss 列', ha='center', va='center')
    axes[0].set_title('Loss 曲线')

# ---- 第二张图：评估指标曲线 ----
# 常见指标列名：mAP50, mAP50-95, precision, recall
metric_cols = [col for col in df.columns if any(x in col.lower() for x in ['map', 'precision', 'recall'])]
if metric_cols:
    for col in metric_cols:
        axes[1].plot(df[col], label=col)
    axes[1].set_title('评估指标')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('数值')
    axes[1].legend()
    axes[1].grid(True)
else:
    axes[1].text(0.5, 0.5, '未找到 mAP/Precision/Recall 列', ha='center', va='center')
    axes[1].set_title('评估指标')

plt.tight_layout()
# 保存图片到 train2 目录下
output_path = os.path.join(os.path.dirname(csv_file), 'training_curves.png')
plt.savefig(output_path, dpi=150)
plt.show()
print(f"图表已保存至：{output_path}")