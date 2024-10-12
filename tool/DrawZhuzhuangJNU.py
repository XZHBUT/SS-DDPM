import matplotlib.pyplot as plt
import numpy as np

# 模型名称
models = ['None', 'SS-DDPM', 'TS-GAN', 'DDPM', 'VAE']

# X轴刻度标签
x_labels = ['20:1', '10:1', '5:1', '2:1']

# 每个模型在不同比例下的数据
data = np.array([
    [38.5, 44.35, 55, 68.5],   # None
    [65.56, 73.63, 79.31, 83.3],   # SS-DDPM
    [58.23, 69.26, 75.25, 80.38],  # TS-GAN
    [60.58, 68.25, 75.62, 77.21],  # DDPM
    [40.21, 53.48, 64.75, 73.56]   # VAE
])

# 设置柱状图参数
bar_width = 0.12  # 每个柱状条的宽度（调整间距）
x = np.arange(len(x_labels))  # X轴位置

# 创建画布
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制柱状图并为每组数据增加间距
for i in range(len(models)):
    ax.bar(x + i * (bar_width + 0.02), data[i], width=bar_width, label=models[i])

# 设置X轴的刻度标签
ax.set_xticks(x + (bar_width + 0.02) * 2)  # 调整X轴刻度，使它们位于中间
ax.set_xticklabels(x_labels, fontsize=12)  # 设置X轴刻度字体大小

# 设置Y轴范围从40开始，并设置Y轴刻度字体大小
ax.set_ylim(30, 90)
ax.set_yticklabels(np.arange(30, 91, 10), fontsize=12)  # 设置Y轴刻度

# 设置图例并调整字体大小
ax.legend(fontsize=16)

# 设置X轴和Y轴标签字体大小
ax.set_xlabel('Ratio of normal to abnormal working condition samples', fontsize=14)
ax.set_ylabel('Accuracy(%)', fontsize=14)  # 将Y轴标签改为 "ACC"


# 显示图形
plt.tight_layout()
plt.show()
