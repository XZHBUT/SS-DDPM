import matplotlib.pyplot as plt
import numpy as np

# 定义每组数据
groups = [
    {"x": [20], "y": [0], "heights": [37.5]},
    {"x": [20, 40], "y": [20, 0], "heights": [50.35, 48.75]},
    {"x": [20, 40, 80], "y": [60, 40, 0], "heights": [63.75, 64.75, 64.75]},
    {"x": [20, 40, 80, 160], "y": [140, 120, 80, 0], "heights": [82.31, 83.19, 82.67, 82.62]},
    {"x": [40, 80, 160], "y": [280, 240, 160], "heights": [87.75, 87.75, 88.28]},
    {"x": [80, 160], "y": [560, 480], "heights": [95.32, 95.22]},
]

# 设置柱状图的相关属性
fig, ax = plt.subplots(figsize=(12, 8))

# 用于存储X轴的位置，每个组之间添加间距
x_pos = 0
group_width = 0.5  # 调整间距

# 保存每组的中心位置，用于设置x轴刻度
group_centers = []

# 循环遍历每组数据
for group_index, group in enumerate(groups):
    x_values = group["x"]
    y_values = group["y"]
    heights = group["heights"]

    # 计算当前组柱子在X轴上的位置
    base_x_pos = x_pos + group_index * group_width * 2  # 每个组之间的间隔

    # 保存组中心位置
    group_centers.append(base_x_pos + (len(x_values) - 1) / 2)

    for i in range(len(x_values)):
        # 绘制柱子
        ax.bar(base_x_pos + i, heights[i], width=0.8, label=f"{x_values[i]}+{y_values[i]}")

        # 在每个柱子的顶部显示 X+Y 的值，设置字体大小和斜体
        ax.text(base_x_pos + i, heights[i] + 1, f"{x_values[i]}+{y_values[i]}", ha='center', va='bottom',
                rotation=45, fontsize=16, fontstyle='italic')

    # 绘制虚线y=a, a为该组的最大值，虚线长度仅到达最后一根柱子
    max_height = max(heights)
    last_bar_x_pos = base_x_pos + len(x_values) - 1  # 最后一根柱子的X位置
    ax.plot([0, last_bar_x_pos + 0.4], [max_height, max_height], color='gray', linestyle='--')

    x_pos += len(x_values)

# 设置X轴刻度，只显示1到6，刻度位于每组数据的中间
ax.set_xticks(group_centers)
ax.set_xticklabels([str(i) for i in range(1, 7)], fontsize=14)

# 设置标题和标签，调整字体大小
ax.set_xlabel('Data Groups(Real samples + Generated samples)', fontsize=14)
ax.set_ylabel('Accuracy (%)', fontsize=16)

# 调整y轴刻度字体大小
ax.tick_params(axis='y', labelsize=16)

# 取消上边框
ax.spines['top'].set_visible(False)

# 显示图表
plt.tight_layout()
plt.show()
