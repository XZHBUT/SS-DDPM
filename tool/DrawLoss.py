from matplotlib import pyplot as plt


def DrawLoss(num_epoch, trainLoss, loss_filepath):
    # 将训练损失从 Tensor 转换为 NumPy
    trainLoss_np = [tensor.cpu().detach().numpy() for tensor in trainLoss]

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))

    # 使用红色加粗线
    plt.plot(range(1, num_epoch + 1), trainLoss_np, label='Training Loss', color='red', linewidth=2.5)

    # 设置刻度字体大小
    plt.xlabel('Epochs', fontsize=16)  # 设置x轴标签字体大小
    plt.ylabel('Loss', fontsize=16)  # 设置y轴标签字体大小
    plt.xticks(fontsize=16)  # 设置x轴刻度字体大小
    plt.yticks(fontsize=16)  # 设置y轴刻度字体大小

    # 显示图例
    plt.legend(fontsize=18)

    # 保存图像
    plt.savefig(loss_filepath)