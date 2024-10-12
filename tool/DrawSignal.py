import matplotlib.pyplot as plt
import torch



def plot_signal(signal_tensor):
    """
    绘制形状为 (1, 2048) 的信号张量图。

    参数:
    signal_tensor (torch.Tensor): 形状为 (1, 2048) 的信号张量
    """
    # 检查输入是否为 PyTorch 张量，并且维度为 (1, 2048)
    if isinstance(signal_tensor, torch.Tensor) and signal_tensor.shape == (1, 2048):
        # 转换为 NumPy 数组以用于绘图
        signal_np = signal_tensor.numpy().flatten()
        # 绘制信号图
        plt.figure(figsize=(10, 4))
        plt.plot(signal_np, color='coral')
        plt.title('Signal Plot')
        plt.xlabel('Time Step')
        plt.ylabel('Amplitude')
        plt.grid(False)
        plt.show()
    else:
        raise ValueError("输入应为形状为 (1, 2048) 的 PyTorch 张量")


if __name__ == '__main__':
    example_tensor = torch.randn(1, 2048)  # 创建一个随机信号张量
    plot_signal(example_tensor)