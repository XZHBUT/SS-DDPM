import numpy as np
import torch

from tool.ChioseDataSample import ChoiseDataCWRU
from tool.DataProcess import read_OrginCWRU
from tool.MixDataset import read_Create_CWRU, read_csv_file


def gaussian_kernel(a, b, kernel_bandwidth):
    sq_dist = torch.cdist(a, b)**2
    return torch.exp(-sq_dist / (2 * kernel_bandwidth ** 2))

def mmd(x, y, kernel_bandwidth=1.0):
    x_kernel = gaussian_kernel(x, x, kernel_bandwidth)
    y_kernel = gaussian_kernel(y, y, kernel_bandwidth)
    cross_kernel = gaussian_kernel(x, y, kernel_bandwidth)
    return x_kernel.mean() + y_kernel.mean() - 2 * cross_kernel.mean()
def fft_transform(tensor):
    # 执行FFT
    fft_result = torch.fft.fft(tensor)
    # 只取实数部分
    real_part = fft_result.real
    # 由于FFT结果是对称的，只保留前一半即可
    return real_part[:, :1024]  # 假设tensor的形状是 (B, 2048)

if __name__ == '__main__':
    # 示例使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X = torch.zeros(100, 2048)  # 假设 X 数据集有 100 个样本，每个样本 2048 维
    Y = torch.randn(120, 2048)  # 假设 Y 数据集有 120 个样本，每个样本 2048 维

    # 计算 MMD 距离
    mmd_distance = mmd(X, Y, kernel_bandwidth=1.0)
    print(f"MMD 距离: {mmd_distance.item()}")

    data_or_0 = ChoiseDataCWRU('../data/CWRU/0HP/normal_0_97.mat', 60, 2048)
    data_or_0 = np.array(data_or_0)
    dataset_or_0 = torch.Tensor(data_or_0).float().to(device)
    data_or_1 = ChoiseDataCWRU('../data/CWRU/1HP/normal_1_98.mat', 60, 2048)
    data_or_1 = np.array(data_or_1)
    dataset_or_1 = torch.Tensor(data_or_1).float().to(device)
    data_or_2 = ChoiseDataCWRU('../data/CWRU/2HP/normal_2_99.mat', 60, 2048)
    data_or_2 = np.array(data_or_2)
    dataset_or_2 = torch.Tensor(data_or_2).float().to(device)
    data_or_3 = ChoiseDataCWRU('../data/CWRU/3HP/normal_3_100.mat', 60, 2048)
    data_or_3 = np.array(data_or_3)
    dataset_or_3 = torch.Tensor(data_or_3).float().to(device)
    # dataset = torch.unsqueeze(dataset, dim=1)
    # print(dataset_or.shape)
    data_or = torch.cat([dataset_or_0, dataset_or_1, dataset_or_2, dataset_or_3], dim=0)

    print(data_or.shape)
    data_creat_0 = read_csv_file(filepath='../data/CreateCWRU/0HP/Create_normal_0_97.csv')
    dataset_creat_0 = torch.Tensor(data_creat_0).float().to(device)
    data_creat_1 = read_csv_file(filepath='../data/CreateCWRU/1HP/Create_normal_1_98.csv')
    dataset_creat_1 = torch.Tensor(data_creat_1).float().to(device)
    data_creat_2 = read_csv_file(filepath='../data/CreateCWRU/2HP/Create_normal_2_99.csv')
    dataset_creat_2 = torch.Tensor(data_creat_2).float().to(device)
    data_creat_3 = read_csv_file(filepath='../data/CreateCWRU/3HP/Create_normal_3_100.csv')
    dataset_creat_3 = torch.Tensor(data_creat_3).float().to(device)
    data_cr = torch.cat([dataset_creat_0[:100], dataset_creat_1[:100], dataset_creat_2[:100], dataset_creat_3[:100]], dim=0)
    mmd_distance = mmd(fft_transform(data_or), fft_transform(data_cr), kernel_bandwidth=100)
    print(f"MMD 距离: {mmd_distance.item()}")