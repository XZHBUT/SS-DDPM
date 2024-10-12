import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()  # 输出真假概率
        )

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 512),  # 噪声维度到隐层
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),  # 输出维度为 2048
            nn.Tanh()  # 生成数据归一化到 [-1, 1]
        )

    def forward(self, z):
        return self.model(z)

if __name__ == '__main__':

    # 示例用法
    input_dim = 2048
    discriminator = Discriminator(input_dim)

    # 输入真实或生成的数据，判别真假
    data = torch.randn(32, input_dim)  # 假设输入的数据形状为 (32, 2048)
    real_or_fake = discriminator(data)  # 输出的形状是 (32, 1)
