import torch
from torch import nn

from Model import  Generator, Discriminator
from pathlib import Path

from tool.ChioseDataSample import ChoiseDataCWRU
import numpy as np
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 训练循环
    epochs = 100  # 假设我们训练 10000 个 epoch
    batch_size = 32
    noise_dim = 100  # 噪声的维度
    real_label = 1
    fake_label = 0



    # 示例用法
    noise_dim = 100  # 输入噪声的维度
    output_dim = 2048  # 生成数据的维度
    generator = Generator(noise_dim, output_dim).to(device)

    # 示例用法
    input_dim = 2048
    discriminator = Discriminator(input_dim).to(device)
    # 损失函数
    criterion = nn.BCELoss()
    # 优化器
    lr = 0.0002
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
    datapath = f'../../data/ChoiseData/12k_Drive_End_B007_3_121.mat'
    folder = Path(datapath)
    data = ChoiseDataCWRU(datapath, 100, 2048)
    data = np.array(data)  # 将列表转换为单一的 numpy.ndarray
    dataset = torch.Tensor(data).float().to(device)
    # dataset = torch.unsqueeze(dataset, dim=1)
    dataloader = torch.utils.data.DataLoader(dataset.to(device), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        # === 训练判别器 ===
        # 获取真实数据
        for inputs in dataloader:
            real_data = inputs  # 从你的数据集中获取 (B, 2048) 的真实数据
            batch_size = real_data.size(0)

            # 创建真实标签和假标签
            real_labels = torch.full((batch_size, 1), real_label, dtype=torch.float).to(device)
            fake_labels = torch.full((batch_size, 1), fake_label, dtype=torch.float).to(device)

            # 生成假数据
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_data = generator(noise)

            # 判别器在真实数据上的损失
            real_output = discriminator(real_data)
            real_loss = criterion(real_output, real_labels)

            # 判别器在生成数据上的损失
            fake_output = discriminator(fake_data.detach())
            fake_loss = criterion(fake_output, fake_labels)

            # 总判别器损失
            discriminator_loss = real_loss + fake_loss

            # 优化判别器
            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # === 训练生成器 ===
            # 我们希望生成器生成的假数据能够欺骗判别器
            fake_output = discriminator(fake_data)
            generator_loss = criterion(fake_output, real_labels)

            # 优化生成器
            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

        print(f"D Loss: {discriminator_loss.item()} | G Loss: {generator_loss.item()}")

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] | D Loss: {discriminator_loss.item()} | G Loss: {generator_loss.item()}")
