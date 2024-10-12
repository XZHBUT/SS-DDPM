import math

import torch
from torch import nn

from Model.DBlock import DBlock
from Model.base import BaseModule
from Model.layers import Conv1dWithInitialization







class FeatureWiseLinearModulation(BaseModule):
    def __init__(self, in_channels, out_channels, N_Steps, EmbeddingL):
        super(FeatureWiseLinearModulation, self).__init__()
        self.signal_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.LeakyReLU(0.2)
        )
        # 位置编码
        self.emb1 = nn.Embedding(N_Steps, EmbeddingL)


        self.scale_conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.shift_conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x, t):
        # print(x.shape)

        outputs = self.signal_conv(x)
        # print(outputs.shape)

        outputs = outputs + torch.unsqueeze(self.emb1(t), dim=1).to(x.device)
        scale, shift = self.scale_conv(outputs), self.shift_conv(outputs)
        # print(scale.shape) torch.Size([10, 64, 512])
        # print(shift.shape) torch.Size([10, 64, 512])
        return scale, shift

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    DBlockTest = DBlock(in_channels=1, out_channels=64, factor=2, dilations=[1, 2, 4]).to(device)
    input_data = torch.randn((10, 1, 1024)).to(device)
    output_data = DBlockTest(input_data)

    # 打印输出数据的形状
    print("Input shape:", input_data.shape)
    print("Output shape:", output_data.shape)

    # 对一个batch生成随机覆盖更多得t
    device = output_data.device
    batch_size = output_data.shape[0]
    t = torch.randint(0, 1000, (batch_size // 2,))
    t = torch.cat([t, 1000 - 1 - t], dim=0).to(device)
    print(t.device)

    FILM = FeatureWiseLinearModulation(in_channels=64, out_channels=64, N_Steps=1000, EmbeddingL=512).to(device)
    output_data = FILM(output_data, t)