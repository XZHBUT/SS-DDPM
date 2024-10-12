
from torch import nn, optim

import torch

from Model.base import BaseModule
from Model.interpolation import InterpolationBlock
from Model.layers import Conv1dWithInitialization


class ConvolutionBlock(BaseModule):
    def __init__(self, in_channels, out_channels, dilation):
        # 该模块只会改变通道，不会改变长度
        super(ConvolutionBlock, self).__init__()
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.convolution = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation
        )

    def forward(self, x):
        # print('x', x.device)
        outputs = self.leaky_relu(x)
        # print('s', outputs.device)
        outputs = self.convolution(outputs)
        # print('s', outputs.device)
        # print(outputs.shape)
        return outputs


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, factor, dilations):
        '''
        :param in_channels:
        :param out_channels:
        :param factor: 缩放因子
        :param dilations:  卷积的膨胀尺度，不改变输出长度，影响卷积核的尺度
        '''
        super(DBlock, self).__init__()

        in_sizes = [in_channels] + [out_channels for _ in range(len(dilations) - 1)]
        # 第一个是in_sizes，后面都是out_channels
        out_sizes = [out_channels for _ in range(len(in_sizes))]
        # 都是out_sizes

        ConvList = []
        for in_size, out_size, dilation in zip(in_sizes, out_sizes, dilations):
            ConvList.append(ConvolutionBlock(in_size, out_size, dilation))
        self.main_branch = torch.nn.Sequential(
            *([
                InterpolationBlock(
                    scale_factor=factor,
                    mode='linear',
                    downsample=True)
             ] + ConvList)
        )
        self.residual_branch = torch.nn.Sequential(*[
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1
            ),
            InterpolationBlock(
                scale_factor=factor,
                mode='linear',
                downsample=True
            )
        ])

    def forward(self, x):
        outputs = self.main_branch(x)

        outputs = outputs + self.residual_branch(x)
        return outputs


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DBlockTest = DBlock(in_channels=1, out_channels=32, factor=1, dilations=[1, 2, 4]).to(device)

    input_data = torch.randn((10, 1, 1024)).to(device)

    output_data = DBlockTest(input_data)

    # 打印输出数据的形状
    print("Input shape:", input_data.shape)
    print("Output shape:", output_data.shape)
