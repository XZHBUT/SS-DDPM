
from torch import nn, optim

import torch

from Model.base import BaseModule
from Model.interpolation import InterpolationBlock
from Model.layers import Conv1dWithInitialization


class ConvolutionBlock(BaseModule):
    def __init__(self, in_channels, out_channels, dilation):
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
        super(DBlock, self).__init__()

        in_sizes = [in_channels] + [out_channels for _ in range(len(dilations) - 1)]

        out_sizes = [out_channels for _ in range(len(in_sizes))]


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



    # 打印输出数据的形状
    print("Input shape:", input_data.shape)
    print("Output shape:", output_data.shape)
