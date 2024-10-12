import math

import torch
from torch import nn

from Model.DBlock import DBlock
from Model.base import BaseModule
from Model.layers import Conv1dWithInitialization







class SiganlFeatureWiseLinearModulation(BaseModule):
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
