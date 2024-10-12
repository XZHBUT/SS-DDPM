import torch

from Model.base import BaseModule
from Model.SFLMM import *
from Model.interpolation import InterpolationBlock
from Model.layers import Conv1dWithInitialization


class FeatureWiseAffine(BaseModule):
    def __init__(self):
        super(FeatureWiseAffine, self).__init__()

    def forward(self, x, scale, shift):
        outputs = scale * x + shift
        return outputs


class BasicModulationBlock(BaseModule):

    def __init__(self, n_channels, dilation):
        super(BasicModulationBlock, self).__init__()
        self.featurewise_affine = FeatureWiseAffine()
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.convolution = torch.nn.Conv1d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation
        )

    def forward(self, x, scale, shift):
        outputs = self.featurewise_affine(x, scale, shift)
        outputs = self.leaky_relu(outputs)
        outputs = self.convolution(outputs)
        return outputs

class UBlock(BaseModule):
    def __init__(self, in_channels, out_channels, factor, dilations):
        super(UBlock, self).__init__()
        self.first_block_main_branch = torch.nn.ModuleDict({
            'upsampling': torch.nn.Sequential(*[
                torch.nn.LeakyReLU(0.2),
                InterpolationBlock(
                    scale_factor=factor,
                    mode='linear',
                ),
                torch.nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=dilations[0],
                    dilation=dilations[0]
                )
            ]),
            'modulation': BasicModulationBlock(
                out_channels, dilation=dilations[1]
            )
        })
        self.first_block_residual_branch = torch.nn.Sequential(*[
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1
            ),
            InterpolationBlock(
                scale_factor=factor,
                mode='linear',
            )
        ])
        self.second_block_main_branch = torch.nn.ModuleDict({
            f'modulation_{idx}': BasicModulationBlock(
                out_channels, dilation=dilations[2 + idx]
            ) for idx in range(2)
        })

    def forward(self, x, scale, shift):
        # First upsampling residual block
        outputs = self.first_block_main_branch['upsampling'](x)
        outputs = self.first_block_main_branch['modulation'](outputs, scale, shift)
        outputs = outputs + self.first_block_residual_branch(x)

        # Second residual block
        residual = self.second_block_main_branch['modulation_0'](outputs, scale, shift)
        outputs = outputs + self.second_block_main_branch['modulation_1'](residual, scale, shift)
        return outputs




    input_data = torch.randn((10, 32, 256)).to(device)
    UP = UBlock(in_channels=32, out_channels=64, factor=2, dilations=[1, 2, 1, 2]).to(device)
    print(UP(input_data, s, b).shape)
