import torch

from Model.base import BaseModule
import torch.nn.functional as F

import torch

from Model.base import BaseModule


class InterpolationBlock(BaseModule):
    def __init__(self, scale_factor, mode='nearest',downsample=False):
        super(InterpolationBlock, self).__init__()
        self.downsample = downsample
        self.scale_factor = scale_factor
        self.mode = mode


    def forward(self, x):

        outputs = torch.nn.functional.interpolate(
            x,
            size=x.shape[-1] * self.scale_factor if not self.downsample else x.shape[-1] // self.scale_factor,
            mode=self.mode,
            recompute_scale_factor=False
        )
        return outputs




    # 打印输出数据的形状
    print("Input shape:", input_data.shape)
    print("Output shape:", output_data.shape)
