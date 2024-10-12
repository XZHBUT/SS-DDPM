import torch

from Model.base import BaseModule
import torch.nn.functional as F

import torch

from Model.base import BaseModule


class InterpolationBlock(BaseModule):
    def __init__(self, scale_factor, mode='nearest',downsample=False):
        """
        :param scale_factor: 一个缩放因子，用于调整输入数据的大小
        :param mode: 插值模式。默认为 'linear'，表示线性插值。其他可能的值包括 'nearest'、'bilinear'、'bicubic' 等。
        :param downsample: 设置为 True，则用于指定下采样因子；否则，用于上采样。
        """
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


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 创建InterpolationBlock实例
    interpolation_block = InterpolationBlock(scale_factor=2, mode='linear', downsample=False).to(device)

    # 生成输入数据（这里假设输入是一个4D张量，比如(batch_size, channels, height, width)）
    input_data = torch.randn((1, 3, 64)).to(device)

    # 进行插值操作
    output_data = interpolation_block(input_data)

    # 打印输出数据的形状
    print("Input shape:", input_data.shape)
    print("Output shape:", output_data.shape)
