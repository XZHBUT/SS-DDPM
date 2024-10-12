import torch
from torch import nn

from Model.FiLM import FeatureWiseLinearModulation

from Model.DBlock import DBlock

from Model.UBlock import UBlock


class Encoder(nn.Module):
    def __init__(self, n_Steps=1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.EncoderBlock1 = DBlock(in_channels=1, out_channels=32, factor=1, dilations=[1, 2, 4])
        self.FiLM1 = FeatureWiseLinearModulation(in_channels=32,
                                                 out_channels=32,
                                                 N_Steps=n_Steps,
                                                 EmbeddingL=2048)

        self.EncoderBlock2 = DBlock(in_channels=32, out_channels=64, factor=2, dilations=[1, 2, 4])
        self.FiLM2 = FeatureWiseLinearModulation(in_channels=64,
                                                 out_channels=64,
                                                 N_Steps=n_Steps,
                                                 EmbeddingL=1024)

        self.EncoderBlock3 = DBlock(in_channels=64, out_channels=128, factor=2, dilations=[1, 2, 4])
        self.FiLM3 = FeatureWiseLinearModulation(in_channels=128,
                                                 out_channels=128,
                                                 N_Steps=n_Steps,
                                                 EmbeddingL=512)

        self.EncoderBlock4 = DBlock(in_channels=128, out_channels=256, factor=2, dilations=[1, 2, 4])
        self.FiLM4 = FeatureWiseLinearModulation(in_channels=256,
                                                 out_channels=256,
                                                 N_Steps=n_Steps,
                                                 EmbeddingL=256)

    def forward(self, x, t):
        red = []
        x1 = self.EncoderBlock1(x)
        s, b = self.FiLM1(x1, t)
        red.append([s, b])

        x2 = self.EncoderBlock2(x1)
        s, b = self.FiLM2(x2, t)
        red.append([s, b])

        x3 = self.EncoderBlock3(x2)
        s, b = self.FiLM3(x3, t)
        red.append([s, b])

        x4 = self.EncoderBlock4(x3)
        s, b = self.FiLM4(x4, t)
        red.append([s, b])

        return x4, red


class Decoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.DecoderBlock1 = UBlock(in_channels=256, out_channels=256, factor=2, dilations=[1, 2, 4, 8])

        self.DecoderBlock2 = UBlock(in_channels=256, out_channels=128, factor=2, dilations=[1, 2, 4, 8])

        self.DecoderBlock3 = UBlock(in_channels=128, out_channels=64, factor=2, dilations=[1, 2, 4, 8])

        self.DecoderBlock4 = UBlock(in_channels=64, out_channels=32, factor=2, dilations=[1, 2, 4, 8])

    def forward(self, x, red):
        x1 = self.DecoderBlock1(x, red[0][0], red[0][1])

        x2 = self.DecoderBlock2(x1, red[1][0], red[1][1])

        x3 = self.DecoderBlock3(x2, red[2][0], red[2][1])

        x4 = self.DecoderBlock4(x3, red[3][0], red[3][1])
        return x4


class FiLM_UNet(nn.Module):
    def __init__(self, n_Steps=1000, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Encoder = Encoder(n_Steps=n_Steps)

        self.Mid = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.Decoder = Decoder()

        self.Lastout = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, x, t):
        Enout, red = self.Encoder(x, t)

        Midout = self.Mid(Enout)

        out = self.Decoder(Midout, red[::-1])

        return self.Lastout(out)


if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # a = FiLM_UNet().to(device)
    # input_data = torch.randn((64, 1, 1024)).to(device)
    # # 对一个batch生成随机覆盖更多得t
    # device = input_data.device
    # batch_size = input_data.shape[0]
    # t = torch.randint(0, 1000, (batch_size // 2,)).to(device)
    # t = torch.cat([t, 1000 - 1 - t], dim=0).to(device)
    # # print(t.device)
    # # print(input_data.device)
    #
    # output_data = a(input_data, t)
    #
    # # 打印输出数据的形状
    # print("Input shape:", input_data.shape)
    # print("Output shape:", output_data.shape)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = FiLM_UNet(n_Steps=1000).to(device)
    input_data = torch.randn((64, 1, 2048)).to(device)
    batch_size = input_data.shape[0]
    t = torch.randint(0, 1000, (batch_size // 2,)).to(device)
    t = torch.cat([t, 1000 - 1 - t], dim=0).to(device)
    output_data = a(input_data, t)
    # 打印输出数据的形状
    print("Input shape:", input_data.shape)
    print("Output shape:", output_data.shape)
