import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out_r = self.relu(self.conv2(self.relu(self.conv1(x))))
        out = self.maxpool(out_r)

        return out, out_r


class Encoder(nn.Module):
    def __init__(self, in_channels, n_Steps=1000,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.EncoderBlock1 = EncoderBlock(in_channels, 32)  # 1024 - 512
        self.EncoderBlock2 = EncoderBlock(32, 32)  # 512 - 256
        self.EncoderBlock3 = EncoderBlock(32, 64)  # 256 - 128
        self.EncoderBlock4 = EncoderBlock(64, 64)  # 128 - 64
        self.EncoderBlock5 = EncoderBlock(64, 128)  # 64 - 32

        self.emb1 = nn.Embedding(n_Steps, 512)
        self.emb2 = nn.Embedding(n_Steps, 256)
        self.emb3 = nn.Embedding(n_Steps, 128)
        self.emb4 = nn.Embedding(n_Steps, 64)
        self.emb5 = nn.Embedding(n_Steps, 32)

    def forward(self, x, t):
        out1, red1 = self.EncoderBlock1(x)
        # print(out1.shape)
        # print(self.emb1(t).shape)
        out1 = out1 + torch.unsqueeze(self.emb1(t), dim=1)
        out2, red2 = self.EncoderBlock2(out1)
        out2 = out2 + torch.unsqueeze(self.emb2(t), dim=1)
        out3, red3 = self.EncoderBlock3(out2)
        out3 = out3 + torch.unsqueeze(self.emb3(t), dim=1)
        out4, red4 = self.EncoderBlock4(out3)
        out4 = out4 + torch.unsqueeze(self.emb4(t), dim=1)
        out5, red5 = self.EncoderBlock5(out4)
        out5 = out5 + torch.unsqueeze(self.emb5(t), dim=1)
        red = [red1, red2, red3, red4, red5]
        return out5, red


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv1d(2 * in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, 3, padding=1)
        self.UpConv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )

    def forward(self, x, red):
        outCat = torch.cat((x, red), dim=1)
        return self.UpConv(self.conv2(self.conv1(outCat)))


class DecoderBlockLast(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv1d(2 * in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, 3, padding=1)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, red):
        # print('lastx', x.shape)
        # print('lastred', red.shape)
        outCat = torch.cat((x, red), dim=1)
        return self.conv3(self.conv2(self.conv1(outCat)))


class Decoder(nn.Module):
    def __init__(self, in_channels,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.DcoderBlock1 = DecoderBlock(in_channels, 64)  # 32 - 64
        self.DcoderBlock2 = DecoderBlock(64, 64)  # 64 - 128
        self.DcoderBlock3 = DecoderBlock(64, 32)  # 128 - 256
        self.DcoderBlock4 = DecoderBlock(32, 32)  # 256 - 512
        self.DcoderBlock5 = DecoderBlockLast(32, 1)  # 512 - 1024

    def forward(self, x, redList):

        out1 = self.DcoderBlock1(x, redList[4])
        out2 = self.DcoderBlock2(out1, redList[3])
        out3 = self.DcoderBlock3(out2, redList[2])
        out4 = self.DcoderBlock4(out3, redList[1])
        out5 = self.DcoderBlock5(out4, redList[0])
        return out5


class UNet1d(nn.Module):
    def __init__(self, N_Steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Encoder = Encoder(1, n_Steps=N_Steps)
        self.Decoder = Decoder(128)

        self.Mid = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2
            )
        )

    def forward(self, x, t):
        outDown, redList = self.Encoder(x, t)

        outMid = self.Mid(outDown)

        return self.Decoder(outMid, redList)


if __name__ == '__main__':


    x_0 = torch.randn(10, 1, 1024)  # 例如，使用输入数据的示例
    device = x_0.device
    batch_size = x_0.shape[0]

    # 对一个batch生成随机覆盖更多得t
    t = torch.randint(0, 1000, (batch_size // 2,)).to(device)
    t = torch.cat([t, 1000 - 1 - t], dim=0).to(device)
    print(t.shape)
    model = UNet1d(1000).to(device)


    macs, params = profile(model, inputs=(x_0, t))
    print(params)
