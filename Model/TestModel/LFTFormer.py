import time
import math

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.nn import functional as F

import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms


import torch
from thop import profile


from torch.utils.data import TensorDataset, DataLoader

from sklearn.manifold import TSNE


class MFE(nn.Module):
    def __init__(self, out_c, gamma=2, b=1, **kwargs):
        super(MFE, self).__init__(**kwargs)

        # 创建多个分支self.create_branch()
        self.branches = nn.ModuleList()
        for i in range(1, out_c + 1):
            branch = self.create_branch(i)
            self.branches.append(branch)

        self.batch_norm = nn.BatchNorm1d(out_c)

        # self.ECABlock_1d = ECABlock_1d(out_c)

    def forward(self, x):
        # 对每个分支进行前向传播
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        # 在这里可以对各个分支的输出进行合并或其他操作
        # 这里简单地将各个分支的输出放入一个列表返回
        x1 = torch.cat(branch_outputs, dim=1)

        # freq_Weight = self.ECABlock_1d(x1)

        # x2 = x1 * freq_Weight.expand_as(x1)
        x2 = x1

        x3 = F.gelu(x2)

        x4 = self.batch_norm(x3)

        return x4

    def create_branch(self, i):
        # 这个方法用于创建一个分支，你可以根据需要自定义每个分支的结构
        branch = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=2 * i - 1, padding=(2 * i - 1 - 1) // 2, stride=1),
        )
        return branch
# 定义自注意力层
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # 自注意力分数
        attn_scores = attn_scores / (q.size(-1) ** 0.5)  # 缩放
        attn_probs = self.softmax(attn_scores)  # 归一化得到注意力权重
        attended_values = torch.matmul(attn_probs, v)  # 注意力加权的值
        return attended_values


# 定义一个 Transformer 编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attention = SelfAttention(d_model)  # 修改此处，d_model 设置为1
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 512),  # 此处可以改变维度
            nn.ReLU(),
            nn.Linear(512, d_model)
        )

    def forward(self, x):
        # 自注意力和归一化
        attn_output = self.self_attention(x)
        x = x + attn_output
        x = self.norm1(x)

        # 前馈神经网络和归一化
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)

        return x


# 定义整个 Transformer 编码器模型
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# 定义主模型
class TransformerClassifier(nn.Module):
    def __init__(self, d_model, num_layers, num_classes):
        super(TransformerClassifier, self).__init__()
        self.MFV = MFE(d_model)
        self.encoder = TransformerEncoder(d_model, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.MFV(x)
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = x.mean(dim=1)  # 在时间步维度上求平均，将 bx1024xd_model 转换为 bx1xd_model
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # net = nn.Sequential(MFE(2), LFTFormer_block(2, 4), LFTFormer_block(4, 8), LFTFormer_block(8, 16),
    #                     nn.AdaptiveAvgPool1d(1), MLPClassifier(16, 32, 10, 0.5))
    #
    # X = torch.rand(size=(10, 1, 1024))
    # for layer in net:
    #     X = layer(X)
    #     print(layer.__class__.__name__, ' output shape:\t', X.shape)
    d_model = 8  # 修改为1，以匹配输入的维度
    num_layers = 3
    num_classes = 10
    model = TransformerClassifier(d_model, num_layers, num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    input_data = torch.randn(1, 1, 1024)  # 例如，使用输入数据的示例
    macs, params = profile(model, inputs=(input_data,))
    print(f"Total FLOPs: {macs / 10 ** 6} MFLOPs")  # 将FLOPs转换为MFLOPs

