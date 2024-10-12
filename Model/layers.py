import torch

from Model.base import BaseModule


class Conv1dWithInitialization(BaseModule):
    def __init__(self, **kwargs):
        super(Conv1dWithInitialization, self).__init__()
        self.conv1d = torch.nn.Conv1d(**kwargs)
        # torch.nn.init.orthogonal_(self.conv1d.weight.data, gain=1).to(self.conv1d.weight.device)

    def forward(self, x):
        # print('s', self.conv1d(x).device)
        return self.conv1d(x)
