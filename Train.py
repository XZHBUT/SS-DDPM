import math
import os

import numpy as np
import torch

from Diffusion import GetNoiseSet as GetPre
from Model.FiLMUNet import FiLM_UNet
from tool.ChioseDataSample import ChoiseDataCWRU, ChoiseDataJNU
from Diffusion.Diffusion_Process import diffusion_loss_fn
from tool.DrawLoss import DrawLoss
from tool.DrawDeNoise import DrawDeNose
from pathlib import Path

def Train(device, model, optimizer, dataset, num_steps, num_epoch,batch_size,max_lr,min_lr,dataset_name,pth_filepath, loss_filepath,isDrawDeNose):
    model.to(device)
    dataloader = torch.utils.data.DataLoader(dataset.to(device), batch_size=batch_size, shuffle=True)

    Betas = GetPre.Get_Betas_linear(num_steps=num_steps).to(device)

    alphas = GetPre.Get_alphas(Betas).to(device)

    alphas_bar = GetPre.Get_alphas_bar(Betas).to(device)

    alphas_bar_sqrt = GetPre.Get_alphas_bar_sqrt(Betas).to(device)

    one_minus_alphas_bar_sqrt = GetPre.Get_one_minus_alphas_bar_sqrt(Betas).to(device)

    trainLoss = []
    # 初始化最低损失为一个较大的值
    min_loss = float('inf')
    for t in range(num_epoch):
        model.train()
        # 使用余弦退火公式计算学习率
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(t / num_epoch * math.pi))
        # 设置优化器的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        loss = 0
        for idx, batch_x in enumerate(dataloader):
            loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        print("epoch:{}, Loss:{}".format(t + 1, loss))
        trainLoss.append(loss)
        if loss < min_loss:
            # 更新最低损失
            min_loss = loss
            # 保存模型参数
            torch.save(model.state_dict(), pth_filepath)

        if (t + 1) % 100 == 0:
            if isDrawDeNose == 1:
                model.eval()
                DrawDeNose(model, device, Tensorshape=dataset[0].shape, Betas=Betas, one_minus_alphas_bar_sqrt= one_minus_alphas_bar_sqrt,
                           num_steps=num_steps, isGPuDraw=1, num_Draw=5, save_path=f"Show/DeNoise/FiLMUnet_({dataset_name})_Epoch({t + 1}).png",
                           )
                model.train()

    DrawLoss(num_epoch=num_epoch,trainLoss=trainLoss, loss_filepath=loss_filepath)








if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_steps = 3000
    GPuDraw = 1
    batch_size = 128
    num_epoch = 400
    # 设置初始学习率和最小学习率
    max_lr = 0.001
    min_lr = 0.00001
    isDrawDeNose = 1

    # 实际数据
    filenames = []
    filepaths = []
    datapath = f'data/ChoiseData'
    folder = Path(datapath)
    # 遍历目录下的所有文件
    for file in folder.iterdir():
        if file.is_file():
            filenames.append(file.stem)  # 使用 stem 获取不含后缀的文件名
            filepaths.append(str(file))  # 将 Path 对象转换为字符串，存储路径

    for filepath_i, filename_i in zip(filepaths, filenames):
        # print(filepath_i)
        # print(filename_i)
        model = FiLM_UNet(num_steps).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
        # loss_filepath = f"Show/Loss/Loss_FiLMUnet_({filename_i})_({num_steps}).png"
        # pth_filepath = f"pth/CWRU/3HP/FiLMUnet_({filename_i})_({num_steps}).pth"
        # data = ChoiseDataCWRU(filepath_i, 100, 2048)
        loss_filepath = f"Show/Loss/Loss_FiLMUnet_({filename_i})_({num_steps}).png"
        pth_filepath = f"pth/JNU/FiLMUnet_({filename_i})_({num_steps}).pth"
        data = ChoiseDataJNU(filepath_i, 100, 2048)
        data = np.array(data)  # 将列表转换为单一的 numpy.ndarray
        dataset = torch.Tensor(data).float().to(device)
        dataset = torch.unsqueeze(dataset, dim=1)

        Train(device=device,
              model=model,
              optimizer=optimizer,
              dataset=dataset,
              num_steps=num_steps,
              num_epoch=num_epoch,
              batch_size=batch_size,
              max_lr=max_lr,
              min_lr=min_lr,
              dataset_name=filename_i,
              loss_filepath=loss_filepath,
              pth_filepath=pth_filepath,
              isDrawDeNose=isDrawDeNose)


