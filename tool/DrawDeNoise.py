import os

import numpy as np
from matplotlib import pyplot as plt

from Diffusion.Diffusion_Process import p_sample_loop
from tool.DrawFFt import sigal_to_fft


# 定义滑动窗口函数
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def DrawDeNose(model,device,Tensorshape, Betas, one_minus_alphas_bar_sqrt, num_steps,isGPuDraw, num_Draw, save_path):
    spanr = num_steps // num_Draw
    if isGPuDraw == 1:
        x_seq = p_sample_loop(model, Tensorshape, Betas, one_minus_alphas_bar_sqrt, num_steps)
        fig, axs = plt.subplots(1, num_Draw, figsize=(28, 3))
        for i in range(1, num_Draw + 1):
            cur_x = x_seq[i * spanr].detach().cpu()
            fftx, ffty = sigal_to_fft(cur_x[0][0][::])
            # 对 FFT 结果进行滑动窗口平滑
            smoothed_ffty = moving_average(ffty, window_size=10)
            # 截取对应长度的 fftx，因为滑动窗口减少了 y 轴的长度
            smoothed_fftx = fftx[:len(smoothed_ffty)]
            # 绘制平滑后的信号
            axs[i - 1].plot(smoothed_fftx, smoothed_ffty, color='b')
            # 设置 y 轴范围为 [-0.02, 0.02]
            axs[i - 1].tick_params(axis='both', which='major', labelsize=14)
            axs[i - 1].set_title(f"$q(\\mathbf{{x}}_{{{i * spanr}}})$", fontsize=26)

        # 保存图形到文件
        plt.savefig(save_path)
        # 关闭当前图形，以便下一次迭代创建新的图形
        plt.close()
    else:
        x_seq = p_sample_loop(model.to('cpu'), Tensorshape, Betas.to('cpu'),
                                 one_minus_alphas_bar_sqrt.to('cpu'), num_steps, device='cpu')
        fig, axs = plt.subplots(1, num_Draw, figsize=(28, 3))
        for i in range(1, num_Draw + 1):
            cur_x = x_seq[i * spanr].detach().cpu()
            fftx, ffty = sigal_to_fft(cur_x[0][0][::])
            axs[i - 1].plot(fftx, ffty, color='b')
            axs[i - 1].set_axis_off()
            # 设置 y 轴范围为 [-0.02, 0.02]
            axs[i - 1].tick_params(axis='both', which='major', labelsize=14)
            axs[i - 1].set_title(f"$q(\\mathbf{{x}}_{{{i * spanr}}})$", fontsize=16)

        # 保存图形到文件
        plt.savefig(save_path)
        # 关闭当前图形，以便下一次迭代创建新的图形
        plt.close()
        model.to(device)
        Betas.to(device)
        one_minus_alphas_bar_sqrt.to(device)
