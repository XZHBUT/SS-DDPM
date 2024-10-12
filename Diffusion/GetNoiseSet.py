import numpy as np
import torch
from matplotlib import pyplot as plt


def Get_Betas_linear(num_steps=1000):
    # betas

    scale = 1000 / num_steps  # scale to 1 (under 1000 timesteps)
    beta_start = scale * 1e-4
    beta_end = scale * 0.02
    betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)

    # betas = torch.linspace(1e-4, 0.02, num_steps, dtype=torch.float32)
    return betas


from scipy.interpolate import CubicSpline


def X2_to_Betas(start=1e-4, end=0.02, T=1000):
    X = torch.linspace(np.sqrt(start), np.sqrt(end), T)
    Y = X * X

    return Y


def CoSinAlpha_to_Betas(T=1000, s=0.008):
    def f_t(t, T=1000, s=0.008):
        alpha_t = np.cos(np.pi * 0.5 * (t / T + 2) / (1 + s)) ** 2
        return alpha_t

    def noise_schedule(t, T, s):
        alpha_t = f_t(t, T, s) / f_t(0, T, s)
        return alpha_t

    def variance_schedule(alpha_t):
        batas = []
        for i in range(1, len(alpha_t)):
            batas.append(min(1 - alpha_t[i] / alpha_t[i - 1], 0.999))
        return batas

    t_values = np.arange(0, T + 1, 1)
    alpha_values = noise_schedule(t_values, T, s)
    beta_values = variance_schedule(alpha_values[:])

    return torch.tensor(beta_values).float()


def fibonacci_to_Betas(b0=1 / 10 ** 6, b1=2 / 10 ** 6, T=25):
    result = np.zeros(T)
    result[0] = b0
    result[1] = b1

    for i in range(2, T):
        result[i] = result[i - 1] + result[i - 2]
    return torch.tensor(result).float()


def generate_exponential_values(min_val, max_val, t):
    # 确保最小值是大于0的，因为指数分布的定义域是(0, +∞)
    assert min_val > 0, "最小值必须大于0"

    # 计算尺度参数，使得指数分布的最大值接近给定的最大值
    # 这里我们使用逆函数方法，即1 - exp(-lambda * max_val) = 1 - epsilon，其中epsilon是一个非常小的数
    epsilon = 1e-6  # 一个非常小的数
    lambda_param = -np.log(epsilon) / max_val

    # 使用numpy的指数分布生成t个值
    values = np.random.exponential(scale=1 / lambda_param, size=t)

    # 只保留那些在给定的最小值和最大值之间的值
    values = values[(values >= min_val) & (values <= max_val)]

    return values


def ShowBetas(Betas, MoreData=0, label=None):
    if MoreData == 1:
        t_values = np.arange(0, len(Betas[0]), 1)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        for i in range(len(Betas)):
            plt.plot(np.arange(0, len(Betas[i]), 1), Betas[i], label=label[i])
        plt.title('Noise Schedule')
        plt.xlabel('t')
        plt.ylabel(r'$\beta_t$')
        plt.legend()
        plt.show()
    else:
        t_values = np.arange(0, len(Betas), 1)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(t_values, Betas, label=r'$\beta_t$')
        plt.title('Noise Schedule')
        plt.xlabel('t')
        plt.ylabel(r'$\beta_t$')
        plt.legend()
        plt.show()


def Get_alphas(Betas):
    return 1 - Betas


# alphas_bar是累乘
def Get_alphas_bar(Betas):
    alphas = Get_alphas(Betas)
    alphas_bar = torch.cumprod(alphas, dim=0)
    # alphas_prod_p = torch.cat([torch.ones(1), alphas_prod[:-1]], dim=0).to(device)
    return alphas_bar


# alphas_bar开根号
def Get_alphas_bar_sqrt(Betas):
    alphas_bar = Get_alphas_bar(Betas)
    alphas_bar_sqrt = torch.sqrt(alphas_bar)
    return alphas_bar_sqrt


# 1 - alphas_bar开根号
def Get_one_minus_alphas_bar_sqrt(Betas):
    alphas_bar = Get_alphas_bar(Betas)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_bar)
    return one_minus_alphas_bar_sqrt


def ShowAlphasBar(AlphasBar, MoreData=0, label=None):
    if MoreData == 1:
        t_values = np.arange(0, len(AlphasBar[0]), 1)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        for i in range(len(AlphasBar)):
            plt.plot(np.arange(0, len(AlphasBar[i]), 1), AlphasBar[i], label=label[i])
        plt.title('Noise Schedule')
        plt.xlabel('t')
        plt.ylabel(r'$\bar{\alpha_t}$')
        plt.legend()
        plt.show()
    else:
        t_values = np.arange(0, len(AlphasBar), 1)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(t_values, AlphasBar, label=r'$\beta_t$')
        plt.title('Noise Schedule')
        plt.xlabel('t')
        plt.ylabel(r'$\bar{\alpha_t}$')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    ShowBetas(generate_exponential_values(1, 10, 1000))
    ShowBetas(Get_Betas_linear())

    ShowBetas(CoSinAlpha_to_Betas())

    ShowBetas(fibonacci_to_Betas(1 / 10 ** 6, 2 / 10 ** 6, 25))

    # alpha_t = np.cos(np.pi * 0.5 * (t / T + 2) / (1 + s)) ** 2

    ShowBetas([Get_Betas_linear(num_steps=500), CoSinAlpha_to_Betas(), fibonacci_to_Betas(1 / 10 ** 6, 2 / 10 ** 6, 25),
               X2_to_Betas()],
              MoreData=1, label=[r'1', r'2', r'3', r'4'])

    ShowAlphasBar(
        AlphasBar=[Get_alphas_bar(Get_Betas_linear(num_steps=500)),
                   Get_alphas_bar(Get_Betas_linear(num_steps=1000)),
                   Get_alphas_bar(CoSinAlpha_to_Betas()),
                   Get_alphas_bar(fibonacci_to_Betas())],
        MoreData=1,
        label=[r'1', r'2', r'3', r'4']
    )

#
# plt.subplot(1, 2, 1)
# plt.plot(t_values, alpha_values, label=r'$\bar{\alpha}_t$')
# plt.title('Noise Schedule')
# plt.xlabel('t')
# plt.ylabel(r'$\bar{\alpha}_t$')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(t_values[1:], beta_values, label=r'$\beta_t$')
# plt.title('Variance Schedule')
# plt.xlabel('t')
# plt.ylabel(r'$\beta_t$')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
