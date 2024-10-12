import torch


# 扩散过程任意时刻采样值
def q_x(x_O, t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device):
    noise = torch.randn_like(x_O).to(device)

    # 取出均值
    alphas_t = alphas_bar_sqrt[t]
    # 取出标准化
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]

    # 参数重采样
    return alphas_t * x_O + alphas_1_m_t * noise


# 逆向去噪过程
def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt, device='cuda:0'):
    '''从x_T恢复x_t'''
    t = torch.tensor([t]).to(x.device)

    # 噪声前系数
    coeff = betas[t] / (one_minus_alphas_bar_sqrt[t])

    # 模型预测的噪声
    eps_theta = model(x, t)

    # 恢复均值
    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))

    z = torch.randn_like(x).to(device)

    # 从噪声的到的方差
    sigma_t = betas[t].sqrt()

    # 恢复
    sample = mean + sigma_t * z
    return (sample)


def p_sample_loop(model, shape, betas, one_minus_alphas_bar_sqrt, n_steps, device='cuda:0'):
    "从x_T恢复到底"
    cur_x = torch.randn(shape).to(device)
    cur_x = cur_x.unsqueeze(0)
    x_seq = [cur_x]
    with torch.no_grad():
        for i in reversed(range(n_steps)):
            cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt, device=device)
            x_seq.append(cur_x)
    torch.enable_grad()
    return x_seq


# 损失函数
def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    # print(x_0.shape)
    device = x_0.device
    batch_size = x_0.shape[0]

    # 对一个batch生成随机覆盖更多得t
    t = torch.randint(0, n_steps, (batch_size // 2,)).to(device)
    t = torch.cat([t, n_steps - 1 - t], dim=0).to(device)
    # print('t', t.shape)
    t = t.unsqueeze(-1).unsqueeze(-1)
    # print('t-1', t.shape)
    # x0的系数
    a = alphas_bar_sqrt[t]

    # eps系数
    aml = one_minus_alphas_bar_sqrt[t]

    # 随机噪声eps
    e = torch.randn_like(x_0).to(device)
    # print('e', e.shape)
    # print('aml', (aml * e).shape)
    # print('ax', (a * x_0).shape)
    # print('amle', (aml * e).shape)

    # 构造输入
    x = a * x_0 + aml * e

    # 输入构造
    output = model(x, t.squeeze(-1).squeeze(-1))

    # 和真实噪声的差距
    return (e - output).square().mean()
