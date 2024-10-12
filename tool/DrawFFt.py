import numpy as np


def sigal_to_fft(data):
    # 进行傅里叶变换
    fft_result = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(len(data), 1 / 2000)

    # 只保留频谱中大于零的部分
    positive_freq_mask = fft_freq > 0
    positive_freq = fft_freq[positive_freq_mask]

    # 取FFT结果的幅度（复数的模）
    fft_result_positive = np.abs(fft_result[positive_freq_mask]) / len(data)  # 取幅值并归一化
    return positive_freq, fft_result_positive