import numpy as np
import os

from scipy.io import loadmat
from sklearn import preprocessing
import pandas as pd


def read_OneOH(filepath):
    '''
    # 获取一种负载文件下的的数据

    :return: dict{} key：类名， Velue：该的全部电波
    '''

    # 获得该文件夹下所有.mat文件名,获取指定文件夹下所有文件的文件名，并将它们存储在filenames变量
    # filenames = os.listdir(filepath)

    files = []
    file_path = filepath
    file = loadmat(file_path)
    file_keys = file.keys()
    for key in file_keys:
        if 'DE' in key:
            files = file[key].ravel()

    return files


def SliceDate(data, SampleNumpy, SampleLength):
    step = int(SampleLength / 2)
    dataSample = []
    for i in range(SampleNumpy):
        sample = data[i * step: i * step + SampleLength]
        dataSample.append(sample)



    return dataSample


def ChoiseDataCWRU(filepath, SampleNumpy=200, SampleLength=1024):
    data = read_OneOH(filepath)
    re = SliceDate(data, SampleNumpy, SampleLength)
    return re  # list




def ChoiseDataJNU(filepath, SampleNumpy=200, SampleLength=1024):

    # file_path = os.path.join(filepath, os.listdir(filepath)[0])
    # file_path = os.path.join(filepath, os.listdir(filepath)[0])

    data = pd.read_csv(filepath, skiprows=16, usecols=[0])
    data = data.values.flatten()

    Samples = []
    step = int(SampleLength / 2)
    for i in range(SampleNumpy):
        sample = data[i * step: i * step + SampleLength]
        Samples.append(sample)

    return np.array(Samples, dtype=float)


if __name__ == '__main__':
    # data = read_OneOH(f'data/ChoiseData')
    # re = SliceDate(data, 200, 1024)
    # print(np.shape(np.array(re)))

    # data = ChoiseDataJNU(f'data/ChoiseData')


    data = ChoiseDataJNU(f'../data/ChoiseData')
    print(data)
