import os

import pandas as pd
import numpy as np
from tool.DataProcess import read_OrginCWRU


def read_csv_file(filepath, num_rows=None):
    """
    读取 CSV 文件，并返回 (B, 2048) 的 NumPy 数组，同时跳过包含 NaN、Inf 或空行的数据。

    参数:
    - filepath: CSV 文件路径
    - num_rows: 要读取的行数，默认为 None，表示读取全部行

    返回:
    - np_array: 形状为 (B, 2048) 的 NumPy 数组
    """
    # 读取 CSV 文件，指定读取的行数（如果 num_rows 为 None 则读取全部）
    df = pd.read_csv(filepath, header=None, nrows=num_rows)

    # 替换 Inf 和 -Inf 为 NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 删除包含 NaN 或空值的行
    df.dropna(inplace=True)

    # 将 DataFrame 转换为 NumPy 数组
    np_array = df.to_numpy()

    return np_array

def read_Create_CWRU_OneOH(filepath, ClassNameList):
    # 获得该文件夹下所有文件名,获取指定文件夹下所有文件的文件名，并将它们存储在filenames变量
    filenames = os.listdir(filepath)
    files = {}
    for i in filenames:
        ClassName = ''
        for one in ClassNameList:
            if one in i:
                ClassName = one
                break

        file_path = os.path.join(filepath, i)
        file = read_csv_file(file_path)
        files[ClassName] = file

    return files
def SliceDate_Create_CWRU(data, slice_rate,SampleNumpy):
    keys = data.keys()
    Train_Samples = {}
    Vaild_Samples = {}
    Test_Samples = {}
    # 遍历所有的数据文件
    for i in keys:
        slice_data = data[i]  # 每个键（数据文件）的数据，将其存储在 slice_data 变量中
        samp_train = int(SampleNumpy * slice_rate[0])  # 计算每一类用于训练的样本数量
        samp_vaild = int(SampleNumpy * slice_rate[1])  # 计算每一类用于验证的样本数量
        samp_test = int(SampleNumpy * slice_rate[2])  # 计算每一类用于测试的样本数量
        Train_sample = slice_data[:samp_train, :]
        Vaild_sample = slice_data[samp_train:samp_train + samp_vaild , :]
        Test_sample = slice_data[samp_train + samp_vaild:samp_train + samp_vaild + samp_test, :]
        Train_Samples[i] = Train_sample
        Vaild_Samples[i] = Vaild_sample
        Test_Samples[i] = Test_sample  # 训练集和测试集样本被存储在 Train_Samples 和 Test_Samples 字典中，以数据文件名 i 作为键

    return Train_Samples, Vaild_Samples, Test_Samples

def add_labels_Create_CWRU(data_Dict, ClassNameList):
    '''

    :param data_Dict: 一个负载下的字典
    :param ClassNameList:  类名
    :return:  两个列表
    '''
    X = []
    Y = []
    label = 0
    for i in ClassNameList:
        x = data_Dict[i]
        X += x.tolist()
        lenx = len(x)
        Y += [label] * lenx
        label += 1
    return X, Y
def read_Create_CWRU(filepath, SampleNum=200, Rate=[0.7, 0.2, 0.1]):
    ClassName = ['B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007', 'OR014', 'OR021', 'normal']
    # 获取指定路径下的所有文件夹名称
    folder_names = [f for f in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, f))]
    Train_X, Train_Y = [], []
    Valid_X, Valid_Y = [], []
    Test_X, Test_Y = [], []
    # 按照不同负载条件分别读取
    for OneOH_File in folder_names:

        OneOH_Dict = read_Create_CWRU_OneOH(filepath + '/' + OneOH_File, ClassName)
        Train_Samples_OneOH, Vaild_Samples_OneOH, Test_Samples_OneOH = SliceDate_Create_CWRU(OneOH_Dict, Rate, int(SampleNum / 4))


        Train_X_OneOH, Train_Y_OneOH = add_labels_Create_CWRU(Train_Samples_OneOH, ClassName)
        Valid_X_OneOH, Valid_Y_OneOH = add_labels_Create_CWRU(Vaild_Samples_OneOH, ClassName)
        Test_X_OneOH, Test_Y_OneOH = add_labels_Create_CWRU(Test_Samples_OneOH, ClassName)
        Train_X += Train_X_OneOH
        Train_Y += Train_Y_OneOH
        Valid_X += Valid_X_OneOH
        Valid_Y += Valid_Y_OneOH
        Test_X += Test_X_OneOH
        Test_Y += Test_Y_OneOH

    Train_X = np.asarray(Train_X)
    Valid_X = np.asarray(Valid_X)
    Test_X = np.asarray(Test_X)
    Test_Y = np.asarray(Test_Y, dtype=np.int32)
    Valid_Y = np.asarray(Valid_Y, dtype=np.int32)
    Train_Y = np.asarray(Train_Y, dtype=np.int32)

    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y

def Read_CWRU_MIX(orign_filepath, create_filepath, orign_num_sample, create_num_sample,SampleLength, Rate=[0.8, 0.1, 0.1] ):

    orign_Train_X, orign_Train_Y, orign_Valid_X, orign_Valid_Y, orign_Test_X, orign_Test_Y = read_OrginCWRU(filepath=orign_filepath,
                                                                        SampleLength=SampleLength,
                                                                        SampleNum=orign_num_sample,
                                                                        normal=False,
                                                                        Rate=Rate
                                                                        )

    # 定义空的创建数据集变量
    create_Train_X = np.empty((0, SampleLength))  # 空的创建训练输入数组
    create_Train_Y = np.empty((0,))  # 空的创建训练标签数组

    create_Valid_X = np.empty((0, SampleLength))  # 空的创建验证输入数组
    create_Valid_Y = np.empty((0,))  # 空的创建验证标签数组

    create_Test_X = np.empty((0, SampleLength))  # 空的创建测试输入数组
    create_Test_Y = np.empty((0,))  # 空的创建测试标签数组

    if create_num_sample > 0:
        create_Train_X, create_Train_Y, create_Valid_X, create_Valid_Y, create_Test_X, create_Test_Y = read_Create_CWRU(filepath=create_filepath,
                                                                                                                    SampleNum=create_num_sample,
                                                                                                                    Rate=Rate)

    Train_X = np.vstack((orign_Train_X, create_Train_X))
    Train_Y = np.concatenate((orign_Train_Y, create_Train_Y))
    Valid_X = np.vstack((orign_Valid_X, create_Valid_X))
    Valid_Y = np.concatenate((orign_Valid_Y, create_Valid_Y))
    Test_X = np.vstack((orign_Test_X, create_Test_X))
    Test_Y = np.concatenate((orign_Test_Y, create_Test_Y))
    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y


def Read_JNU_MIX(orign_filepath, create_filepath, orign_num_sample, create_num_sample,SampleLength, Rate=[0.8, 0.1, 0.1] ):
    orign_Train_X, orign_Train_Y, orign_Valid_X, orign_Valid_Y, orign_Test_X, orign_Test_Y = read_OrginCWRU(filepath=orign_filepath,
                                                                        SampleLength=SampleLength,
                                                                        SampleNum=orign_num_sample,
                                                                        normal=False,
                                                                        Rate=Rate
                                                                        )



if __name__ == '__main__':


    # 使用示例
    filepath = '../data/CreateCWRU/0HP/Create_normal_0_97.csv'
    np_array = read_csv_file(filepath, num_rows=None)  # 读取前10行
    print(np_array.shape)  # 输出形状 (B, 2048)

    Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = read_Create_CWRU(filepath='../data/CreateCWRU')
    print(Train_X.shape)

    Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = Read_CWRU_MIX(orign_filepath='../data/CWRU',create_filepath='../data/CreateCWRU',
                  orign_num_sample=200, create_num_sample=200,
                  SampleLength=2048, Rate=[0.8, 0.1, 0.1])

    print(Train_X.shape)
    print(Train_Y.shape)
