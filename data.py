import pandas as pd
'''
该文件用于获取数据并处理
注：初始文件在压缩包中，应提前解压缩
'''


def get_train_data(path="train.csv"):
    '''获取训练集的属性值和标签值'''
    df = pd.read_csv(path)
    value_map = {
        'Class_1': 0,
        'Class_2': 1,
        'Class_3': 2,
        'Class_4': 3,
        'Class_5': 4,
        'Class_6': 5,
        'Class_7': 6,
        'Class_8': 7,
        'Class_9': 8,
    }
    df = df.replace({'target': value_map})
    df = df.drop(columns=['id'])
    x_train = df.iloc[:, :-1]
    y_train = df['target']
    return x_train, y_train


def get_test_data(path="test.csv"):
    '''获取测试数据，只有属性值'''
    df = pd.read_csv(path)
    x_test = df.iloc[:, 1:]  # keep the id column for output
    return x_test


def data_split(path="train.csv", frac=0.5):
    '''
    将有标签数据划分为训练集和测试集

    Parameters
    ----------
    path : 训练集的文件路径
    frac : 训练集占总数据条数的比例，默认为0.5

    Return
    ------
    x_train:训练集的属性值
    y_train:训练集的类别向量
    x_test:测试集的属性值
    y_test:测试集的类别向量
    '''
    df = pd.read_csv(path).iloc[:10000]
    value_map = {
        'Class_1': 0,
        'Class_2': 1,
        'Class_3': 2,
        'Class_4': 3,
        'Class_5': 4,
        'Class_6': 5,
        'Class_7': 6,
        'Class_8': 7,
        'Class_9': 8,
    }
    df = df.replace({'target': value_map})
    df = df.drop(columns=['id'])
    df = df.sample(frac=1)
    data_len = len(df)
    bound = int(data_len * frac)
    df_train = df.iloc[:bound]
    df_test = df.iloc[bound:]
    return (
        df_train.iloc[:, :-1],
        df_train['target'],
        df_test.iloc[:, :-1],
        df_test['target'],
    )