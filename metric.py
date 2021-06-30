import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def evaluation(result, test_y) -> float:
    '''
    用于计算两个Dataframe数据的对数损失

    Parameters
    ----------
    result: 学习器预测出的概率矩阵Dataframe
    test_y: 对应数据的类别向量

    Returns
    -------
    loss: 分类器在给定数据上的对数损失
    '''
    result.drop("id", axis=1, inplace=True)
    result = np.array(result)
    test_y_array = np.array(test_y).reshape(-1, 1)

    # 计算对数损失时的trick：调整数值为0项以减少惩罚
    result[result == 0.0] = 0.0001

    enc = OneHotEncoder()
    enc.fit(test_y_array)
    y = enc.transform(test_y_array).toarray()

    loss = -np.sum(np.log(result) * y) / result.shape[0]
    return loss