import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from deepforest.cascade import CascadeForestClassifier

from data import data_split, get_test_data, get_train_data
from metric import evaluation
'''
本文件是随机森林和深度森林的实验
'''


def train(train_x, train_y, model):
    '''
    由于深度森林和随机森林有类似的接口，于是将其训练流程合并

    Parameters
    ----------
    train_x : 训练集属性
    train_y : 训练集标签

    Returns
    -------
    model : 训练好的模型
    '''
    X = train_x.values
    y = train_y.values
    model.fit(X, y)
    return model


def test(test_x, model, write=False, filename="my_submission.csv"):
    '''
    用输入的模型预测输入的测试集，并决定是否写入预测结果

    Parameters
    ----------
    test_x : 无标签、带预测的数据
    model : 已经训练好的模型
    write : 指定是否将预测结果写入csv
    filename : 写入情况下的csv文件名

    Returns
    -------
    output : 预测概率，可用于训练效果的计算
    '''
    proba = model.predict_proba(test_x.values)
    output = pd.DataFrame({
        'id': test_x.index,
        'Class_1': proba[:, 0],
        'Class_2': proba[:, 1],
        'Class_3': proba[:, 2],
        'Class_4': proba[:, 3],
        'Class_5': proba[:, 4],
        'Class_6': proba[:, 5],
        'Class_7': proba[:, 6],
        'Class_8': proba[:, 7],
        'Class_9': proba[:, 8],
    })
    output.id += 200000
    if write:
        output.to_csv(filename, index=False)
    return output


def forest_test():
    '''
    模型测试
    '''
    loss1 = []
    loss2 = []
    # 测试5次，并将测试结果绘制
    for i in range(5):
        train_x, train_y, test_x, test_y = data_split(frac=0.67)

        model1 = RandomForestClassifier()
        model2 = CascadeForestClassifier()

        model1 = train(train_x, train_y, model1)
        model2 = train(train_x, train_y, model2)

        result1 = test(test_x, model1)
        result2 = test(test_x, model2)

        loss1.append(evaluation(result1, test_y))
        loss2.append(evaluation(result2, test_y))

        print("iter %d over" % (i + 1))

    sns, set()
    plt.plot(range(0, 5), loss1, label="RF")
    plt.plot(range(0, 5), loss2, label="DF")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_x, train_y = get_train_data()
    test_x = get_test_data()
    model = train(
        train_x,
        train_y,
        CascadeForestClassifier(n_estimators=500, max_depth=20),
    )
    output = test(test_x, model, write=True, filename="forest_classify.csv")
