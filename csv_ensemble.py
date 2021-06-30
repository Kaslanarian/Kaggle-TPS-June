import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from data import data_split
from forest import test
from metric import evaluation
'''
我们在实验报告中提出可以对输出概率进行加权集成以提高学习效果的可能性，
我们在这一模块设计了用来进行这一过程的函数
'''


def csv_ensemble(df_list: list, weight_list: list):
    '''
    Parameters
    ----------
    df_list : 存储将要进行集成的结果对应的DataFrame
    weight_list : 权重向量，分别对应集成时的权重

    Returns
    -------
    result : 集成后的结果
    '''
    assert len(df_list) == len(weight_list)
    result = sum([weight_list[i] * df_list[i]
                  for i in range(len(df_list))]) / sum(weight_list)
    result.id = range(200000, 300000)
    return result


if __name__ == "__main__":
    list_rf, list_ada, list_gbdt, list_ens = [], [], [], []
    for i in range(20):
        '''
        训练20次，比较各自方法和集成后的区别
        '''
        logloss1, logloss2, logloss3, logloss4 = 0, 0, 0, 0
        for j in range(5):
            train_x, train_y, test_x, test_y = data_split(frac=1 / 3)
            model1 = RandomForestClassifier()
            model2 = AdaBoostClassifier()
            model3 = GradientBoostingClassifier()

            model1.fit(train_x, train_y)
            model2.fit(train_x, train_y)
            model3.fit(train_x, train_y)

            output1 = test(test_x, model1)
            output2 = test(test_x, model2)
            output3 = test(test_x, model3)

            output4 = (output1 + output2 + output3) / 3

            logloss1 += evaluation(output1, test_y)
            logloss2 += evaluation(output2, test_y)
            logloss3 += evaluation(output3, test_y)
            logloss4 += evaluation(output4, test_y)

        print("%d iter %d cross validation" % (i + 1, j + 1))

        list_rf.append(logloss1 / 5)
        list_ada.append(logloss2 / 5)
        list_gbdt.append(logloss3 / 5)
        list_ens.append(logloss4 / 5)
    
    sns.set()
    plt.plot(list_rf, label="rf")
    plt.plot(list_ada, label="ada")
    plt.plot(list_gbdt, label="gbdt")
    plt.plot(list_ens, label="ensemble")
    plt.legend()
    plt.show()
