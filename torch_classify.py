import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from data import get_train_data, get_test_data
from metric import evaluation
'''
该部分是使用Torch框架下的简单神经网络进行分类
'''


class Net(nn.Module):
    '''
    双隐层神经网络类
    '''
    def __init__(self):
        super(Net, self).__init__()

        self.input = nn.Linear(
            in_features=75,
            out_features=100,
        )
        self.hidden1 = nn.Linear(
            in_features=100,
            out_features=100,
        )
        self.hidden2 = nn.Linear(
            in_features=100,
            out_features=50,
        )
        self.output = nn.Linear(
            in_features=50,
            out_features=9,
        )

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.output(x)
        return x


def data_wrap(X, y, batch_size=16, train_all=True, frac=1 / 3):
    '''
    Pytorch使用其自己定义的DataLoader进行数据集的封装与训练
    我们在这里用train_all决定是否划分训练集和测试集训练，类似于在forest.py
    中test函数的write参数

    Parameters
    ----------
    X : 特征向量
    y : 标签向量
    batch_size : 训练数据集批次大小
    train_all : 是否划分训练
    frac : 划分训练集比例

    Returns
    -------
    train_loader : 用于之后训练的数据封装
    X_test_tensor : 测试集特征张量
    y_test_tensor : 测试集标签张量
    '''
    if train_all:
        X_train, y_train = X, y
        X_test, y_test = X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            random_state=42,
            train_size=frac,
        )

    X_train_tensor = torch.from_numpy(X_train.values).type(torch.FloatTensor)
    y_train_tensor = torch.from_numpy(y_train.values).type(torch.LongTensor)
    X_test_tensor = torch.from_numpy(X_test.values).type(torch.FloatTensor)
    y_test_tensor = torch.from_numpy(y_test.values).type(torch.FloatTensor)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, X_test_tensor, y_test_tensor


def torch_train(
    net: Net,
    train_loader: DataLoader,
    epochs,
    optimizer,
    criterion,
):
    for epoch in range(epochs):

        running_loss = 0.0

        for batch_idx, data in enumerate(train_loader):
            # get inputs and labels
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if (batch_idx + 1) % 1000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, EPOCHS, batch_idx + 1, len(train_loader),
                    loss.item()))
    return net


if __name__ == "__main__":
    EPOCHS = 5
    BATCH_SIZE = 32

    X, y = get_train_data()
    train_loader, X_test_tensor, y_test_tensor = data_wrap(
        X,
        y,
        train_all=True,
    )

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    predict = F.softmax(
        torch_train(
            net,
            train_loader,
            EPOCHS,
            optimizer,
            criterion,
        )(X_test_tensor)).detach().numpy()

    output = pd.DataFrame({
        'id': range(predict.shape[0]),
        'Class_1': predict[:, 0],
        'Class_2': predict[:, 1],
        'Class_3': predict[:, 2],
        'Class_4': predict[:, 3],
        'Class_5': predict[:, 4],
        'Class_6': predict[:, 5],
        'Class_7': predict[:, 6],
        'Class_8': predict[:, 7],
        'Class_9': predict[:, 8],
    })
    output.id = range(200000, 300000)
    output.to_csv("torch_classify.csv", index=False)
