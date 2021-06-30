from csv_ensemble import csv_ensemble
import os
import pandas as pd
'''我们将main函数设计为运行我们介绍过的三种方法，并进行集成'''

print("用简单神经网络进行分类：")
os.system("python torch_classify.py")
print("分类完成")

print("用TF框架的复杂网络进行分类：")
os.system("python tf_classify.py")
print("分类完成")

print("进行深度森林分类：")
os.system("python forest.py")
print("分类完成")

print("读取分类结果")
df1 = pd.read_csv("torch_classify.csv")
df2 = pd.read_csv("tf_classify.csv")
df3 = pd.read_csv("forest_classify.csv")

result = csv_ensemble([df1, df2, df3], [1, 1, 1])
result.to_csv("my_submission.csv", index=False)