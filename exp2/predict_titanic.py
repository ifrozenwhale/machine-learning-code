import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from model.decision_tree_c import tree_genrate, predict
from draw_tree import draw
from data_process import split_multi_dataset
import csv
df_train = pd.read_csv("data/titanic/train.csv")
df_test = pd.read_csv("data/titanic/test.csv")
print('训练数据集:', df_train.shape)
print('测试数据集:', df_test.shape)
data = df_train.append(df_test, ignore_index=True)
print('合并数据集:', data.shape)
print(data.info())
print(data.describe())

print('缺失值:\n', data.isnull().sum())
# 数值类缺失值：使用中位数进行填充
data.Age = data.Age.fillna(data.Age.median())
data.Fare = data.Fare.fillna(data.Fare.median())

# print('缺失值:\n', data.isnull().sum())
# 特征提取
data.Sex = data.Sex.map(lambda i: 0 if i == 'male' else 1)
# print('性别:\n', data.Sex.value_counts())
# print('\n客舱:\n', data.Cabin.value_counts())


features = pd.concat([data.Pclass,
                      data.Age,
                      data.Sex,
                      data.Fare], axis=1)
labels = ['Pclass', 'Age', 'Sex', 'Fare']
passenger_id = data.loc[891:, 'PassengerId'].values.tolist()
trainx = features.loc[0:890, :].values.tolist()

trainy = data.loc[0:890, 'Survived'].values.tolist()
trainy = [str(y) for y in trainy]

# start predict
testx = features.loc[891:, :].values.tolist()
# testy = data.loc[890:, 'Survived'].values.tolist()

tree = tree_genrate(trainx, trainy, labels)
draw(tree, "ticnic", "svg")
ypred = predict(testx, tree, labels)
ypred = [int(float(x)) for x in ypred]
xy = {"PassengerId": passenger_id, "Survived": ypred}
df_ypred = pd.DataFrame(xy, columns=[
                        'PassengerId', 'Survived'])
# print(df_ypred)
df_ypred.to_csv('./data/titanic_pred.csv', index=False)
