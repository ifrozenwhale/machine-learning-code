# -*- coding: utf-8 -*-

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
print(iris)
# print(list(iris.keys()))
# print(iris['DESCR'])
# print(iris['feature_names'])
X = iris['data'][:, 3:]
# print(X)
# 二分类，变成0--1
y = (iris['target'] == 2).astype(np.int)
print(y)
log_reg = LogisticRegression(solver="sag", max_iter=1000)
log_reg.fit(X, y)
X_new = np.linspace(1, 3, 1000).reshape(-1, 1)
# 注意predict_proba.与predict的区别
y_proba = log_reg.predict_proba(X_new)
y_hat = log_reg.predict(X_new)
print(y_proba)
print(y_hat)
