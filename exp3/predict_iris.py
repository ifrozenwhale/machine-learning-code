from csv import reader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt
import math
import sys
from data_process import split_multi_dataset, maxminnorm
from model.bpnn2 import BP_network


def read_data2(path='data/Iris.csv', rate=0.7):
    f = open(path, encoding='utf-8')
    data = np.loadtxt(f, str, delimiter=",", skiprows=1)
    # x = data[:, 1:5].astype(np.float64)
    n = data.shape[1]-1
    x = data[:, 1:n].astype(float)
    y = data[:, n]
    x = maxminnorm(x)
    # 映射y
    ys = []
    for e in y:
        if e == 'Iris-setosa':
            ys.append([1, 0, 0])
        elif e == 'Iris-versicolor':
            ys.append([0, 1, 0])
        else:
            ys.append([0, 0, 1])

    ys = np.array(ys)
    ys = ys.astype(float)

    return x, ys


if __name__ == "__main__":
    x, y = read_data2()
    rate = 0.8
    n = int(rate * len(x))
    trainx = x[0:n, ]
    trainy = y[0:n, ]
    testx = x[n:, ]
    testy = y[n:, ]
    sizes = [4, 8, 4, 3]
    bpnn = BP_network(sizes)
    bpnn.train(trainx=trainx, trainy=trainy,
               epochs=4000, learning_rate=0.2)
    _, acc = bpnn.evaluate(testx, testy)
    print(acc)

    bpnn.plot()
