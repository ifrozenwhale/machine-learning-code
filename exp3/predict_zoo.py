from csv import reader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt
import math
import sys
from data_process import split_multi_dataset, maxminnorm
from model.bpnn2 import BP_network


def read_data2(path='data/zoo.csv', rate=0.6):
    f = open(path, encoding='utf-8')
    data = np.loadtxt(f, str, delimiter=",", skiprows=1)
    # x = data[:, 1:5].astype(np.float64)
    n = data.shape[1]-1
    x = data[:, 1:n].astype(float)
    y = data[:, n]
    x = maxminnorm(x)
    # 映射y
    ys = []
    for i in y:
        yy = [0] * 7
        yy[int(i)-1] = 1
        ys.append(yy)

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
    sizes = [len(trainx[0]), 4, len(trainy[0])]
    bpnn = BP_network(sizes)
    bpnn.train(trainx=trainx, trainy=trainy,
               epochs=10000, learning_rate=0.1)
    _, acc = bpnn.evaluate(testx, testy)
    print(acc)

    bpnn.plot()
