
import numpy as np
from model.decision_tree import tree_genrate, predict
from data_process import split_multi_dataset
from collections import Counter
from draw_tree import draw

import copy
import csv
from itertools import islice


def read_data(path='data/Iris.csv', rate=0.7):
    f = open(path, encoding='utf-8')
    data = np.loadtxt(f, str, delimiter=",", skiprows=1)
    # x = data[:, 1:5].astype(np.float64)
    n = data.shape[1]-1
    x = data[:, 1:n]
    y = data[:, n]
    tax, tay, tex, tey = split_multi_dataset(x, y.reshape(len(y), 1), rate)
    return tax.tolist(), tay.reshape(1, len(tay))[0].tolist(), tex.tolist(), tey.reshape(1, len(tey))[0].tolist()


def read_data2(path):
    labels = []
    x = []
    y = []
    cnt = 0
    with open(path, encoding="utf-8-sig") as f:
        # labels.append(f.readline(1).split(','))
        lines = csv.reader(f)
        for line in lines:
            # print(line)
            cnt += 1
            if cnt == 1:
                labels = line[1:-1]
            else:
                x.append(line[1:-1])
                y.append(line[-1])

    return x, y, labels


def read_labels(path='data/Iris.csv'):
    f = open(path, encoding='utf-8')
    labels = f.readline().strip()

    return labels.split(',')[1:-1]


def test():
    p = "data/watermelon20_2.csv"
    # read_data2(p)

    # labels = read_labels(path=p)

    x, y, labels = read_data2(path=p)
    trainx = x[0:10]
    trainy = y[0:10]
    testx = x[10:]
    testy = y[10:]

    # print(main_class)
    tree = tree_genrate(trainx, trainy, labels)
    print(tree)

    predict(testx, testy, tree, labels)
    draw(tree, "watermelonDevisionTreeTrain", "png")


if __name__ == '__main__':
    test()
