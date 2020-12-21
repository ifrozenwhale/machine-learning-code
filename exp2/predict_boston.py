

from model.reg_tree import tree_genrate, predict, evaluate
from draw_tree import draw
from data_process import split_multi_dataset
import numpy as np
import csv
import matplotlib.pyplot as plt


def read_data2(path):
    labels = []
    x = []
    y = []
    cnt = 0
    with open(path, encoding="utf-8-sig") as f:
        lines = csv.reader(f)
        for line in lines:
            if line[-1] == '':
                continue
            cnt += 1
            if cnt == 1:
                labels = line[0:-1]
            else:
                digl = [float(e) for e in line[0:-1]]
                # digl = [float(e) for e in line[5:6]]
                # x.append([float(line[5]), float(line[7])])
                x.append(digl)
                y.append(float(line[-1]))
    return x, y, labels


def read_labels(path):
    f = open(path, encoding='utf-8')
    labels = f.readline().strip()
    return labels.split(',')[0:-1]


def test():
    p = "data/boston_housing_data.csv"

    x, y, labels = read_data2(path=p)
    rate = 0.5
    n = len(y)
    len_train = int(n * rate)
    trainx = x[0:len_train]
    trainy = y[0:len_train]
    testx = x[len_train:]
    testy = y[len_train:]

    tree = tree_genrate(trainx, trainy, labels)

    ypred = predict(testx, tree, labels)
    ypred = [float(e) for e in ypred]

    dif = []
    for i in range(len(ypred)):
        dif.append(float(ypred[i]) - float(testy[i]))

    xx = np.linspace(0, len(ypred), len(ypred))

    plt.plot(xx, ypred, 'o')
    plt.plot(xx, testy, 'o')
    plt.legend(['predict value', 'observed value'])
    plt.title("Boston house price prdict with regression tree")
    plt.xlabel("data id")
    plt.ylabel("price")
    plt.savefig("./img/Boston_house_price_predict.png", dpi=600)
    plt.show()

    R2 = evaluate(ypred, testy)
    print("R^2 ", R2)


if __name__ == '__main__':
    test()
