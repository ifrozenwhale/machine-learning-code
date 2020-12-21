
import numpy as np
from draw_tree import draw
from model.decision_tree import predict, tree_genrate
from data_process import split_multi_dataset


def read_data(path='data/zoo.csv', rate=0.7):
    f = open(path, encoding='utf-8')
    data = np.loadtxt(f, str, delimiter=",", skiprows=1)
    # x = data[:, 1:5].astype(np.float64)
    n = data.shape[1]-1
    x = data[:, 1:n].astype(float)
    y = data[:, n]
    tax, tay, tex, tey = split_multi_dataset(x, y.reshape(len(y), 1), rate)

    return tax.tolist(), tay.reshape(1, len(tay))[0].tolist(), tex.tolist(), tey.reshape(1, len(tey))[0].tolist()

# 自己定义测试集和训练集


def read_labels(path='data/Iris.csv'):
    f = open(path, encoding='utf-8')
    labels = f.readline().strip()

    return labels.split(',')[1:-1]


def test():
    p = "data/zoo.csv"
    # read_data2(p)

    # labels = read_labels(path=p)

    # x, y, labels = read_data2(path=p)
    # trainx = x[:]
    # trainy = y[:]
    # testx = x[10:]
    # testy = y[10:]
    trainx, trainy, testx, testy = read_data(p, rate=0.7)
    # print(trainy)
    labels = read_labels(p)
    # print(labels)
    # print(main_class)
    tree = tree_genrate(trainx, trainy, labels)
    print(tree)

    ypred = predict(testx, testy, tree, labels)
    draw(tree, "zooDevisionTree", "png")
    # res = np.array(testy) == np.array(ypred)
    # print("准确率 ", res[res == True].size / res.size)


if __name__ == '__main__':
    test()
