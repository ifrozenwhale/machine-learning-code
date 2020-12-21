
import numpy as np
from draw_tree import draw
from model.decision_tree_c import predict, tree_genrate
from data_process import split_multi_dataset


def read_data(path='data/Iris.csv', rate=0.7):
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
    p = "data/Iris.csv"
    trainx, trainy, testx, testy = read_data(p, rate=0.1)
    labels = read_labels(p)
    # print(main_class)
    tree = tree_genrate(trainx, trainy, labels)
    print(tree)

    ypred = predict(testx, tree, labels)
    res = np.array(testy) == np.array(ypred)
    print("准确率 ", res[res == True].size / res.size)

    draw(tree, "irisDevisionTree", "png")


if __name__ == '__main__':
    test()
