import numpy as np
import csv
import matplotlib.pyplot as plt

from logistic_reg import LogisticRegression, sigmoid
from data_process import split_dataset, split_multi_dataset


def plot(x, labels, w, title):
    plt.figure()

    plt.scatter(x[:, 0], x[:, 1], c=labels.reshape(
        1, len(labels)), cmap=plt.cm.Spectral)
    xx = np.linspace(3, 10, 100)
    yy = -w[0] / w[2] - w[1] / w[2] * xx
    plt.title(title)
    plt.plot(xx, yy)
    plt.savefig(
        "./img/{}.png".format(title),  dpi=500)
    plt.show()


def read_data(path='data/Iris.csv', one='Iris-setosa', test=False, rate=0.7):
    with open(path, encoding='utf-8') as f:
        data = np.loadtxt(f, str, delimiter=",", skiprows=1)
        x = data[:, 1:5].astype(np.float64)
        ori_y = data[:, 5]
        if test is True:  # 如果是评估时候，对每一个标签映射为一个数值
            y_list = [(1 if i == 'Iris-setosa' else (2 if i ==
                                                     'Iris-versicolor' else 3)) for i in ori_y]
            y = np.array(y_list, dtype=np.float64)
            return split_multi_dataset(x, y.reshape(len(y), 1), rate)
        else:  # 如果是训练，则将指定的属性映射为1，剩余的映射为0
            y_list = [(1 if i == one else 0) for i in ori_y]
            y = np.array(y_list, dtype=np.float32)
            return split_dataset(x, y.reshape(len(y), 1), rate)


def test(x, labels, w):
    np.set_printoptions(suppress=True)  # 设置输出格式
    labels = labels.reshape(1, len(labels))  # 标签
    xx = np.c_[np.ones((labels.shape[1], 1)), x]  # 拼接xx

    # print(xx)
    z = xx.dot(w)
    y_pred = sigmoid(z)  # 使用sigmod得到预测值
    return y_pred.reshape(1, labels.shape[1])
    # z = z.reshape(1, labels.shape[1])
    # print(z > 0)
    # # result = (z.reshape(1, labels.shape[1]) > 0) == labels
    # return z


if __name__ == '__main__':
    rate = 0.7
    logistic_reg = LogisticRegression(100000, 0.01)
    # Iris-setosa 为正，其余为负
    train_x1, train_y1, test_x1, test_y1 = read_data(one='Iris-setosa')
    w1, loss1 = logistic_reg.fit(train_x1, train_y1)
    # plot(train_x1, train_y1, w1, "(Iris-setosa) logistic regression on training set")

    # Iris-versicolor 为正，其余为负
    train_x2, train_y2, test_x2, test_y2 = read_data(one='Iris-versicolor')
    w2, loss2 = logistic_reg.fit(train_x2, train_y2)
    # plot(train_x2, train_y2, w2,"(Iris-versicolor) logistic regression on training set")

    # Iris-virginica 为正，其余为负
    train_x3, train_y3, test_x3, test_y3 = read_data(one='Iris-virginica')
    w3, loss3 = logistic_reg.fit(train_x3, train_y3)
    # plot(train_x3, train_y3, w3,"(Iris-virginica) logistic regression on training set")

    # 得到三个分类器w1, w2, w3
    # w1: 1 setosa
    # w2: 1 versicolor
    # w3: 1 virginica
    print(w1.T)
    print(w2.T)
    print(w3.T)
    print("loss: ", round(loss1, 3), round(loss2, 3), round(loss3, 3))
    # 给定测试集，分别丢给三个分类器
    test_x, test_y = read_data(test=True)
    # print(test_x)
    # print(test_y)

    # 得到一个[True, False, ..., True] 数组
    res1 = test(test_x, test_y, w1)  # 表示是否被分为setosa
    res2 = test(test_x, test_y, w2)  # 同理，是够被分为 versicolor
    res3 = test(test_x, test_y, w3)  # 同理
    res = np.r_[res1, res2, res3]    # 合并成一个

    idx = np.argmax(res, axis=0)     # 相当于投票
    # 投票决定
    # res1 = res1 > 0
    # res2 = res2 > 0
    # res3 = res3 > 0

    # res = np.r_[res1, res2, res3]
    # 三个分类器的结果，然后对三个分类器投票
    # print(res)

    each_len = int(len(test_x) / 3)
    cls1 = idx[0:each_len]
    cls2 = idx[each_len:2 * each_len]
    cls3 = idx[2 * each_len:3 * each_len]

    # 检查 setosa
    # cla[cls==0] 0表示第0类也就是 setosa
    print(cls1)
    ac_rate1 = len(cls1[cls1 == 0]) / len(cls1)
    print("Accuracy rate of setosa     ", round(ac_rate1, 3))

    # 检查versicolor
    print(cls2)
    ac_rate2 = len(cls2[cls2 == 1]) / len(cls1)
    print("Accuracy rate of versicolor ", round(ac_rate2, 3))

    # 检查virginica
    print(cls3)
    ac_rate3 = len(cls3[cls3 == 2]) / len(cls1)
    print("Accuracy rate of virginica  ", round(ac_rate3, 3))
