import numpy as np
import csv
import matplotlib.pyplot as plt

from logistic_reg import LogisticRegression
from data_process import split_dataset

# plt.style.use('seaborn')


def plot(x, labels, w, name):
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=labels.reshape(
        1, len(labels)), cmap=plt.cm.Spectral)

    xx = np.linspace(0, 1, 100)
    yy = -w[0] / w[2] - w[1] / w[2] * xx  # 由于只有2个 属性，转化为截距式，为了作图
    plt.plot(xx, yy)

    plt.xlabel("Density")
    plt.ylabel("Sugar content")
    plt.title(name)
    plt.savefig("./img/{}.png".format(name), dpi=600)
    plt.show()


def read_data(path='data/watermelon_3a.csv'):
    with open(path, encoding='utf-8') as f:
        data = np.loadtxt(f, delimiter=",", skiprows=1)
        x = data[:, 1:3]
        y = data[:, 3]
    return split_dataset(x, y.reshape(len(y), 1), 0.5)


def test(x, labels, w):
    labels = labels.reshape(1, len(labels))
    # print(w)
    # print(x)
    # print(labels)
    # c = np.ones((len(x), 1))
    # xx = np.c_[c, x]
    # print(xx)
    # print(xx.dot(w))
    y = -w[0] / w[2] - w[1] / w[2] * x[:, 0]

    result = (labels == (x[:, 1] < y))
    return result[0]


if __name__ == '__main__':
    logistic_reg = LogisticRegression(5000, 0.02)
    train_x, train_y, test_x, test_y = read_data()
    # print(train_x)
    # print(train_y)
    w, loss = logistic_reg.fit(train_x, train_y)
    print("w: ")
    print(w)
    print("loss: " + str(loss))
    plot(train_x, train_y, w, "(watermelon) logistic regression on training set")
    plot(test_x, test_y, w, "(watermelon) logistic regression on test set")

    result = test(test_x, test_y, w)
    print("result:")
    print(result)
    print("accurate: %f" % (len(np.where(result == True)[0]) / len(result)))
