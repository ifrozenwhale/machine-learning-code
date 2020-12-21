import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from model.reg_tree import tree_genrate, predict, evaluate
from draw_tree import draw
import csv


def load_data(path="data/advertising.csv"):
    df_ad = pd.read_csv(path)
    labels = df_ad.columns.tolist()

    x = df_ad.iloc[:, 0:-1].values.tolist()
    y = df_ad.iloc[:, -1].values.tolist()

    return x, y, labels


def split_dataset(x, y, rate):
    n = int(len(y) * rate)
    return x[0:n], y[0:n], x[n:], y[n:]


if __name__ == "__main__":
    x, y, labels = load_data("data/advertising.csv")
    trainx, trainy, testx, testy = split_dataset(x, y, 0.7)
    # print(trainy)
    tree = tree_genrate(trainx, trainy, labels)
    ypred = predict(testx, tree, labels)
    ypred = [float(e) for e in ypred]

    R2 = evaluate(ypred, testy)

    print("R^2: ", R2)
    xx = range(0, len(ypred))
    plt.plot(xx, testy, '-b')
    plt.plot(xx, ypred, '-r')
    plt.legend(['observed value', 'predict value'])
    plt.title("advertising predict with regression tree")
    plt.savefig("./img/ad_result.png", dpi=600)
    plt.show()
