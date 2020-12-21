# -*- coding: UTF-8 -*-
# 基于决策树的回归树

import time
import numpy as np
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
    x = data[:, 1:n].astype(float)
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
                digl = [float(e) for e in line[1:-1]]
                x.append(digl)
                y.append(line[-1])
                # y.append("YES" if line[-1] == '1' else "NO")

    return x, y, labels


def read_labels(path='data/Iris.csv'):
    f = open(path, encoding='utf-8')
    labels = f.readline().strip()

    return labels.split(',')[1:-1]


def trans2list(x):
    # print(x)
    return list(x.reshape(1, len(x))[0])


def cal_err(x, y):
    # 计算残差平方和
    err = 0
    ymean = np.mean(y)
    for i in range(len(y)):
        err += (y[i] - ymean) ** 2
    return err


def get_best_feature(x, y):
    # 得到最优划分
    # 如果数据集为空，返回 -1
    if len(x) == 0:
        return -1, -1
    # 属性类型数量
    features_num = len(x[0])
    # 最优值
    global_min_err = 1e6
    best_value = -1
    best_fid = -1
    for fid in range(features_num):
        best_valuei = None
        features = [e[fid] for e in x]
        features_set = set(features)
        # 属性的取值排序
        sorted_features = sorted(list(features_set))
        # print(sorted_features)
        min_err = 1e6
        for i in range(len(sorted_features) - 1):
            # 找候选划分点
            value = (float(sorted_features[i]) +
                     float(sorted_features[i+1])) / 2
            # print("part value ", value)
            data_left, y_left = get_sub_data(x, y, value, fid, 'L')
            data_right, y_right = get_sub_data(x, y, value, fid, 'R')
            # 计算左右子树的误差平方和
            err = cal_err(data_left, y_left) + cal_err(data_right, y_right)
            # 找到属性内的最优划分点
            if err < min_err:
                min_err = err
                best_valuei = value
                # print("best value i ", best_valuei)

        # print()
        # 找到最优划分属性
        if global_min_err > min_err:
            global_min_err = min_err
            best_fid = fid
            best_value = best_valuei

    return best_fid, round(best_value, 10)


def get_sub_data(x, y, value, fid, dir='L'):
    resx = []
    resy = []
    for i, e in enumerate(x):
        if e[fid] < value and dir == 'L' or e[fid] > value and dir == 'R':
            resx.append(e)
            resy.append(y[i])
    return resx, resy


def tree_genrate(xo, yo, labelso):
    x = copy.deepcopy(xo)
    # print(x)
    y = copy.deepcopy(yo)
    labels = copy.deepcopy(labelso)
    # print("labels: ", labels)

    fid, best_value = get_best_feature(x, y)
    # 如果无法分割 返回叶节点的值的均值
    if best_value == -1:
        return str(np.mean(y))
    # print("best value ", best_value)

    # print("best ", fid)
    label = labels[fid] + '<' + str(best_value)
    tree_dict = {label: {}}

    attrlist = [sample[fid] for sample in x]
    subx_left, suby_left = get_sub_data(x, y, best_value, fid, 'L')

    subx_right, suby_right = get_sub_data(x, y, best_value, fid, 'R')
    sublabels = labels[:]
    tree_dict[label]['YES'] = tree_genrate(subx_left, suby_left, sublabels)
    tree_dict[label]['NO'] = tree_genrate(subx_right, suby_right, sublabels)

    return tree_dict


def search(x, tree, labels):
    import types
    tree_key = list(tree.keys())[0]

    tree_dict = tree[tree_key]
    less_index = tree_key.find('<')
    test_key_index = labels.index(tree_key[:less_index])
    test_key = x[test_key_index]
    sub_left_tree = tree_dict['YES']
    sub_right_tree = tree_dict['NO']

    value = float(tree_key[less_index+1:])
    if x[test_key_index] < value:
        if isinstance(sub_left_tree, dict):
            return search(x, sub_left_tree, labels)
        else:
            return sub_left_tree
    else:
        if isinstance(sub_right_tree, dict):
            return search(x, sub_right_tree, labels)
        else:
            return sub_right_tree


def predict(x_list, tree, labels):
    print()
    y_pred = []
    for i in range(len(x_list)):
        # print("x ", x_list[i])
        y_pred.append(search(x_list[i], tree, labels))
        # print()
    # print(y_list)
    # print(y_pred)
    # print("准确率 ", res[res == True].size / res.size)
    return y_pred


def evaluate(ypred, y):
    SYY = 0
    SSR = 0

    ymean = np.mean(y)
    ypred = [float(e) for e in ypred]
    for i in range(len(y)):
        # print("i=", i)
        SYY += (y[i] - ypred[i]) ** 2
        SSR += (y[i] - ymean) ** 2
    return 1 - SYY / SSR


def test_read_data():
    train_x, train_y, test_x, test_y = read_data()
    print("test x")
    print(test_x)
    print("test y")
    print(test_y)

    print("train x")
    print(train_x)
    print("train y")
    print(train_y)


def test():
    p = "data/Iris.csv"
    # read_data2(p)

    # labels = read_labels(path=p)

    # x, y, labels = read_data2(path=p)
    # trainx = x[:]
    # trainy = y[:]
    # testx = x[10:]
    # testy = y[10:]
    trainx, trainy, testx, testy = read_data(p, rate=0.7)
    labels = read_labels(p)

    # print(main_class)
    tree = tree_genrate(trainx, trainy, labels)
    print(tree)

    draw(tree, "dvTreeIrisTrainTest", "png")
    ypred = predict(testx, testy, tree, labels)
    res = np.array(testy) == np.array(ypred)
    print("准确率 ", res[res == True].size / res.size)


if __name__ == '__main__':
    test()
