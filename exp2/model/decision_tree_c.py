# -*- coding: UTF-8 -*-


import time
import numpy as np
from data_process import split_multi_dataset
from collections import Counter
from draw_tree import draw

import copy
import csv
from itertools import islice

# 自动划分


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
    return x, y, labels

# 读取属性名


def read_labels(path='data/Iris.csv'):
    f = open(path, encoding='utf-8')
    labels = f.readline().strip()

    return labels.split(',')[1:-1]


# 找到最大的分类对应的class
def find_main_class(y):
    counter = Counter(list(y))
    counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return counter[0][0]


def trans2list(x):
    # print(x)
    return list(x.reshape(1, len(x))[0])

# 计算信息熵


def cal_Ent_D(y):
    Ent = 0
    y_set = set(y)
    for k in y_set:
        p = len([e for e in y if e == k]) / len(y)
        Ent += p * np.log2(p)
    return -Ent

# 得到最优属性


def get_best_feature(x, y):
    # 1. 计算数据集D的信息增益
    Ent_D = cal_Ent_D(y)
    if len(x) == 0:
        return -1, -1
    features_num = len(x[0])
    # 最优值
    max_Gain_D = -1
    best_fid = -1
    best_value = -1
    for fid in range(features_num):
        best_valuei = None
        features = [e[fid] for e in x]
        features_set = set(features)
        # 对属性的取值进行排序
        sorted_features = sorted(list(features_set))
        # print(sorted_features)
        min_ent = 1e6
        for i in range(len(sorted_features) - 1):
            # 两两取均值，作为候选划分点
            value = (float(sorted_features[i]) +
                     float(sorted_features[i+1])) / 2
            # print("part value ", value)
            # 左右子树
            data_left, y_left = get_sub_data(x, y, value, fid, 'L')
            data_right, y_right = get_sub_data(x, y, value, fid, 'R')
            prob_left = len(data_left) / float(len(x))
            prob_right = len(data_right) / float(len(x))

            # 分别计算左右子树的信息增益，取和
            ent = prob_left * cal_Ent_D(y_left) + \
                prob_right * cal_Ent_D(y_right)
            # 求得属性a下的最优候选房划分点
            if ent < min_ent:
                min_ent = ent
                best_valuei = value
                # print("best value i ", best_valuei)
        # 更新，求得最优的属性
        Gain_D = Ent_D - min_ent

        if Gain_D > max_Gain_D:
            max_Gain_D = Gain_D
            best_fid = fid
            best_value = best_valuei
            # print("fid[{}] part value[{}] GainD[{}]".format(fid, best_value, max_Gain_D))
            # print("fid[", str(fid), "] part value ", str(best_value), "")

    return best_fid, round(best_value, 5)


# 得到子数据集
def get_sub_data(x, y, value, fid, dir='L'):
    resx = []
    resy = []
    for i, e in enumerate(x):
        # 二叉树，左子树的值小于root、右子树的值大于等于value
        if e[fid] < value and dir == 'L' or e[fid] >= value and dir == 'R':
            resx.append(e)
            resy.append(y[i])
    return resx, resy


global cnt
cnt = 0


def tree_genrate(xo, yo, labelso):
    x = copy.deepcopy(xo)
    # print(x)
    y = copy.deepcopy(yo)
    labels = copy.deepcopy(labelso)
    # print("labels: ", labels)
    classDict = set(y)

    # 所有的样本都是一类样本
    if len(classDict) == 1:
        # print("只有一类了")
        return y[0]

    # 属性只剩下一类，返回最多的类
    if len(labels) == 1:
        # print("标签只有一种了")
        return find_main_class(y)

    fid, best_value = get_best_feature(x, y)

    # 等价于只剩下一类
    if len(x) == 0:
        return "NULL"

    # 无法继续划分，精度不够的情况下
    if best_value == -1:
        return find_main_class(y)

    # 定义节点名称 属性 < value
    label = labels[fid] + '<' + str(best_value)
    tree_dict = {label: {}}

    attrlist = [sample[fid] for sample in x]
    subx_left, suby_left = get_sub_data(x, y, best_value, fid, 'L')
    subx_right, suby_right = get_sub_data(x, y, best_value, fid, 'R')
    sublabels = labels[:]
    # 左右子树
    tree_dict[label]['YES'] = tree_genrate(subx_left, suby_left, sublabels)
    tree_dict[label]['NO'] = tree_genrate(subx_right, suby_right, sublabels)

    return tree_dict


# 搜索
def search(x, tree, labels):
    import types
    tree_key = list(tree.keys())[0]
    tree_dict = tree[tree_key]
    # 找到<在list中的index
    less_index = tree_key.find('<')
    # 属性名字
    test_key_index = labels.index(tree_key[:less_index])
    test_key = x[test_key_index]
    sub_left_tree = tree_dict['YES']
    sub_right_tree = tree_dict['NO']
    # 属性值
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
    return y_pred


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

    main_class = find_main_class(trainy)
    # print(main_class)
    tree = tree_genrate(trainx, trainy, labels)
    print(tree)

    draw(tree, "dvTreeIrisTrainTest", "png")
    ypred = predict(testx, tree, labels)
    res = np.array(testy) == np.array(ypred)
    print("准确率 ", res[res == True].size / res.size)


if __name__ == '__main__':
    test()
