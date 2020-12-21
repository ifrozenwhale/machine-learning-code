# -*- coding: UTF-8 -*-


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


# 寻找最多的一类class
def find_main_class(y):
    counter = Counter(list(y))
    counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return counter[0][0]


def trans2list(x):
    # print(x)
    return list(x.reshape(1, len(x))[0])


# 计算信息熵 Ent
def cal_Ent_D(y):
    Ent = 0
    y_set = set(y)
    for k in y_set:
        p = len([e for e in y if e == k]) / len(y)
        Ent += p * np.log2(p)
    return -Ent


def cal_Ent_Dv(features, y):
    """    
    计算属性a对样本集D进行划分的信息增益计算公式中的

    Sum |Dv|/|D| *Ent(Dv), for v=1..V

    features 为属性a的取值
    """

    Ent = 0
    V = set(features)
    for v in V:
        fv = [e for e in features if e == v]  # 找到属性a取值为v的那一列数据
        index = [i for (i, e) in enumerate(features)
                 if e == v]  # 找到属性a取值为v的数据的index
        D = [y[i] for i in index]  # 从标签y中拿到对应行的数据
        e = cal_Ent_D(D)  # 计算信息增益Ent Dv
        Ent += len(fv) / len(features) * e
    return Ent


def get_best_feature(x, y):
    """
    得到最优属性划分
    Args:
        x : 属性数据
        y : 标签y
    """
    # 1. 计算数据集D的信息增益
    Ent_D = cal_Ent_D(y)

    # 属性类型的数量
    features_num = len(x[0])
    # 最优值初始化
    max_Gain_D = -1
    best_fid = -1
    for fid in range(features_num):
        # print("*" * 50)
        # fid 列属性
        features = [e[fid] for e in x]
        # print("features " + str(lfeatures))
        Gain_D = Ent_D - cal_Ent_Dv(features, y)
        # print("infogain ", Gain_D)

        # 更新信息熵
        if Gain_D >= max_Gain_D:
            max_Gain_D = Gain_D
            best_fid = fid
    return best_fid, max_Gain_D


def get_sub_data(x, y, label, fid):
    """
    得到划分的子数据集
    """
    resx = []
    resy = []
    # 去除fid列的数据，重组数据x和标签y
    for i, e in enumerate(x):
        if e[fid] == label:
            vec = e[:fid]
            vec.extend(e[fid+1:])
            resx.append(vec)
            resy.append(y[i])
    # print(res)
    return resx, resy


def tree_genrate(xo, yo, labelso):
    """ 
    生成决策树
    """
    # 传参为list时，为了防止造成原数据改变，深拷贝一份
    x = copy.deepcopy(xo)
    y = copy.deepcopy(yo)
    labels = copy.deepcopy(labelso)
    # print("labels: ", labels)
    classDict = set(y)
    # 如果剩余所有的样本的分类一致
    if len(classDict) == 1:
        # print("只有一类了")
        return y[0
    # 如果属性的类型只剩下一种
    if len(labels) == 1:
        return find_main_class(y)
    # 得到最优属性的列id和信息增益
    fid, Gain_D = get_best_feature(x, y)
    # 得到fid对应的属性的名字
    label = labels[fid]
    # print("best ", label)
    # 初始化一个子树根节点
    tree_dict = {label: {}}
    # 删除用过的，为了保证fid的对应关系
    del (labels[fid])
    attrlist = [sample[fid] for sample in x]
    for i in set(attrlist):  # 对每一个属性
        # 得到子数据集
        subx, suby = get_sub_data(x, y, i, fid)
        sublabels = labels[:]
        tree_dict[label][i] = tree_genrate(subx, suby, sublabels)
    return tree_dict


def search(x, tree, labels):
    """ 
    数的遍历搜索
    """
    import types
    # 得到树的根节点key（名字）
    tree_key = list(tree.keys())[0]
    # 树的子树
    tree_dict = tree[tree_key]
    # 得到根节点属性名字对应的index
    test_key_index = labels.index(tree_key)
    # 得到测试集在该节点下的属性取值
    test_key = x[test_key_index]
    sub_tree = tree_dict[test_key]
    # 得到测试集在该节点下的子树

    # 如果仍是树，继续递归
    if isinstance(sub_tree, dict):
        # print("key ", test_key)
        # print("subtree ", sub_tree)
        return search(x, sub_tree, labels)
    # 否则表示为叶节点，找到了
    else:
        return sub_tree


def predict(x_list, y_list, tree, labels):
    """
    预测
    """
    print()
    y_pred = []
    for i in range(len(x_list)):
        y_pred.append(search(x_list[i], tree, labels))
        # print()
    print(y_list)
    print(y_pred)
    res = np.array(y_list) == np.array(y_pred)
    print("准确率 ", res[res == True].size / res.size)


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
    p = "data/watermelon20_2.csv"
    # read_data2(p)

    # labels = read_labels(path=p)

    x, y, labels = read_data2(path=p)
    trainx = x[0:10]
    trainy = y[0:10]
    testx = x[10:]
    testy = y[10:]

    # for i in trainx:
    #     print(i)
    # print(y)
    # trainx, trainy, testx, testy = read_data(path=p, rate=0.7)
    # print(trainy)
    main_class = find_main_class(trainy)
    # print(main_class)
    tree = tree_genrate(trainx, trainy, labels)
    print(tree)

    draw(tree, "devisionTreeTrain", "png")
    predict(testx, testy, tree, labels)


if __name__ == '__main__':
    test()
