import numpy as np


def split_dataset(x, y, rate):
    """数据集划分 

    Args:
        x ([type]): 数据
        y ([type]): 标签
        rate ([type]): training set 比例

    Returns:
        [ndarry]: 划分后的training x, training y, test x, test y
    """
    data_len = int(len(y) * rate)
    positive_index = np.where(1 == y)[0]
    negative_index = np.where(0 == y)[0]
    # print(positive_index)
    positive_y = y[positive_index]
    positive_x = x[positive_index, ]
    negative_x = x[negative_index, ]
    negative_y = y[negative_index]

    positive_len = int(len(positive_y) * rate)
    negative_len = int(len(negative_y) * rate)

    # print(positive_x)
    # print(positive_y)
    # print("*" * 20)
    # print(negative_x)
    train_x = np.r_[positive_x[0:positive_len, ], negative_x[0:negative_len, ]]
    train_y = np.r_[positive_y[0:positive_len], negative_y[0:negative_len]]

    test_x = np.r_[positive_x[positive_len:, ], negative_x[negative_len:, ]]
    test_y = np.r_[positive_y[positive_len:], negative_y[negative_len:]]

    return train_x, train_y, test_x, test_y


def split_multi_dataset(x, y, rate):
    """多分类的数据集划分

    Args:
        x (ndarray): 数据
        y (ndarray): 标签
        rate (float): 划分比例

    Returns:
        ndarray: training x， training y， test x， test y
    """
    data_len = int(len(y) * rate)
    attribute_x = []
    attribute_y = []
    attribute_arr = np.unique(y)
    for i in attribute_arr:
        index = np.where(i == y)[0]
        # print("index")
        # print(index)
        attribute_x.append(x[index, ])
        attribute_y.append(y[index])

    # print(attribute_x)

    attribute_len = int(attribute_y[0].shape[0] * rate)

    test_x = test_y = np.zeros((1, x.shape[1]))
    test_y = np.zeros((1, 1))
    # print(test_x.shape)
    for i in range(len(attribute_arr)):
        # print(test_x.shape)
        # print(attribute_x[i][attribute_len:, ].shape)
        test_x = np.r_[test_x, attribute_x[i][attribute_len:, ]]
        test_y = np.r_[test_y, attribute_y[i][attribute_len:, ]]

    return test_x[1:, ], test_y[1:, ]
