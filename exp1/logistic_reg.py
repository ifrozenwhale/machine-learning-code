import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    """sigmod函数
    """
    return 1.0 / (1 + np.exp(-z))


class LogisticRegression:
    def __init__(self, max_iter, learning_rate):
        self.max_iter = max_iter
        self.leaning_rate = learning_rate

    def fit(self, x, y):
        """进行对数逻辑回归的参数训练

        Args:
            x (ndarray): 数据集x，一个数据样本是一行
            y (naddy): 标签y，一个样本是一行

        Returns:
            ndarray, float: 返回线性模型的参数w以及损失函数值
        """
        c = np.ones((len(x), 1))
        xx = np.c_[c, x]  # 拼接一列1
        ww = np.ones((len(x[0]) + 1, 1))  # 合并了w和b
        ww = np.random.rand(len(x[0]) + 1, 1)  # 随机赋初值

        dy = -1
        dy_all = []
        for i in range(self.max_iter):
            y_pred = sigmoid(xx.dot(ww))  # 预测值，y = 1 / (1 + e^(-z))
            # 经过推导，dl/dz = y^ - y
            dy = y_pred - y
            # 记录误差
            dy_all.append(dy.sum())
            # 梯度下降迭代
            ww = ww - self.leaning_rate * xx.transpose().dot(dy)
        self.plot(dy_all)

        return ww, dy.sum()

    def plot(self, y):
        plt.figure()
        x = np.linspace(0, self.max_iter, self.max_iter)
        plt.plot(x, y)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Convergence curve")
        plt.show()
