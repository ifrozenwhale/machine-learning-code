from csv import reader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt
import math


def sigmoid(z):
    """激活函数(Sigmoid)：f(z) = Sigmoid(z)
    Returns:
        [float]: [函数值]
    """
    return 1.0 / (1.0 + np.exp(-z))


def d_sigmod(y):
    """Sigmoid激活函数求导
    """

    return y * (1.0 - y)


def calc_derrors(o, y):
    """计算d误差，使用简单的均方误差
    E = 1/2 * (y - o)^2
    求导得到 dE = y - o
    """
    return o - y


def calc_errors(o, y):
    """计算误差，使用简单的均方误差
    E = 1/2 * (y - o)^2

    """
    return 0.5 * np.sum((o - y) ** 2)


class BP_network:
    def __init__(self, sizes):
        # 网络层数
        self.layers_num = len(sizes)

        # 网络结构
        # 假设size为[2 3 2]表示输入层2个节点，隐含层3个节点，输出层2个节点
        self.sizes = sizes
        # 构造初始的权重矩阵W
        # 输入层到隐含层： W为2*3（2行3列）的矩阵
        # 隐含层到输出层： W为3*2（3行2列）的矩阵
        # 即对于每两层L1（M个节点）和L2（N个节点）的权重矩阵，W.T为一个M*N的矩阵

        self.weights = list()
        for M, N in zip(sizes[:-1], sizes[1:]):
            self.weights.append(np.random.randn(M, N)/np.sqrt(M))
        # 构造隐含层和输出层神经元的阈值
        self.threshold = [np.random.randn(1, x) for x in sizes[1:]]

        # 中间结果记录
        self.losses = []
        self.accs = []

    def bp(self, x, y):
        # 更新的权重矩阵变化量
        dw = [np.zeros(w.shape) for w in self.weights]
        # 更新后的阈值变化量
        db = [np.zeros(b.shape) for b in self.threshold]
        # Y，保存网络的每一层的输出
        Y = [np.atleast_2d(x)]

        # 前向传播
        for l in range(0, self.layers_num-1):
            # 计算当前层的输出
            net = Y[l] @ self.weights[l] - self.threshold[l]
            # print(net.shape)
            o = sigmoid(net)
            Y.append(o)

        # 反向传播
        # 计算输出层权重矩阵的梯度g
        g = calc_derrors(Y[-1], y) * d_sigmod(Y[-1])
        dw[-1] = Y[-2].T @ g
        db[-1] = -g
        eh = g
        # 计算中间层的权重矩阵的梯度e
        for l in range(self.layers_num-2, 0, -1):
            eh = (eh @ self.weights[l].T) * d_sigmod(Y[l])
            db[l-1] = -eh
            dw[l-1] = Y[l-1].T @ eh
        error = calc_errors(Y[-1], y)
        return (dw, db, error)

    def update_parameters(self, x, y, eta):
        # 更新w和b
        dws, dbs, error = self.bp(x, y)
        self.weights = [w - eta * dw for w, dw in zip(self.weights, dws)]
        self.threshold = [b - eta * db for b,
                          db in zip(self.threshold, dbs)]
        return error

    def train(self, trainx, trainy, epochs=5, learning_rate=0.2):
        """使用梯度下降算法进行训练
        """
        acc = []
        n = int(0.7 * len(trainx))
        tx = trainx[0:n, ]
        ty = trainy[0:n, ]
        vx = trainx[n:, ]
        vy = trainy[n:, ]
        for epoch in range(epochs):  # 训练epochs个回合
            err = 0
            for x, y in zip(trainx, trainy):
                err += self.update_parameters(x, y, learning_rate)

            _, acc = self.evaluate(trainx, trainy)
            self.accs.append(acc)
            self.losses.append(err)
            if epoch % 100 == 0:
                print("[epoch {} ] acc_rate: {:.5f}, loss: {:.5f}".format(
                    epoch, acc, err))

    def predict(self, x):
        """使用bp网络进行预测，实际是一次前向计算
        """
        for w, b in zip(self.weights, self.threshold):
            x = sigmoid(np.dot(x, w) - b)
        return x

    def evaluate(self, testx, testy):
        res = [(np.argmax(self.predict(x)), np.argmax(y))
               for x, y in zip(testx, testy)]
        # 返回正确识别的个数
        return res, sum(int(ypred == y) for (ypred, y) in res) / len(testy)

    def plot(self):
        fig = plt.figure()
        # Loss
        ax1 = fig.add_subplot(111)
        ax1.plot(self.losses, '#B22222', label='loss')
        plt.xlabel('data id')
        maxloss = np.max(self.losses) * 1.2
        ax1.set_yticks(np.arange(0, maxloss, maxloss/20))
        ax1.set_ylabel('')
        # Acc rate
        ax2 = ax1.twinx()
        ax2.plot(self.accs, '#006400',
                 label='acc rate')
        ax2.set_yticks(np.arange(0, 1, 0.1))
        ax2.set_ylabel('acc rate')

        plt.legend(loc=2)
        plt.title('Loss and acc rate in the process of training')
        plt.savefig('训练曲线.png', dpi=600, bbox_inches='tight')
        plt.show()
