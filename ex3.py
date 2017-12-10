import scipy.io as spio
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt
import numpy as np
from random import gauss
from itertools import cycle


Lambda = 8
eps = 1e-10
iter_label = cycle([10] + [i for i in range(1, 10)])  # 10, 1, 2, 3,..., 9, 10, 1, 2,...


class Data(object):
    def __init__(self):
        self.X = None  # トレーニングデータ
        self.label_num = 10  # ラベル数
        self.X_num = [None for _ in range(self.label_num)]  # ラベルごとのトレーニングデータ
        self.y = None  # 出力期待値
        self.theta = np.array([None]*self.label_num)  # 学習パラメータ(0-9の10パターン)

    def input_data(self, filename):
        mat = spio.loadmat(filename)
        self.X, self.y = np.array(mat['X']), np.array(mat['y'])
        m, n = self.X.shape

        self.X = np.column_stack((np.ones(m), self.X))
        self.theta = np.array([[gauss(0, 0.1)] * (n + 1) for _ in range(10)], dtype=float)

        for i in range(self.label_num):
            self.X_num[i] = self.X[self.y.T[0] == next(iter_label)]

        return self.X, self.y, self.theta

    def plot_data(self, X, label=0):
        plt.imshow(X[label][1:].reshape(20, 20).T, 'gray', vmin=0, vmax=1.0)
        plt.show()


class Calc(object):
    def __init__(self, X, y, theta):
        self.X = X
        self.y = y
        self.theta = theta
        self.m = self.X.shape[0]

    def _sigmoid(self, z):
        return (1 + np.exp(-z))**-1

    def hypothesis(self, theta, X):
        return self._sigmoid(np.matmul(X, theta))

    def _cost(self, theta, *args):
        X, y, label = args
        h = self.hypothesis(theta, X)
        y = np.array(y == label, dtype=float).reshape(1, self.m)[0]
        J = np.mean(-y * np.log(h + eps) - (1 - y) * np.log(1 - h + eps)) + Lambda / (2 * self.m) * np.sum(theta ** 2)
        return J

    def _gradient(self, theta, *args):
        X, y, label = args
        h = self._sigmoid(np.matmul(X, theta))
        y = np.array(y == label, dtype=float).reshape(1, self.m)[0]
        tmp_theta, theta[0] = theta[0], 0.0
        grad = np.matmul((h - y), X) / self.m + Lambda / self.m * theta
        theta[0] = tmp_theta
        return grad

    def optimize_theta(self, theta, label):
        theta = fmin_cg(f=self._cost, x0=theta, args=(self.X, self.y, label), fprime=self._gradient)
        return theta


v = Data()
X, y, theta = v.input_data(filename='ex3data1.mat')
c = Calc(X, y, theta)

for i in range(v.label_num):
    label = next(iter_label)
    print('iter_label={0}'.format(label))
    theta[i] = c.optimize_theta(theta=theta[i], label=label)
