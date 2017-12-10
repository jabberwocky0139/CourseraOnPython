import numpy as np
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt


class Ex2Data(object):
    def __init__(self, filename):
        self.X = []
        self.y = []
        self.m = None
        self.filename = filename

    def input_data(self):
        raw_data = np.loadtxt(self.filename, delimiter=',')  # numpyに直接格納
        self.y = raw_data[:, 2]
        self.m = len(self.y)

        self.X = raw_data[:, :2]
        self.X = np.column_stack((np.ones(self.m), self.X))

    def plot_data(self, theta):
        X1 = self.X[self.y == 1]
        X2 = self.X[self.y == 0]
        plt.plot(X1[:, 1], X1[:, 2], 'x')
        plt.plot(X2[:, 1], X2[:, 2], '+')
        plt.plot(self.X[:, 1], - (theta[0] + theta[1] * self.X[:, 1]) / theta[2])  # リニア
        plt.show()


class CostFunction(object):
    def __init__(self, data):
        self.data = data
        self.theta = np.array([-24, 0.2, 0.2])  # 初期値
        self.J = None

    def sigmoid(self, z):
        return (1 + np.exp(-z))**-1

    def hypothesis(self, theta):
        return self.sigmoid(np.matmul(self.data.X, theta))

    def compute_J(self, theta):
        h = self.hypothesis(theta)
        y = self.data.y
        self.J = np.mean(-y * np.log(h) - (1 - y) * np.log(1 - h))

        return self.J

    def optimize_theta(self):
        self.theta = fmin_cg(self.compute_J, self.theta, maxiter=1000)


class Interface(object):
    def __init__(self):
        self.data = Ex2Data(filename='ex2data1.txt')
        self.cost = CostFunction(self.data)

    def compute(self):
        # データ入力
        self.data.input_data()

        # 最小値問題
        self.cost.optimize_theta()

        # コスト関数
        print('cost after: {0}'.format(self.cost.compute_J(self.cost.theta)))

        print('theta: {0}'.format(self.cost.theta))

        self.data.plot_data(self.cost.theta)


if __name__ == '__main__':

    hoge = Interface()
    hoge.compute()
