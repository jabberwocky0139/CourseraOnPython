import numpy as np
import matplotlib.pyplot as plt


class Ex1Data(object):
    def __init__(self, filename):
        self.x = []
        self.y = []
        self.alpha = 0.01
        self.filename = filename

    def input_data(self):
        with open(self.filename, 'r') as f:
            for line in f.readlines():
                a, b = line.split(',')
                self.x.append(float(a))
                self.y.append(float(b))

        self.x = np.array(self.x)  # numpyに変換
        self.y = np.array(self.y)  # numpyに変換

        self.m = len(self.x)
        self.X = np.hstack((np.ones((self.m, 1)),
                            np.reshape(self.x, (self.m, 1))))

    def plot_data(self, theta):
        plt.plot(self.x, self.y, 'x')
        plt.xlabel('population')
        plt.ylabel('profit')
        plt.plot(self.x, theta[0] + theta[1]*self.x)
        plt.show()


class CostFunction(object):
    def __init__(self, data):
        self.data = data
        self.theta = np.array([0.0, 0.0])  # 初期値
        self.J = None

    def hypothesis(self):
        return np.matmul(self.data.X, self.theta)

    def compute_J(self):
        h = self.hypothesis()
        self.J = 0.5 * np.mean((h - self.data.y)**2)

        return self.J

    def compute_grad(self):
        h = self.hypothesis()
        self.theta -= self.data.alpha / self.data.m * np.matmul(h - self.data.y, self.data.X)


class Interface(object):
    def __init__(self):
        self.data = Ex1Data(filename='ex1data1.txt')
        self.cost = CostFunction(self.data)

    def compute(self):
        # データ入力
        self.data.input_data()

        # コスト関数
        print('cost before: {0}'.format(self.cost.compute_J()))

        # 最急降下法
        for _ in range(1500):
            self.cost.compute_grad()

        # コスト関数
        print('cost after: {0}'.format(self.cost.compute_J()))

        print('theta: {0}'.format(self.cost.theta))

        self.data.plot_data(self.cost.theta)


if __name__ == '__main__':

    hoge = Interface()
    hoge.compute()
