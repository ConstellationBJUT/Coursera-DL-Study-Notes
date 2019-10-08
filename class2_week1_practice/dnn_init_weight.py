"""
@Time : 2019/10/4 19:31 PM
@Author : bjjoy2009
dnn初始化权重实验
"""
import numpy as np
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt

from class2_week1_practice.dnn_v3 import DNN
from utils import plot_decision_boundary


def load_dataset():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    # Visualize the data
    # plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y


if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = load_dataset()

    layer_dims = [train_X.shape[0], 10, 5, 1]
    dnn = DNN(X=train_X, Y=train_Y, layer_dims=layer_dims, max_iter=15000, alpha=0.01,
              print_loss=True, print_loss_iter=1000, lambd=0.7, keep_prob=0.86, initialization="he")
    dnn.fit()
    # dnn.fit_regularization()
    # dnn.fit_dropout()
    accuracy = dnn.score(train_X, train_Y)
    print('train:', accuracy)
    print('test', dnn.score(test_X, test_Y))
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: dnn.predict(x.T), train_X, train_Y, 'my random init')