"""
@Time : 2019/10/8 19:31 PM
@Author : bjjoy2009
dnn优化算法实验
"""
import numpy as np
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt

from class2_week2_practice.dnn_v4 import DNN
from utils import plot_decision_boundary


def load_dataset():
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    return train_X, train_Y


if __name__ == '__main__':
    train_X, train_Y = load_dataset()

    layer_dims = [train_X.shape[0], 5, 2, 1]
    dnn = DNN(X=train_X, Y=train_Y, layer_dims=layer_dims, epochs=10000, alpha=0.0007,
              print_loss=True, print_loss_iter=1000, initialization="he", optimizer='momentum')
    dnn.fit()
    # dnn.fit_regularization()
    # dnn.fit_dropout()
    accuracy = dnn.score(train_X, train_Y)
    print('train:', accuracy)
    # print('test', dnn.score(test_X, test_Y))
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    plot_decision_boundary(lambda x: dnn.predict(x.T), train_X, train_Y, 'momentum')