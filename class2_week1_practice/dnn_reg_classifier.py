"""
@Time : 2019/10/3 19:41 PM
@Author : bjjoy2009
dnn正则化二分类实验
"""

import scipy.io
import matplotlib.pyplot as plt

from class2_week1_practice.dnn_v3 import DNN
from utils import plot_decision_boundary


def load_2D_dataset():
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    # plt.scatter(train_X[0, :], train_X[1, :], c=train_Y.ravel(), s=40, cmap=plt.cm.Spectral)
    return train_X, train_Y, test_X, test_Y


if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = load_2D_dataset()

    layer_dims = [train_X.shape[0], 20, 3, 1]
    dnn = DNN(X=train_X, Y=train_Y, layer_dims=layer_dims, max_iter=30000, alpha=0.3,
              print_loss=True, print_loss_iter=10000, lambd=0.7, keep_prob=0.86)
    # dnn.fit()
    # dnn.fit_regularization()
    dnn.fit_dropout()
    accuracy = dnn.score(train_X, train_Y)
    print('train:', accuracy)
    print('test', dnn.score(test_X, test_Y))
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: dnn.predict(x.T), train_X, train_Y, 'Model dropout')