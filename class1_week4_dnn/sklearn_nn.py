"""
@Time : 2019/9/27 19:41 PM
@Author : guoxiaoming
调用sklearn神经网络分类，作对比实验
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from utils import *


def multi_class_nn():

    X, Y = load_planar_dataset()
    # 划分训练集与测试集
    x_train, x_test, y_train, y_test = train_test_split(X.T, Y.ravel(), test_size=0.1)
    clf = MLPClassifier(alpha=0.1, max_iter=10000, hidden_layer_sizes=(8, 4), random_state=1)
    clf.fit(x_train, y_train)
    # 模型效果获取
    r = clf.score(x_train, y_train)
    print("R值(准确率):", r)
    # 预测
    # 绘制测试集结果验证
    plot_decision_boundary(lambda x: clf.predict(x), X, Y)

multi_class_nn()


