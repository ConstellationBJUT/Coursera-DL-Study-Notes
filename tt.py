"""
@Time : 2019/9/30 11:27 AM 
@Author : bjjoy2009
"""
import numpy as np
import tensorflow as tf


def convert_to_one_hot(Y, C):
    """
    :param Y: labels
    :param C: 类别数
    :return:
    """
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


a = np.array([[0, 1], [2, 3]])
b = np.array([[0, 1], [2, 3]])
np.flip
# print(convert_to_one_hot(a, 3))
# a = np.zeros((3, 2, 2, 3))
tf.contrib.la