"""
@Time : 2019/10/1 19:47 PM
@Author : bjjoy2009
判断图片是否是猫的实验，week4作业2，使用课程提供数据
结果与课程提供notebook一样
"""
import h5py
import numpy as np
from PIL import Image

from week4_dnn.dnn_v2 import DNN


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def is_cat(my_image, nn, num_px, classes, is_show=False):
    image = Image.open(my_image)
    x = np.array(image.resize((num_px, num_px)))
    x = x.reshape(num_px*num_px*3, -1)
    x = x/255
    my_predicted_image = nn.score(x, np.array([1]))
    print("y = " + str(np.squeeze(my_predicted_image)) + ', your L-layer model predicts a \"' + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    if is_show:
        image.show()


if __name__ == '__main__':
    np.random.seed(1)
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.
    layers_dims = [12288, 20, 7, 5, 1]
    nn = DNN(X=train_x, Y=train_y, layer_dims=layers_dims, max_iter=2500, alpha=0.0075, print_loss=True, activation='relu')
    nn.fit()
    accuracy = nn.score(train_x, train_y)
    print('train:', accuracy)
    print('test:', nn.score(test_x, test_y))

    is_cat('images/my_image.jpg', nn, 64, classes, is_show=True)