"""
@Time : 2019/10/9 19:25 PM
@Author : bjjoy2009
学习TensorFlow1.14
"""
import math
import numpy as np
import tensorflow as tf
import h5py

from tensorflow.python.framework import ops

# y_hat = tf.constant(36, name='y_hat')
# y = tf.constant(39, name='y')
# loss = tf.Variable((y - y_hat) ** 2, name='loss')
# init = tf.global_variables_initializer()
# with tf.Session() as session:
#     session.run(init)
#     print(session.run(loss))

# a = tf.constant(2)
# b = tf.constant(10)
# c = tf.multiply(a, b)
# print(c)
# with tf.Session() as session:
#     print(session.run(c))

# x = tf.placeholder(tf.int64, name='x')
# with tf.Session() as session:
#     print(session.run(2 * x, feed_dict={x: 3}))


def linear_function():
    """
    Y = WX + b
    :return:
    """
    np.random.seed(1)
    W = tf.constant(np.random.randn(4, 3), name='W')
    X = tf.constant(np.random.randn(3, 1), name='X')
    b = tf.constant(np.random.randn(4, 1), name='b')
    Y = tf.add(tf.matmul(W, X), b)
    with tf.Session() as session:
        print(session.run(Y))


def sigmoid(z):
    x = tf.placeholder(tf.float32, name='x')
    a = tf.sigmoid(x)
    with tf.Session() as session:
        result = session.run(a, feed_dict={x: z})
    return result


def cost(logits, labels):
    z = tf.placeholder(tf.float32, name='z')
    y = tf.placeholder(tf.float32, name='y')
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)
    with tf.Session() as session:
        result = session.run(cost, feed_dict={z: logits, y: labels})
    return result

# logits = sigmoid(np.array([0.2,0.4,0.7,0.9]))
# cost = cost(logits, np.array([0, 0, 1, 1]))
# print("cost = " + str(cost))


def one_hot_matrix(labels, C):
    C = tf.constant(C, name='C')
    one_hot_mat = tf.one_hot(indices=labels, depth=C, axis=0)
    with tf.Session() as session:
        result = session.run(one_hot_mat)
    return result


# labels = np.array([1, 2, 3, 0, 2, 1])
# one_hot = one_hot_matrix(labels, C=4)
# print("one_hot = " + str(one_hot))


def ones(shape):
    ones = tf.ones(shape)
    with tf.Session() as session:
        result = session.run(ones)
    return result

# print(ones([3]))


def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=[n_x, None], name='X')
    Y = tf.placeholder(tf.float32, shape=[n_y, None], name='Y')
    return X, Y


def init_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable('W2', [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable('W3', [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    return Z3


def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=labels))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):

    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X, Y = create_placeholders(n_x, n_y)

    parameters = init_parameters()

    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = session.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
            # Print the cost every epoch
            if print_cost is True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
        parameters = session.run(parameters)

        # Calculate the correct predictions
        # tf.argmax(Z3)：每个样本正向传播计算得到一个Z3向量，在计算Z3向量最大值位置，对应分类
        # tf.argmax(Y)：实际每个样本对应的labels所在位置，即真实分类
        # tf.equal：计算正向传播得到样本分类和实际分类，相同为True，不同为False
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set，将True False向量转化为float数值后，计算向量平均值
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    # index = 0
    # plt.imshow(X_train_orig[index])
    # print("y = " + str(np.squeeze(Y_train_orig[:, index])))
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    X_train = X_train_flatten/255
    X_test = X_test_flatten/255
    Y_train = convert_to_one_hot(Y_train_orig, 6)
    Y_test = convert_to_one_hot(Y_test_orig, 6)
    paras = model(X_train, Y_train, X_test, Y_test, num_epochs=1500)
    print('')