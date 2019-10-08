"""
@Time : 2019/10/4 19:17 PM
@Author : bjjoy2009
梯度检测
"""
import numpy as np

from class2_week1_practice.dnn_v3 import DNN

# 一个参数theta梯度检测,y = theta*x
def forward_propagation(x, theta):
    J = theta*x
    return J


def backward_propagation(x, theta):
    dtheta = x
    return dtheta

# x, theta = 2, 4
# dtheta = backward_propagation(x, theta)
# print ("dtheta = " + str(dtheta))


def gradient_check(x, theta, epsilon=1e-7):
    thetaplus = theta + epsilon
    thetaminus = theta - epsilon
    J_plus = forward_propagation(x, thetaplus)
    J_minus = forward_propagation(x, thetaminus)
    gradapprox = (J_plus - J_minus) / (2 * epsilon)
    grad = backward_propagation(x, theta)
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator
    if difference < 1e-7:
        print("The gradient is correct!")
    else:
        print("The gradient is wrong!")

    return difference

# print(gradient_check(2, 4))


def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:

        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys


def vector_to_dictionary(theta):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    parameters["W1"] = theta[:20].reshape((5,4))
    parameters["b1"] = theta[20:25].reshape((5,1))
    parameters["W2"] = theta[25:40].reshape((3,5))
    parameters["b2"] = theta[40:43].reshape((3,1))
    parameters["W3"] = theta[43:46].reshape((1,3))
    parameters["b3"] = theta[46:47].reshape((1,1))

    return parameters


def gradients_to_vector(grads):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """

    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
        # flatten parameter
        new_vector = np.reshape(grads[key], (-1, 1))

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta


def gradient_check_n(dnn, parameters, grads, X, Y, epsilon=1e-7):
    parameters_vector, _ = dictionary_to_vector(parameters)
    num_parameters = parameters_vector.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    for i in range(num_parameters):
        thetaplus = np.copy(parameters_vector)
        thetaplus[i][0] += epsilon
        caches, al = dnn.forward_propagation(vector_to_dictionary(thetaplus), X)
        J_plus[i][0] = dnn.compute_cost(Y, al)

        thetaminus = np.copy(parameters_vector)
        thetaminus[i][0] -= epsilon
        caches, al = dnn.forward_propagation(vector_to_dictionary(thetaminus), X)
        J_minus[i][0] = dnn.compute_cost(Y, al)

        gradapprox[i][0] = (J_plus[i][0] - J_minus[i][0])/(2.*epsilon)

    grads_vector = gradients_to_vector(grads)
    numerator = np.linalg.norm(grads_vector - gradapprox)
    denominator = np.linalg.norm(grads_vector) + np.linalg.norm(gradapprox)
    difference = numerator / denominator
    if difference < 1e-7:
        print("The gradient is correct!")
    else:
        print("The gradient is wrong!")
    return difference


if __name__ == '__main__':
    np.random.seed(1)
    X = np.random.randn(4, 3)
    Y = np.array([1, 1, 0])
    layer_dims = [X.shape[0], 5, 3, 1]
    dnn = DNN(X=X, Y=Y, layer_dims=layer_dims, max_iter=30000, alpha=0.3,
              print_loss=True, print_loss_iter=10000, lambd=0.7, keep_prob=0.86, initialization='random')
    parameters = dnn.init_parameters()
    caches, al = dnn.forward_propagation(parameters, X)
    grads = dnn.back_propagation(parameters, caches)
    print(gradient_check_n(dnn, parameters, grads, X, Y))
