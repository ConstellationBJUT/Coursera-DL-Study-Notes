"""
@Time : 2019/9/19 2:25 PM 
@Author : bjjoy2009
one hidden layer 神经网络实现二分类
"""
from utils import *


def sigmoid(z):
    return 1/(1+np.exp(-z))


def sigmoid_loss(y, al):
    result = -1/len(y) * (np.dot(y, np.log(al).T) + np.dot(1-y, np.log(1-al).T))
    return result


class OneNN:
    def __init__(self, X, Y, max_iter=10000, alpha=1.2, nh=4, print_loss=False):
        """
        :param X: 特征
        :param Y: label
        :param max_iter: 迭代次数
        :param alpha: 梯度下降学习率
        :param nh: hidden_layer的神经元个数
        :param print_loss: 输出迭代loss
        """
        self.X = X
        self.Y = Y
        self.max_iter = max_iter
        self.alpha = alpha
        self.nh = nh
        self.nx = X.shape[0]
        self.ny = Y.shape[0]
        self.m = X.shape[1]
        self.print_loss = print_loss
        self.parameters = {}

    def init_parameters(self):
        # 初始化参数
        W1 = np.random.random((self.nh, self.nx)) * 0.01
        b1 = np.zeros((self.nh, 1))
        W2 = np.random.random((self.ny, self.nh)) * 0.01
        b2 = np.zeros((self.ny, 1))
        parameters = {
            'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2
        }
        return parameters

    def forward_propagation(self, parameters, X):
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)
        cache = {
            'Z1': Z1,
            'A1': A1,
            'Z2': Z2,
            'A2': A2
        }
        return cache, A2

    def back_propagation(self, parameters, cache):
        Z1 = cache['Z1']
        A1 = cache['A1']
        Z2 = cache['Z2']
        A2 = cache['A2']
        W2 = parameters['W2']

        dZ2 = A2 - self.Y
        dW2 = 1/self.m * np.dot(dZ2, A1.T)
        db2 = 1/self.m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
        dW1 = 1/self.m * np.dot(dZ1, self.X.T)
        db1 = 1/self.m * np.sum(dZ1, axis=1, keepdims=True)
        grads = {
            'dW2': dW2,
            'db2': db2,
            'dW1': dW1,
            'db1': db1
        }
        return grads

    def update_parameters(self, parameters, grads):
        """
        梯度下降
        :param parameters:
        :param grads:
        :return:
        """
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        dW2 = grads['dW2']
        db2 = grads['db2']
        dW1 = grads['dW1']
        db1 = grads['db1']

        W1 = W1 - self.alpha * dW1
        b1 = b1 - self.alpha * db1
        W2 = W2 - self.alpha * dW2
        b2 = b2 - self.alpha * db2

        parameters = {
            'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2
        }
        return parameters

    def fit(self):
        parameters = self.init_parameters()
        for i in range(self.max_iter):
            cache, a2 = self.forward_propagation(parameters, self.X)
            if self.print_loss and i % 1000 == 0:
                loss = sigmoid_loss(self.Y, a2)
                print(i, loss)
            grads = self.back_propagation(parameters, cache)
            parameters = self.update_parameters(parameters, grads)
        self.parameters = parameters

    def predict(self, X):
        cache, a2 = self.forward_propagation(self.parameters, X)
        predicts = (a2 > 0.5)
        return predicts


np.random.seed(1)
X, Y = load_planar_dataset()
nn = OneNN(X=X, Y=Y, print_loss=False)
nn.fit()
predicts = nn.predict(X)
print('Accuracy: %f ' % float((np.dot(Y, predicts.T) + np.dot(1-Y, 1-predicts.T))/float(Y.size)*100) + '%')
plot_decision_boundary(lambda x: nn.predict(x.T), X, Y)

