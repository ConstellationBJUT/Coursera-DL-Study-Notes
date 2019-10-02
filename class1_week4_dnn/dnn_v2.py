"""
@Time : 2019/9/30 20:48 PM
@Author : bjjoy2009
模仿课件将程序模块化，采用课件用的数据进行程序实现
"""
import numpy as np


def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))
    return A


def sigmoid_backward(A):
    """
    sigmoid求导
    :param Z:
    :return:
    """
    dZ = A * (1 - A)
    return dZ


def relu(Z):
    A = np.maximum(0, Z)
    return A


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def tanh_backward(A):
    dZ = 1 - np.power(A, 2)
    return dZ


def sigmoid_loss(y, al):
    result = -1/len(y) * (np.dot(y, np.log(al).T) + np.dot(1-y, np.log(1-al).T))
    return result


class DNN:
    def __init__(self, X, Y, layer_dims, max_iter=10000, alpha=0.05, print_loss=False, activation='relu'):
        """
        :param X: 训练集
        :param Y: labels
        :param layer_dims: 各个层节点数list，layer_dims[0]训练集特征数，layer_dims[L]输出层节点数
        :param alpha: 梯度下降学习率
        """
        self.X = X
        self.Y = Y
        self.layer_dims = layer_dims
        self.max_iter = max_iter
        self.alpha = alpha
        self.m = self.X.shape[1]
        self.L = len(layer_dims) - 1  # 网络层数，不计算输入层
        self.print_loss = print_loss
        self.parameters = {}
        self.activation = activation

    def init_parameters(self):
        """
        初始化1~L层参数
        :return:
        """
        np.random.seed(1)
        parameters = {}
        for l in range(1, self.L + 1):
            # 比直接*0.01效果好
            Wl = np.random.randn(self.layer_dims[l], self.layer_dims[l-1])/np.sqrt(self.layer_dims[l-1])  # * 0.01
            bl = np.zeros((self.layer_dims[l], 1))
            parameters['W' + str(l)] = Wl
            parameters['b' + str(l)] = bl
        return parameters

    def linear_forward(self, A_pre, W, b):
        """
        计算l层Z=WX+b
        :param A_pre: l-1层A
        :param W: l层W
        :param b: l层b
        :return:
        """
        Z = np.dot(W, A_pre) + b
        return Z

    def linear_activation_forward(self, Z, activation):
        """
        第l层计算，A=σ(Z)
        :return:
        """
        if activation == 'relu':
            A = relu(Z)
        elif activation == 'tanh':
            A = np.tanh(Z)
        elif activation == 'sigmoid':
            A = sigmoid(Z)
        return A

    def forward_propagation(self, parameters, X):
        """
        前向传播
        :param parameters: 参数字典Wl，bl
        :param X: 特征数据
        :return:caches 缓存字典A0~AL和Z1~ZL，AL
        """
        caches = {'A0': X}
        Al = X  # 输入层
        # hidden layer 前项传播
        for l in range(1, self.L):
            Zl = self.linear_forward(A_pre=Al, W=parameters['W' + str(l)], b=parameters['b'+str(l)])
            Al = self.linear_activation_forward(Zl, activation=self.activation)
            caches['A' + str(l)] = Al
            caches['Z' + str(l)] = Zl

        # 输出层计算
        Zl = self.linear_forward(A_pre=Al, W=parameters['W' + str(self.L)], b=parameters['b'+str(self.L)])
        Al = self.linear_activation_forward(Zl, activation='sigmoid')
        caches['A' + str(self.L)] = Al
        caches['Z' + str(self.L)] = Zl

        return caches, Al

    def compute_cost(self, Y, AL):
        """
        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """

        cost = (-1./self.m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL)))
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        return cost

    def linear_backward(self, dZ, A_pre, W):
        """
        计算l层dZ,dW,db
        :param dZ: l层dZ
        :param A_pre: l-1层A
        :param W: l层W
        :return: l-1层dA_pre,l层dW,l层db
        """
        dW = 1/self.m * np.dot(dZ, A_pre.T)
        db = 1/self.m * np.sum(dZ, axis=1, keepdims=True)
        dA_pre = np.dot(W.T, dZ)
        return dA_pre, dW, db

    def linear_activation_backward(self, dA, Z, A, A_pre, W, activation):
        """
        计算l层dZ,dW,db和l-1层dA_pre
        :param dA: l层dA
        :param Z: l层Z
        :param A: l层A 用于计算激活函数的导数，减少计算σ(Z)
        :param A_pre: l-1层A，用于求dW
        :return: l-1层dA_pre,l层dW,l层db
        """
        if activation == 'relu':
            dZ = relu_backward(dA, Z)
        elif activation == 'tanh':
            dZ = dA * tanh_backward(A)
        elif activation == 'sigmoid':
            dZ = dA * sigmoid_backward(A)
        dA_pre, dW, db = self.linear_backward(dZ, A_pre, W)
        return dA_pre, dW, db

    def back_propagation(self, parameters, caches):
        """
        反向传播
        :param parameters: 各层W，b
        :param caches: list，各层Al
        :return: grads
        """
        grads = {}
        # 1.计算输出层和上一层的dA_pre
        L = self.L  # 输出层所在层数编号
        AL = caches['A' + str(L)]
        ZL = caches['Z' + str(L)]
        # 交叉熵损失函数求dl/dAl
        dAl = - (np.divide(self.Y, AL) - np.divide(1 - self.Y, 1 - AL))
        A_pre = caches['A' + str(L - 1)]
        grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = self.linear_activation_backward(dAl, ZL, AL, A_pre, parameters['W' + str(L)], 'sigmoid')
        # 2.计算hidden layer
        for l in reversed(range(1, L)):
            dAl = grads['dA' + str(l)]
            Al = caches['A' + str(l)]
            Al_pre = caches['A' + str(l - 1)]
            Zl = caches['Z' + str(l)]
            Wl = parameters['W' + str(l)]
            grads['dA' + str(l-1)], grads['dW' + str(l)], grads['db' + str(l)] = self.linear_activation_backward(dAl, Zl, Al, Al_pre, Wl, self.activation)
        return grads

    def update_parameters(self, parameters, grads):
        for l in range(1, self.L + 1):
            parameters['W' + str(l)] = parameters['W' + str(l)] - self.alpha * grads['dW' + str(l)]
            parameters['b' + str(l)] = parameters['b' + str(l)] - self.alpha * grads['db' + str(l)]
        return parameters

    def fit(self):
        """
        模型训练
        :return: 参数W和b
        """
        parameters = self.init_parameters()
        for i in range(self.max_iter):
            caches, al = self.forward_propagation(parameters, self.X)
            if self.print_loss and i % 100 == 0:
                loss = self.compute_cost(self.Y, al)
                print(i, loss)
            grads = self.back_propagation(parameters, caches)
            parameters = self.update_parameters(parameters, grads)
        self.parameters = parameters

    def predict(self, X):
        """
        预测函数
        :param X:
        :return: 包含0，1的list
        """
        cache, al = self.forward_propagation(self.parameters, X)
        predicts = (al > 0.5)
        return predicts

    def score(self, X, Y):
        predicts = self.predict(X)
        accuracy = float((np.dot(Y, predicts.T) + np.dot(1-Y, 1-predicts.T))/float(Y.size))
        return accuracy


# test
if __name__ == '__main__':
    from utils import load_planar_dataset, plot_decision_boundary
    np.random.seed(1)
    X, Y = load_planar_dataset()
    nn = DNN(X=X, Y=Y, layer_dims=[2, 4, 4, 1], max_iter=10000, alpha=1.2, print_loss=True, activation='tanh')
    nn.fit()
    predicts = nn.predict(X)
    accuracy = float((np.dot(Y, predicts.T) + np.dot(1-Y, 1-predicts.T))/float(Y.size)*100)
    print('Accuracy: %f ' % accuracy+ '%')
    plot_decision_boundary(lambda x: nn.predict(x.T), X, Y, 'tanh[2881]=' + str(accuracy) + '%')