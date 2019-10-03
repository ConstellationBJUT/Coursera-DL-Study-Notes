"""
@Time : 2019/9/26 20:29 PM
@Author : bjjoy2009
深层神经网络实现，hidden layer采用不同的激活函数，使用week3实验数据进行对比
该与课件程序差别较大，
"""
from utils import *


def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))
    return A


def relu(Z):
    A = np.maximum(0, Z)
    return A


def sigmoid_loss(y, al):
    result = -1/y.shape[1] * (np.dot(y, np.log(al).T) + np.dot(1-y, np.log(1-al).T))
    return np.squeeze(result)


class DNN:
    def __init__(self, X, Y, layer_dims, max_iter=10000, alpha=0.05, print_loss=False, activation='relu'):
        """
        :param X: 训练集
        :param Y: labels
        :param layer_dims: 各个层节点数list，layer_dims[0]训练集特征数，layer_dims[L]=1输出层节点数
        :param alpha: 梯度下降学习率
        """
        self.X = X
        self.Y = Y
        self.layer_dims = layer_dims
        self.max_iter = max_iter
        self.alpha = alpha
        self.m = self.X.shape[1]
        self.L = len(layer_dims)
        self.print_loss = print_loss
        self.parameters = {}
        self.activation = activation

    def init_parameters(self):
        parameters = {}
        for l in range(1, self.L):
            Wl = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
            bl = np.zeros((self.layer_dims[l], 1))
            parameters['W' + str(l)] = Wl
            parameters['b' + str(l)] = bl
        return parameters

    def forward_propagation(self, parameters, X):
        # 输入层初始化A0
        cache = {'A0': self.X}
        # hidden layer前向传播计算
        A_pre = X
        for l in range(1, self.L-1):
            Wl = parameters['W' + str(l)]
            bl = parameters['b' + str(l)]
            Zl = np.dot(Wl, A_pre) + bl
            if self.activation == 'relu':
                Al = relu(Zl)  # relu做激活函数
            elif self.activation == 'tanh':
                Al = np.tanh(Zl)  # tanh做激活函数
            A_pre = Al
            cache['A' + str(l)] = Al
            cache['Z' + str(l)] = Zl

        # 输出层计算
        Wl = parameters['W' + str(self.L-1)]
        bl = parameters['b' + str(self.L-1)]
        Zl = np.dot(Wl, A_pre) + bl
        Al = sigmoid(Zl)
        cache['A' + str(self.L-1)] = Al
        cache['Z' + str(self.L-1)] = Zl

        return cache, Al

    def back_propagation(self, parameters, cache):
        grads = {}
        # 输出层（L-1层）反向传播
        L = self.L - 1
        Al = cache['A' + str(L)]
        Zl = cache['Z' + str(L)]
        dZl = Al - self.Y
        dWl = 1/self.m * np.dot(dZl, cache['A' + str(L-1)].T)
        dbl = 1/self.m * np.sum(dZl, axis=1, keepdims=True)
        grads['dW' + str(L)] = dWl
        grads['db' + str(L)] = dbl
        # 隐藏层和输入层反向传播（0~L-2层）
        for l in reversed(range(1, L)):
            dAl = np.dot(parameters['W' + str(l+1)].T, dZl)
            if self.activation == 'relu':
                dZl = np.array(dAl, copy=True)  # relu做激活使用
                dZl[cache['Z' + str(l)] <= 0] = 0  # relu做激活使用
            elif self.activation == 'tanh':
                dZl = np.multiply(dAl, (1-np.power(cache['A' + str(l)], 2)))  # tanh做激活使用
            dWl = 1/self.m * np.dot(dZl, cache['A' + str(l-1)].T)
            dbl = 1/self.m * np.sum(dZl, axis=1, keepdims=True)
            grads['dW' + str(l)] = dWl
            grads['db' + str(l)] = dbl
        return grads

    def update_parameters(self, parameters, grads):
        for l in range(1, self.L):
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
            cache, al = self.forward_propagation(parameters, self.X)
            if self.print_loss and i % 1000 == 0:
                loss = sigmoid_loss(self.Y, al)
                print(i, loss)
            grads = self.back_propagation(parameters, cache)
            parameters = self.update_parameters(parameters, grads)
        self.parameters = parameters

    def predict(self, X):
        """
        预测函数
        :param X:
        :return: 包含0，1的list
        """
        cache, a2 = self.forward_propagation(self.parameters, X)
        predicts = (a2 > 0.5)
        return predicts


np.random.seed(1)
X, Y = load_planar_dataset()
nn = DNN(X=X, Y=Y, layer_dims=[2, 4, 4, 1], max_iter=20000, alpha=0.1, print_loss=True, activation='relu')
nn.fit()
predicts = nn.predict(X)
accuracy = float((np.dot(Y, predicts.T) + np.dot(1-Y, 1-predicts.T))/float(Y.size)*100)
print('Accuracy: %f ' % accuracy+ '%')
plot_decision_boundary(lambda x: nn.predict(x.T), X, Y, 'relu[2881]=' + str(accuracy) + '%')
