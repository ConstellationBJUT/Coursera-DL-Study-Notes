"""
@Time : 2019/10/8 20:19 PM
@Author : bjjoy2009
class2 week2 优化算法
深层神经网络v4:
添加mini_batch梯度下降，momentum，Adam
"""

import numpy as np

import class2_week2_practice.optimizer_class as op


def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))
    return A


def sigmoid_backward(A):
    """
    sigmoid求导
    :param A: l层A=σ(Z)
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


class DNN:
    def __init__(self, X, Y, layer_dims, epochs=10000, alpha=0.05, print_loss=False, print_loss_iter=1000,
                 activation='relu', lambd=0, keep_prob=1, initialization="he", optimizer='gd',
                 batch_size=64, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        :param X: 训练集
        :param Y: labels
        :param layer_dims: 各个层节点数list，layer_dims[0]训练集特征数，layer_dims[L]输出层节点数
        :param epochs: 训练集迭代次数
        :param alpha: 梯度下降学习率
        :param print_loss: 是否打印loss
        :param print_loss_iter: 迭代多少次打印一次loss
        :param activation: hidden layer激活函数
        :param lambd: L2正则化参数
        :param keep_prob: dropout保留节点的比例
        :param initialization: 权重初始化方式
        :param optimizer: 优化算法(gd, momentum, RMSprop, Adam)
        :param batch_size: mini_batch块大小
        :param beta1: momentum算法需要的参数，Adam参数
        :param beta2: RMSprop算法需要的参数，Adam参数
        :param epsilon: momentum, RMSprop, Adam防止分母为零参数
        """
        self.X = X
        self.Y = Y
        self.layer_dims = layer_dims
        self.epochs = epochs
        self.alpha = alpha
        self.m = self.X.shape[1]
        self.L = len(layer_dims) - 1  # 网络层数，不计算输入层
        self.print_loss = print_loss
        self.print_loss_iter = print_loss_iter
        self.activation = activation
        self.lambd = lambd
        self.keep_prob = keep_prob
        self.initialization = initialization
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.parameters = {}

    def init_parameters(self):
        """
        初始化1~L层参数
        :return:
        """
        np.random.seed(3)
        parameters = {}
        for l in range(1, self.L + 1):
            if self.initialization == 'zeros':
                Wl = np.zeros((self.layer_dims[l], self.layer_dims[l-1]))
            elif self.initialization == 'random':
                Wl = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 10
            elif self.initialization == 'he':
                Wl = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * (np.sqrt(2/ self.layer_dims[l-1]))
            else:
                Wl = np.random.randn(self.layer_dims[l], self.layer_dims[l-1])/np.sqrt(self.layer_dims[l-1])
            bl = np.zeros((self.layer_dims[l], 1))
            # bl = np.random.randn(self.layer_dims[l], 1)
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

        # cost = (-1./self.m) * np.sum(np.multiply(Y, np.log(np.clip(AL, 1e-6, 1))) + np.multiply((1-Y), np.log(np.clip(1-AL, 1e-6, 1))))
        cost = (-1./Y.shape[1]) * np.nansum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL)))
        cost = np.squeeze(cost)
        return cost

    def linear_backward(self, dZ, A_pre, W):
        """
        计算l层dZ,dW,db
        :param dZ: l层dZ
        :param A_pre: l-1层A
        :param W: l层W
        :return: l-1层dA_pre,l层dW,l层db
        """
        m = A_pre.shape[1]
        dW = 1./m * np.dot(dZ, A_pre.T)
        db = 1./m * np.sum(dZ, axis=1, keepdims=True)
        dA_pre = np.dot(W.T, dZ)
        return dA_pre, dW, db

    def linear_activation_backward(self, dA, Z, A, A_pre, W, activation):
        """
        计算l层dZ,dW,db和l-1层dA_pre
        :param dA: l层dA
        :param Z: l层Z
        :param A: l层A 用于计算激活函数的导数，减少计算σ(Z)
        :param A_pre: l-1层A，用于求dW
        :param Y_batch:
        :return: l-1层dA_pre,l层dW,l层db
        """
        if activation == 'relu':
            dZ = relu_backward(dA, Z)
            # dZ = np.multiply(dA, np.int64(A > 0))
        elif activation == 'tanh':
            dZ = dA * tanh_backward(A)
        elif activation == 'sigmoid':
            dZ = dA * sigmoid_backward(A)  # 在sigmoid可用在hidden layer
        dA_pre, dW, db = self.linear_backward(dZ, A_pre, W)
        return dA_pre, dW, db

    def output_backward(self, A_pre, W, AL, Y_batch):
        """
        输出层是二分类，采用sigmoid，计算反向传播
        :param A_pre: L-1层A
        :param W: L层W
        :param AL: L层AL
        :param Y_batch: 分块Y
        :return:
        """
        dZ = AL - Y_batch
        dA_pre, dW, db = self.linear_backward(dZ, A_pre, W)
        return dA_pre, dW, db

    def back_propagation(self, parameters, caches, Y_batch):
        """
        反向传播
        :param parameters: 各层W，b
        :param caches: list，各层Al
        :param Y_batch:
        :return: grads
        """
        grads = {}
        # 1.计算输出层和前一层的dA_pre
        L = self.L  # 输出层所在层数编号
        AL = caches['A' + str(L)]
        A_pre = caches['A' + str(L - 1)]
        grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = self.output_backward(A_pre, parameters['W' + str(L)], AL, Y_batch)
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

    def get_batch(self, seed):
        """
        分割数据集到各个batch
        :return:
        """
        # 下面随机排列数据，是为了和课件保持一致
        np.random.seed(seed)            # To make your "random" minibatches the same as ours
        m = self.X.shape[1]
        permutation = list(np.random.permutation(m))
        shuffled_X = self.X[:, permutation]
        shuffled_Y = self.Y[:, permutation].reshape((1, m))

        batch_list = []
        batch_num = int(self.m/self.batch_size) + 1
        for k in range(batch_num):
            # 根据batch编号，获取每个batch训练集
            start = k * self.batch_size
            if start >= self.m:
                # 起始值超过数据量最大值，已没有数据
                break
            end = start + self.batch_size
            if end >= self.m:
                # 块结束超过数据集最大值，结束值设置为m
                end = self.m
            X_batch = shuffled_X[:, start:end]
            Y_batch = shuffled_Y[:, start:end]
            batch_list.append((X_batch, Y_batch))
        return batch_list

    def fit(self):
        """
        模型训练
        :return: 参数W和b
        """
        seed = 10
        t = 0
        parameters = self.init_parameters()
        # Initialize the optimizer
        if self.optimizer == "gd":
            pass  # no initialization required for gradient descent
        elif self.optimizer == "momentum":
            v = op.initialize_momentum(parameters)
        elif self.optimizer == "adam":
            v, s = op.initialize_adam(parameters)

        for i in range(self.epochs):
            # 每个batch进行梯度下降
            seed += 1
            for X_batch, Y_batch in self.get_batch(seed):
                caches, al = self.forward_propagation(parameters, X_batch)
                cost = self.compute_cost(Y_batch, al)
                grads = self.back_propagation(parameters, caches, Y_batch)

                # Update parameters
                if self.optimizer == "gd":
                    parameters = op.gd(parameters, grads, self.alpha)
                elif self.optimizer == "momentum":
                    parameters, v = op.momentum(parameters, grads, v, self.alpha, self.beta1)
                elif self.optimizer == "adam":
                    t = t + 1  # Adam counter
                    parameters, v, s = op.adam(parameters, grads, v, s, t, self.alpha, self.beta1, self.beta2, self.epsilon)

            if self.print_loss and i % self.print_loss_iter == 0:
                print(i, cost)
        self.parameters = parameters

    # --------------------regularization start-----------------------------------
    def compute_cost_with_regularization(self, Y, AL, parameters, lambd):
        """
        L2正则化计算损失函数 cost=1/m * L(al,Y) + λ/2m * (W二范数平方)
        :param Y: labels（0，1）
        :param AL: 每个样本输入得到的结果概率
        :param parameters: dnn参数（Wl, bl）
        :param lambd: 正则化参数
        :return:
        """
        m = Y.shape[1]
        cross_entropy_cost = self.compute_cost(Y, AL)
        L2_regularization_cost = 0
        for l in range(1, self.L + 1):
            L2_regularization_cost += np.sum(np.square(parameters['W' + str(l)]))
        L2_regularization_cost = lambd/(2*m) * L2_regularization_cost
        cost = cross_entropy_cost + L2_regularization_cost
        cost = np.squeeze(cost)
        return cost

    def linear_backward_with_regularization(self, dZ, A_pre, W):
        """
        计算l层dZ,dW,db
        :param dZ: l层dZ
        :param A_pre: l-1层A
        :param W: l层W
        :return: l-1层dA_pre,l层dW,l层db
        """
        m = A_pre.shape[1]
        dW = 1/m * np.dot(dZ, A_pre.T) + (self.lambd/m) * W
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA_pre = np.dot(W.T, dZ)
        return dA_pre, dW, db

    def linear_activation_backward_with_regularization(self, dA, Z, A, A_pre, W, activation):
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
        dA_pre, dW, db = self.linear_backward_with_regularization(dZ, A_pre, W)
        return dA_pre, dW, db

    def back_propagation_with_regularization(self, parameters, caches, Y_batch):
        """
        反向传播regularization
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
        dAl = - (np.divide(Y_batch, AL) - np.divide(1 - Y_batch, 1 - AL))
        A_pre = caches['A' + str(L - 1)]
        grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = self.linear_activation_backward_with_regularization(dAl, ZL, AL, A_pre, parameters['W' + str(L)], 'sigmoid')
        # 2.计算hidden layer
        for l in reversed(range(1, L)):
            dAl = grads['dA' + str(l)]
            Al = caches['A' + str(l)]
            Al_pre = caches['A' + str(l - 1)]
            Zl = caches['Z' + str(l)]
            Wl = parameters['W' + str(l)]
            grads['dA' + str(l-1)], grads['dW' + str(l)], grads['db' + str(l)] = self.linear_activation_backward_with_regularization(dAl, Zl, Al, Al_pre, Wl, self.activation)
        return grads

    def fit_regularization(self):
        """
        模型训练
        :return:
        """
        seed = 10
        t = 0
        parameters = self.init_parameters()
        # Initialize the optimizer
        if self.optimizer == "gd":
            pass  # no initialization required for gradient descent
        elif self.optimizer == "momentum":
            v = op.initialize_momentum(parameters)
        elif self.optimizer == "adam":
            v, s = op.initialize_adam(parameters)

        for i in range(self.epochs):
            for X_batch, Y_batch in self.get_batch(seed):
                caches, al = self.forward_propagation(parameters, X_batch)
                cost = self.compute_cost_with_regularization(Y_batch, al, parameters, self.lambd)
                grads = self.back_propagation_with_regularization(parameters, caches, Y_batch)

                # Update parameters
                if self.optimizer == "gd":
                    parameters = op.gd(parameters, grads, self.alpha)
                elif self.optimizer == "momentum":
                    parameters, v = op.momentum(parameters, grads, v, self.alpha, self.beta1)
                elif self.optimizer == "adam":
                    t = t + 1  # Adam counter
                    parameters, v, s = op.adam(parameters, grads, v, s, t, self.alpha, self.beta1, self.beta2,  self.epsilon)
            if self.print_loss and i % self.print_loss_iter == 0:
                print(i, cost)
        self.parameters = parameters
    # --------------------regularization end-----------------------------------

    # --------------------dropout start-----------------------------------
    def forward_propagation_with_dropout(self, parameters, X):
        """
        前向传播dropout
        :param parameters: 参数字典Wl，bl
        :param X: 特征数据
        :return:caches 缓存字典A0~AL和Z1~ZL，AL
        """
        np.random.seed(1)
        caches = {'A0': X}
        Al = X  # 输入层
        # hidden layer 前项传播
        for l in range(1, self.L):
            Zl = self.linear_forward(A_pre=Al, W=parameters['W' + str(l)], b=parameters['b'+str(l)])
            Al = self.linear_activation_forward(Zl, activation=self.activation)

            # step1~4,dropout
            Dl = np.random.rand(Al.shape[0], Al.shape[1])  #initialize matrix Dl = np.random.rand(..., ...)
            Dl = Dl < self.keep_prob                   # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
            Al = np.multiply(Al, Dl)                      # Step 3: shut down some neurons of A1
            Al /= self.keep_prob               # Step 4: scale the value of neurons that haven't been shut down

            caches['A' + str(l)] = Al
            caches['Z' + str(l)] = Zl
            caches['D' + str(l)] = Dl

        # 输出层计算
        Zl = self.linear_forward(A_pre=Al, W=parameters['W' + str(self.L)], b=parameters['b'+str(self.L)])
        Al = self.linear_activation_forward(Zl, activation='sigmoid')
        caches['A' + str(self.L)] = Al
        caches['Z' + str(self.L)] = Zl
        return caches, Al

    def back_propagation_with_dropout(self, parameters, caches, Y_batch):
        """
        反向传播dropout
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
        # dAl = - (np.divide(self.Y, np.clip(AL, 1e-7, 1)) - np.divide(1 - self.Y, np.clip(1 - AL, 1e-7, 1)))
        dAl = - (np.divide(Y_batch, AL) - np.divide(1 - Y_batch, 1 - AL))
        A_pre = caches['A' + str(L - 1)]
        grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = self.linear_activation_backward(dAl, ZL, AL, A_pre, parameters['W' + str(L)], 'sigmoid')
        # 2.计算hidden layer
        for l in reversed(range(1, L)):
            dAl = grads['dA' + str(l)]
            # dropout处理dAl
            dAl = np.multiply(dAl, caches['D' + str(l)])
            dAl /= self.keep_prob

            Al = caches['A' + str(l)]
            Al_pre = caches['A' + str(l - 1)]
            Zl = caches['Z' + str(l)]
            Wl = parameters['W' + str(l)]
            grads['dA' + str(l-1)], grads['dW' + str(l)], grads['db' + str(l)] = self.linear_activation_backward(dAl, Zl, Al, Al_pre, Wl, self.activation)
        return grads

    def fit_dropout(self):
        """
        模型训练dropout
        :return: 参数W和b
        """
        seed = 10
        t = 0
        parameters = self.init_parameters()
        # Initialize the optimizer
        if self.optimizer == "gd":
            pass  # no initialization required for gradient descent
        elif self.optimizer == "momentum":
            v = op.initialize_momentum(parameters)
        elif self.optimizer == "adam":
            v, s = op.initialize_adam(parameters)

        for i in range(self.epochs):
            for X_batch, Y_batch in self.get_batch(seed):
                caches, al = self.forward_propagation_with_dropout(parameters, X_batch)
                cost = self.compute_cost(Y_batch, al)
                grads = self.back_propagation_with_dropout(parameters, caches, Y_batch)
                # Update parameters
                if self.optimizer == "gd":
                    parameters = op.gd(parameters, grads, self.alpha)
                elif self.optimizer == "momentum":
                    parameters, v = op.momentum(parameters, grads, v, self.alpha, self.beta1)
                elif self.optimizer == "adam":
                    t = t + 1  # Adam counter
                    parameters, v, s = op.adam(parameters, grads, v, s, t, self.alpha, self.beta1, self.beta2,  self.epsilon)
            if self.print_loss and i % self.print_loss_iter == 0:
                print(i, cost)
        self.parameters = parameters
    # --------------------dropout end-----------------------------------

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
    pass