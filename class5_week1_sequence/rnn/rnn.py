"""
@Time : 2019/12/7 2:38 PM 
@Author : bjjoy2009
rnn类，实现前向和反向传播，构建rnn模型
主要参数：
T_x:cell数
n_a:每个cell计算后输出->输入到下一层的a的维度
n_x:每个cell输入x的维度
n_y:每个cell输出y的维度，也就是多分类类别数
m:每个batch的大小，也就是每个batch是m个输入序列，每个序列是T_x个输入

总结：每次输入m个字符串（或句子），每个字符串T_x个字符（或单词），每个字符（或单词）是n_x维向量表示，
每个cell对应一个输出n_y（做文本生成实验n_y=n_x，做文本分类实验n_y是分类数）
"""

import numpy as np


def softmax(x):
    """
    softmax函数，减去最大值防止爆炸
    下面公式等价于 e_x=np.exp(x), e_x/e_x.sum(axis=0)
    :param x:
    :return:
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def smooth(loss, cur_loss):
    """
    loss平滑公式，相当于取1000次平均损失
    :param loss:
    :param cur_loss:
    :return:
    """
    return loss * 0.999 + cur_loss * 0.001


def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('----\n %s \n----' % (txt, ))


def get_initial_loss(vocab_size, seq_length):
    """
    初始化损失，个人理解：softmax损失函数L=-sum(yi * log(yi_hat))，i=0,1,...,vocab_size
    在预测下一个字符实验，下面公式相当于每个cell预测每个字符概率相等，都是1/vocab_size。
    y是vocab_size维向量，第i个位置是标记正确的是1，其余位置是0。
    有seq_length个cell。
    :param vocab_size: 字符（或单词）数量
    :param seq_length: cell数量
    :return:
    """
    return -np.log(1.0/vocab_size)*seq_length


class RNN:
    def __init__(self, epochs=20, n_a=16, alpha=0.01, batch_size=32):
        """
        :param epochs: 迭代次数
        :param n_a: 隐藏层节点数
        :param alpha: 梯度下降参数
        :param batch_size: 每个batch大小
        """
        self.epochs = epochs
        self.n_a = n_a
        self.alpha = alpha
        self.parameters = {}
        self.loss = 0.0
        self.n_x = 2
        self.n_y = 2
        self.m = batch_size

    def initialize_parameters(self, n_a, n_x, n_y):
        """
        Initialize parameters with small random values
        :param n_a: 每个cell输出a的维度
        :param n_x: 每个cell输入xi的维度
        :param n_y: 每个cell输出yi的维度

        Returns:
        parameters -- python dictionary containing:
            Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
            Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
            Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
            b --  Bias, numpy array of shape (n_a, 1)
            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
        """
        np.random.seed(1)
        Wax = np.random.randn(n_a, n_x)*0.01  # input to hidden
        Waa = np.random.randn(n_a, n_a)*0.01  # hidden to hidden
        Wya = np.random.randn(n_y, n_a)*0.01  # hidden to output
        ba = np.zeros((n_a, 1))  # hidden bias
        by = np.zeros((n_y, 1))  # output bias
        self.parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
        self.n_x = n_x
        self.n_y = n_y

    def rnn_cell_forward(self, xt, a_prev):
        """
        Implements a single forward step of the RNN-cell as described in Figure (2)

        Arguments:
        xt -- your input data at timestep "t", numpy array of shape (n_x, m).
        a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
        parameters -- python dictionary containing:
                            Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                            Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                            Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                            ba --  Bias, numpy array of shape (n_a, 1)
                            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
        Returns:
        a_next -- next hidden state, of shape (n_a, m)
        yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
        cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
        """

        # Retrieve parameters from "parameters"
        Wax = self.parameters["Wax"]
        Waa = self.parameters["Waa"]
        Wya = self.parameters["Wya"]
        ba = self.parameters["ba"]
        by = self.parameters["by"]

        # compute next activation state using the formula given above
        a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
        # compute output of the current cell using the formula given above
        yt_pred = softmax(np.dot(Wya, a_next) + by)

        # store values you need for backward propagation in cache
        cache = (a_next, a_prev, xt)

        return a_next, yt_pred, cache

    def rnn_forward(self, x, a_prev):
        """
        Arguments:
        x -- Input data for every time-step, of shape (n_x, m, T_x).
        a_prev -- Initial hidden state, of shape (n_a, m)
        parameters -- python dictionary containing:
            Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
            Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
            Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
            ba --  Bias numpy array of shape (n_a, 1)
            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

        Returns:
        a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
        y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
        caches -- tuple of values needed for the backward pass, contains (list of caches, x)
        """

        # Initialize "caches" which will contain the list of all caches
        caches = []

        # Retrieve dimensions from shapes of x and parameters["Wya"]
        n_x, m, T_x = x.shape
        n_y, n_a = self.parameters["Wya"].shape

        # initialize "a" and "y" with zeros
        a = np.zeros((n_a, m, T_x))
        y_pred = np.zeros((n_y, m, T_x))

        # Initialize a_next (≈1 line)
        a_next = a_prev

        # loop over all time-steps
        for t in range(T_x):
            # Update next hidden state, compute the prediction, get the cache
            a_next, yt_pred, cache = self.rnn_cell_forward(xt=x[:, :, t], a_prev=a_next)
            # Save the value of the new "next" hidden state in a (≈1 line)
            a[:, :, t] = a_next
            # Save the value of the prediction in y (≈1 line)
            y_pred[:, :, t] = yt_pred

            # Append "cache" to "caches" (≈1 line)
            caches.append(cache)

        # store values needed for backward propagation in cache
        caches = (caches, x)

        return a, y_pred, caches

    def compute_loss(self, y_hat, y):
        """
        计算损失函数
        :param y_hat: (n_y, m, T_x),经过rnn正向传播得到的值
        :param y: (n_y, m, T_x),标记的真实值
        :return: loss
        """
        n_y, m, T_x = y.shape
        for t in range(T_x):
            self.loss -= 1/m * np.sum(np.multiply(y[:, :, t], np.log(y_hat[:, :, t])))
        return self.loss

    def rnn_cell_backward(self, dz, gradients, cache):
        """
        Implements the backward pass for the RNN-cell (single time-step).

        Arguments:
        dz -- 由这两个公式计算出，cell输出y_hat = softmax(z), z=np.dot(Wya, z) + by
        gradients -- Gradient of loss with respect to next hidden state
        cache -- python dictionary containing useful values (output of rnn_cell_forward())

        Returns:
        gradients -- python dictionary containing:
                            dx -- Gradients of input data, of shape (n_x, m)
                            da_prev -- Gradients of previous hidden state, of shape (n_a, m)
                            dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                            dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                            dba -- Gradients of bias vector, of shape (n_a, 1)
        """

        # Retrieve values from cache
        (a_next, a_prev, xt) = cache

        # Retrieve values from parameters
        Wax = self.parameters["Wax"]
        Waa = self.parameters["Waa"]
        Wya = self.parameters["Wya"]
        ba = self.parameters["ba"]
        by = self.parameters["by"]

        gradients['dWya'] += np.dot(dz, a_next.T)
        gradients['dby'] += np.sum(dz, axis=1, keepdims=True)
        # cell的da由两部分组成，
        da = np.dot(Wya.T, dz) + gradients['da_next']

        # compute the gradient of tanh with respect to a_next (≈1 line)
        dtanh = np.multiply(da, 1 - np.square(a_next))
        # compute the gradient of the loss with respect to Wax (≈2 lines)
        gradients['dxt'] = np.dot(Wax.T, dtanh)
        gradients['dWax'] += np.dot(dtanh, xt.T)

        # compute the gradient with respect to Waa (≈2 lines)
        gradients['dWaa'] += np.dot(dtanh, a_prev.T)

        # compute the gradient with respect to b (≈1 line)
        gradients['dba'] += np.sum(dtanh, axis=1, keepdims=True)

        # 前一个cell的da_next
        gradients['da_next'] = np.dot(Waa.T, dtanh)

        return gradients

    def rnn_backward(self, y, y_hat, caches):
        """
        Implement the backward pass for a RNN over an entire sequence of input data.
        :param y: label，shape(n_y, m, T_x)
        :param y_hat: softmax rnn forward output ，shape(n_y, m, T_x)
        :param caches: tuple containing information from the forward pass (rnn_forward)

        Returns:
        gradients -- python dictionary containing:
            dx -- Gradient w.r.t. the input data, numpy-array of shape (n_x, m, T_x)
            da0 -- Gradient w.r.t the initial hidden state, numpy-array of shape (n_a, m)
            dWax -- Gradient w.r.t the input's weight matrix, numpy-array of shape (n_a, n_x)
            dWaa -- Gradient w.r.t the hidden state's weight matrix, numpy-arrayof shape (n_a, n_a)
            dba -- Gradient w.r.t the bias, of shape (n_a, 1)
            dWya -- Gradient w.r.t the output's state's weight matrix, numpy-arrayof shape (n_y, n_a)
            dby -- Gradient w.r.t the output's bias, of shape (n_y, 1)
        """

        # Retrieve values from the first cache (t=1) of caches
        (caches, x) = caches
        n_x, m, T_x = x.shape
        # initialize the gradients with the right sizes
        gradients = {}
        dx = np.zeros((n_x, m, T_x))
        gradients['dWax'] = np.zeros((self.n_a, self.n_x))
        gradients['dWaa'] = np.zeros((self.n_a, self.n_a))
        gradients['dba'] = np.zeros((self.n_a, 1))
        gradients['da_next'] = np.zeros((self.n_a, self.m))
        gradients['dWya'] = np.zeros((self.n_y, self.n_a))
        gradients['dby'] = np.zeros((self.n_y, 1))
        dz = y_hat - y  # y_hat=softmax(z), dz=dl/dy_hat * dy_hat/dz

        # Loop through all the time steps
        for t in reversed(range(T_x)):
            gradients = self.rnn_cell_backward(dz=dz[:, :, t], gradients=gradients, cache=caches[t])
            dx[:, :, t] = gradients["dxt"]

        return gradients

    def update_parameters(self, gradients):
        """
        梯度下降
        :param gradients:
        :return:
        """
        self.parameters['Wax'] += -self.alpha * gradients['dWax']
        self.parameters['Waa'] += -self.alpha * gradients['dWaa']
        self.parameters['Wya'] += -self.alpha * gradients['dWya']
        self.parameters['ba'] += -self.alpha * gradients['dba']
        self.parameters['by'] += -self.alpha * gradients['dby']

    def clip(self, gradients, maxValue=5):
        """
        Clips the gradients' values between minimum and maximum.

        Arguments:
        gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
        maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue

        Returns:
        gradients -- a dictionary with the clipped gradients.
        """

        dWaa, dWax, dWya, dba, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['dba'], gradients['dby']

        # clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]. (≈2 lines)
        for gradient in [dWax, dWaa, dWya, dba, dby]:
            np.clip(gradient, -1*maxValue, maxValue, out=gradient)

        gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "dba": dba, "dby": dby}

        return gradients

    def optimize(self, X, Y, a_prev):
        """
        Execute one step of the optimization to train the model.

        Arguments:
        X -- 输入数据序列，维度(n_x, m, T_x)，n_x是每个step输入xi的维度，m是一个batch数据量，T_x一个序列长度
        Y -- 每个输入xi对应的输出yi (n_y, m, T_x)，n_y是输出向量（分类数，只有一位是1）
        a_prev -- previous hidden state.

        Returns:
        loss -- value of the loss function (cross-entropy)
        gradients -- python dictionary containing:
                            dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                            dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                            dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                            db -- Gradients of bias vector, of shape (n_a, 1)
                            dby -- Gradients of output bias vector, of shape (n_y, 1)
        a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
        """

        # 正向传播
        a, y_pred, caches = self.rnn_forward(X, a_prev)
        # 计算损失
        loss = self.compute_loss(y_hat=y_pred, y=Y)

        gradients = self.rnn_backward(Y, y_pred, caches)

        gradients = self.clip(gradients=gradients, maxValue=5)

        self.update_parameters(gradients)

        return loss, gradients, a[:, :, -1]
