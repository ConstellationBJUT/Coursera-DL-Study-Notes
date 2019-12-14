"""
@Time : 2019/12/13 2:26 PM 
@Author : bjjoy2009
"""
import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LSTM:
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
        :param n_a: 每个cell输出a的维度
        :param n_x: 每个cell输入xi的维度
        :param n_y: 每个cell输出yi的维度
        """
        np.random.seed(1)
        Wf = np.random.randn(n_a, n_a + n_x)*0.01
        bf = np.zeros((n_a, 1))
        Wi = np.random.randn(n_a, n_a + n_x)*0.01
        bi = np.zeros((n_a, 1))
        Wc = np.random.randn(n_a, n_a + n_x)*0.01
        bc = np.zeros((n_a, 1))
        Wo = np.random.randn(n_a, n_a + n_x)*0.01
        bo = np.zeros((n_a, 1))
        Wy = np.random.randn(n_y, n_a)*0.01
        by = np.zeros((n_y, 1))

        self.parameters = {
            "Wf": Wf,
            "bf": bf,
            "Wi": Wi,
            "bi": bi,
            "Wc": Wc,
            "bc": bc,
            "Wo": Wo,
            "bo": bo,
            "Wy": Wy,
            "by": by,
        }
        self.n_x = n_x
        self.n_y = n_y

    def lstm_cell_forward(self, xt, a_prev, c_prev):
        """
        Implement a single forward step of the LSTM-cell as described in Figure (4)

        Arguments:
        xt -- your input data at timestep "t", numpy array of shape (n_x, m).
        a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
        c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
        parameters -- python dictionary containing:
                            Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                            bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                            Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                            bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                            Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                            bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                            Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                            bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                            Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

        Returns:
        a_next -- next hidden state, of shape (n_a, m)
        c_next -- next memory state, of shape (n_a, m)
        yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
        cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)

        Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
              c stands for the memory value
        """

        # Retrieve parameters from "parameters"
        Wf = self.parameters["Wf"]
        bf = self.parameters["bf"]
        Wi = self.parameters["Wi"]
        bi = self.parameters["bi"]
        Wc = self.parameters["Wc"]
        bc = self.parameters["bc"]
        Wo = self.parameters["Wo"]
        bo = self.parameters["bo"]
        Wy = self.parameters["Wy"]
        by = self.parameters["by"]

        # Retrieve dimensions from shapes of xt and Wy
        n_x, m = xt.shape
        n_y, n_a = Wy.shape

        # Concatenate a_prev and xt (≈3 lines)
        concat = np.zeros((n_a + n_x, m))
        concat[: n_a, :] = a_prev
        concat[n_a :, :] = xt

        # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given figure (4) (≈6 lines)
        ft = sigmoid(np.dot(Wf, concat) + bf)
        it = sigmoid(np.dot(Wi, concat) + bi)
        cct = np.tanh(np.dot(Wc, concat) + bc)
        c_next = np.multiply(ft, c_prev) + np.multiply(it, cct)
        ot = sigmoid(np.dot(Wo, concat) + bo)
        a_next = np.multiply(ot, np.tanh(c_next))

        # Compute prediction of the LSTM cell (≈1 line)
        yt_pred = softmax(np.dot(Wy, a_next) + by)

        # store values needed for backward propagation in cache
        cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt)

        return a_next, c_next, yt_pred, cache

    def lstm_forward(self, x, a0):
        """
        Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (3).

        Arguments:
        x -- Input data for every time-step, of shape (n_x, m, T_x).
        a0 -- Initial hidden state, of shape (n_a, m)
        parameters -- python dictionary containing:
                            Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                            bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                            Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                            bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                            Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                            bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                            Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                            bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                            Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

        Returns:
        a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
        y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
        caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
        """

        # Initialize "caches", which will track the list of all the caches
        caches = []

        # Retrieve dimensions from shapes of x and parameters['Wy'] (≈2 lines)
        n_x, m, T_x = x.shape
        n_y, n_a = self.parameters['Wy'].shape

        # initialize "a", "c" and "y" with zeros (≈3 lines)
        a = np.zeros((n_a, m, T_x))
        c = np.zeros((n_a, m, T_x))
        y = np.zeros((n_y, m, T_x))

        # Initialize a_next and c_next (≈2 lines)
        a_next = a0
        c_next = np.zeros((n_a, m))

        # loop over all time-steps
        for t in range(T_x):
            # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
            a_next, c_next, yt, cache = self.lstm_cell_forward(xt=x[:, :, t], a_prev=a_next, c_prev=c_next)
            # Save the value of the new "next" hidden state in a (≈1 line)
            a[:, :, t] = a_next
            # Save the value of the prediction in y (≈1 line)
            y[:, :, t] = yt
            # Save the value of the next cell state (≈1 line)
            c[:, :, t] = c_next
            # Append the cache into caches (≈1 line)
            caches.append(cache)

        # store values needed for backward propagation in cache
        caches = (caches, x)

        return a, y, c, caches

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

    def lstm_cell_backward(self, dz, da_next, dc_next, cache):
        """
        Implement the backward pass for the LSTM-cell (single time-step).

        Arguments:
        da_next -- Gradients of next hidden state, of shape (n_a, m)
        dc_next -- Gradients of next cell state, of shape (n_a, m)
        cache -- cache storing information from the forward pass

        Returns:
        gradients -- python dictionary containing:
            dxt -- Gradient of input data at time-step t, of shape (n_x, m)
            da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
            dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
            dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
            dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
            dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
            dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
            dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
            dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
            dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
            dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
        """

        # Retrieve information from "cache"
        (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt) = cache

        # Retrieve dimensions from xt's and a_next's shape (≈2 lines)
        n_a, m = a_next.shape

        dWy = np.dot(dz, a_next.T)
        dby = np.sum(dz, axis=1, keepdims=True)
        # cell的da由两部分组成，
        da_next = np.dot(self.parameters['Wy'].T, dz) + da_next

        # Compute gates related derivatives, you can find their values can be found by looking carefully at equations (7) to (10) (≈4 lines)
        dot = da_next * np.tanh(c_next) * ot * (1 - ot)
        dcct = (dc_next * it + ot * (1 - np.square(np.tanh(c_next))) * it * da_next) * (1 - np.square(cct))
        dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)
        dft = (dc_next * c_prev + ot * (1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)

        # Compute parameters related derivatives. Use equations (11)-(14) (≈8 lines)
        concat = np.vstack((a_prev, xt)).T
        dWf = np.dot(dft, concat)
        dWi = np.dot(dit, concat)
        dWc = np.dot(dcct, concat)
        dWo = np.dot(dot, concat)
        dbf = np.sum(dft, axis=1, keepdims=True)
        dbi = np.sum(dit, axis=1, keepdims=True)
        dbc = np.sum(dcct, axis=1, keepdims=True)
        dbo = np.sum(dot, axis=1, keepdims=True)

        # Compute derivatives w.r.t previous hidden state, previous memory state and input. Use equations (15)-(17). (≈3 lines)
        da_prev = np.dot(self.parameters['Wf'][:, :n_a].T, dft) + np.dot(self.parameters['Wi'][:, :n_a].T, dit) + np.dot(self.parameters['Wc'][:, :n_a].T, dcct) + np.dot(self.parameters['Wo'][:, :n_a].T, dot)
        dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next
        dxt = np.dot(self.parameters['Wf'][:, n_a:].T, dft) + np.dot(self.parameters['Wi'][:, n_a:].T, dit) + np.dot(self.parameters['Wc'][:, n_a:].T, dcct) + np.dot(self.parameters['Wo'][:, n_a:].T, dot)

        # Save gradients in dictionary
        gradients = {"dxt": dxt, "da_next": da_prev, "dc_next": dc_prev, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                     "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo, "dWy": dWy, "dby": dby}

        return gradients

    def lstm_backward(self, y, y_hat, caches):

        """
        Implement the backward pass for the RNN with LSTM-cell (over a whole sequence).

        Arguments:
        da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
        dc -- Gradients w.r.t the memory states, numpy-array of shape (n_a, m, T_x)
        caches -- cache storing information from the forward pass (lstm_forward)

        Returns:
        gradients -- python dictionary containing:
                dx -- Gradient of inputs, of shape (n_x, m, T_x)
                da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                dWo -- Gradient w.r.t. the weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                dbo -- Gradient w.r.t. biases of the save gate, of shape (n_a, 1)
        """

        # Retrieve values from the first cache (t=1) of caches.
        (caches, x) = caches
        (a1, c1, a0, c0, f1, i1, cc1, o1, x1) = caches[0]

        # Retrieve dimensions from da's and x1's shapes (≈2 lines)
        n_x, m, T_x = x.shape
        n_a = self.n_a
        # initialize the gradients with the right sizes (≈12 lines)
        dx = np.zeros((n_x, m, T_x))
        da0 = np.zeros((n_a, m))
        da_next = np.zeros((n_a, m))
        dc_next = np.zeros((n_a, m))
        dWf = np.zeros((n_a, n_a + n_x))
        dWi = np.zeros((n_a, n_a + n_x))
        dWc = np.zeros((n_a, n_a + n_x))
        dWo = np.zeros((n_a, n_a + n_x))
        dWy = np.zeros((self.n_y, n_a))
        dbf = np.zeros((n_a, 1))
        dbi = np.zeros((n_a, 1))
        dbc = np.zeros((n_a, 1))
        dbo = np.zeros((n_a, 1))
        dby = np.zeros((self.n_y, 1))
        dz = y_hat - y  # y_hat=softmax(z), dz=dl/dy_hat * dy_hat/dz

        # loop back over the whole sequence
        for t in reversed(range(T_x)):
            # Compute all gradients using lstm_cell_backward
            gradients = self.lstm_cell_backward(dz=dz[:, :, t], da_next=da_next, dc_next=dc_next, cache=caches[t])
            # Store or add the gradient to the parameters' previous step's gradient
            dx[:, :, t] = gradients["dxt"]
            dWf = dWf+gradients["dWf"]
            dWi = dWi+gradients["dWi"]
            dWc = dWc+gradients["dWc"]
            dWo = dWo+gradients["dWo"]
            dWy = dWy+gradients["dWy"]
            dbf = dbf+gradients["dbf"]
            dbi = dbi+gradients["dbi"]
            dbc = dbc+gradients["dbc"]
            dbo = dbo+gradients["dbo"]
            dby = dby+gradients["dby"]
            da_next = gradients['da_next']
            dc_next = gradients['dc_next']

        # Set the first activation's gradient to the backpropagated gradient da_prev.
        da0 = gradients['da_next']

        gradients = {"dx": dx, "da0": da0, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                     "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo, "dWy": dWy, "dby": dby}

        return gradients

    def update_parameters(self, gradients):
        """
        梯度下降
        :param gradients:
        :return:
        """
        self.parameters['Wf'] += -self.alpha * gradients["dWf"]
        self.parameters['Wi'] += -self.alpha * gradients["dWi"]
        self.parameters['Wc'] += -self.alpha * gradients['dWc']
        self.parameters['Wo'] += -self.alpha * gradients["dWo"]
        self.parameters['Wy'] += -self.alpha * gradients['dWy']

        self.parameters['bf'] += -self.alpha * gradients['dbf']
        self.parameters['bi'] += -self.alpha * gradients['dbi']
        self.parameters['bc'] += -self.alpha * gradients['dbc']
        self.parameters['bo'] += -self.alpha * gradients['dbo']
        self.parameters['by'] += -self.alpha * gradients['dby']

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
        a, y_pred, c, caches = self.lstm_forward(X, a_prev)
        # 计算损失
        loss = self.compute_loss(y_hat=y_pred, y=Y)

        gradients = self.lstm_backward(Y, y_pred, caches)

        # gradients = self.clip(gradients=gradients, maxValue=5)

        self.update_parameters(gradients)

        return loss, gradients, a[:, :, -1]
