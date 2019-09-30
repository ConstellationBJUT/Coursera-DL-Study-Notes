"""
@Time : 2019/9/18 3:58 PM 
@Author : bjjoy2009
手写lr二分类，采用sklearn datasets.iris数据
iris数据格式：共150条数据，分3类，前50的label是0，中间50的label是1，后50的label是2
特征是4维向量，格式[1.2, 2.2, 3.3, 4.2]
具体格式如下：
|特征1|特征2|特征3|特征4|label|
|1.2 |2.2 |3.3 |4.2 |  0  |
"""
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression


def sk_lr():
    """
    调用sklearn LR，训练模型，打印准确率
    :return:
    """
    data = datasets.load_iris()
    X = data['data'][50:150]
    Y = data['target'][50:150]
    lr = LogisticRegression()
    lr.fit(X, Y)
    print('sklear-lr:', lr.score(X[0:100], Y[0:100]))

# 生成结果，与my_lr结果进行对比
# sk_lr()

class MyLr:
    """
    自定义lr类
    method: 可选梯度下降算法(gb, mini, momentum, RMSprop, adam)
    m: 训练集数量
    nx: 数据X特征数
    W: 参数矩阵,维度(1, nx)
    b: bias 标量
    alpha: 梯度下降学习率
    threshold: 二分类阈值
    epoch: 批量梯度下降数据集遍历次数
    batch: 批量梯度下降数据集分块大小
    print_loss: 是否打印每次迭代损失
    beta1: momentum算法需要的参数
    beta2: RMSprop算法需要的参数
    """
    def __init__(self, method='gb', alpha=0.1, max_iter=10, threshold=0.5, batch=128, epoch=10, print_loss=False,
                 beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.method = method
        self.m = 0
        self.nx = 0
        self.W = np.zeros((1, self.nx))
        self.b = 0
        self.alpha = alpha
        self.max_iter = max_iter
        self.threshold = threshold
        self.batch = batch
        self.epoch = epoch
        self.print_loss = print_loss
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def sigmoid(self, z):
        sg = 1/(1 + np.exp(-z))
        return sg

    def loss(self, y, ym):
        result = -1/len(y) * (np.dot(y, np.log(ym).T) + np.dot(1-y, np.log(1-ym).T))
        return result

    def gd(self, X, Y, W, b):
        """
        普通梯度下降
        :return: W, b
        """
        # 梯度下降
        for i in range(self.max_iter):
            z = np.dot(W, X.T) + b
            a = self.sigmoid(z)
            dz = a - Y
            dw = 1/self.m * np.dot(dz, X)
            db = 1/self.m * np.sum(dz)
            W = W - self.alpha * dw
            b = b - self.alpha * db
            if self.print_loss:
                print(self.loss(Y, a))
        return W, b

    def mini_batch_gd(self, X, Y, W, b):
        """
        批量梯度下降
        :return: W, b
        """
        batch_num = int(self.m/self.batch) + 1
        for i in range(self.epoch):
            A = []
            for k in range(batch_num):
                # 根据batch编号，获取每个batch训练集
                start = k * self.batch
                if start >= self.m:
                    break
                end = start + self.batch
                if end >= self.m:
                    end = self.m
                x = X[start:end]
                y = Y[start:end]

                z = np.dot(W, x.T) + b
                a = self.sigmoid(z)
                dz = a - y
                dw = 1/len(y) * np.dot(dz, x)
                db = 1/len(y) * np.sum(dz)
                W = W - self.alpha * dw
                b = b - self.alpha * db
                A.extend(a.ravel())
            if self.print_loss:
                print(i, k, self.loss(Y, np.array(A)))
        return W, b

    def momentum(self, X, Y, W, b):
        """
        momentum梯度下降
        :return: W, b
        """
        Vdw = np.zeros(W.shape)
        Vdb = 0
        batch_num = int(self.m/self.batch) + 1
        for i in range(self.epoch):
            A = []
            for k in range(batch_num):
                # 根据batch编号，获取每个batch训练集
                start = k * self.batch
                if start >= self.m:
                    break
                end = start + self.batch
                if end >= self.m:
                    end = self.m
                x = X[start:end]
                y = Y[start:end]

                z = np.dot(W, x.T) + b
                a = self.sigmoid(z)
                dz = a - y
                dw = 1/len(y) * np.dot(dz, x)
                db = 1/len(y) * np.sum(dz)
                Vdw = self.beta1 * Vdw + (1-self.beta1)*dw
                Vdb = self.beta1 * Vdb + (1-self.beta1)*db
                W = W - self.alpha * Vdw
                b = b - self.alpha * Vdb
                A.extend(a.ravel())
            if self.print_loss:
                print(i, k, self.loss(Y, np.array(A)))
        return W, b

    def RMSprop(self, X, Y, W, b):
        """
        RMSprop梯度下降
        :return: W, b
        """
        Sdw = np.zeros(W.shape)
        Sdb = 0
        batch_num = int(self.m/self.batch) + 1
        for i in range(self.epoch):
            A = []
            for k in range(batch_num):
                # 根据batch编号，获取每个batch训练集
                start = k * self.batch
                if start >= self.m:
                    break
                end = start + self.batch
                if end >= self.m:
                    end = self.m
                x = X[start:end]
                y = Y[start:end]

                z = np.dot(W, x.T) + b
                a = self.sigmoid(z)
                dz = a - y
                dw = 1/len(y) * np.dot(dz, x)
                db = 1/len(y) * np.sum(dz)
                Sdw = self.beta2 * Sdw + (1-self.beta2)*(dw**2)
                Sdb = self.beta2 * Sdb + (1-self.beta2)*(db**2)
                W = W - self.alpha * (dw/(np.sqrt(Sdw)+self.epsilon))
                b = b - self.alpha * (db/(np.sqrt(Sdb)+self.epsilon))
                A.extend(a.ravel())
            if self.print_loss:
                print(i, k, self.loss(Y, np.array(A)))
        return W, b

    def Adam(self, X, Y, W, b):
        """
        Adam梯度下降
        :return: W, b
        """
        Vdw = np.zeros(W.shape)
        Vdb = 0
        Sdw = np.zeros(W.shape)
        Sdb = 0
        batch_num = int(self.m/self.batch) + 1
        for i in range(self.epoch):
            A = []
            for k in range(batch_num):
                # 根据batch编号，获取每个batch训练集
                start = k * self.batch
                if start >= self.m:
                    break
                end = start + self.batch
                if end >= self.m:
                    end = self.m
                x = X[start:end]
                y = Y[start:end]

                z = np.dot(W, x.T) + b
                a = self.sigmoid(z)
                dz = a - y
                dw = 1/len(y) * np.dot(dz, x)
                db = 1/len(y) * np.sum(dz)

                Vdw = self.beta1 * Vdw + (1-self.beta1)*dw
                Vdb = self.beta1 * Vdb + (1-self.beta1)*db

                Sdw = self.beta2 * Sdw + (1-self.beta2)*(dw**2)
                Sdb = self.beta2 * Sdb + (1-self.beta2)*(db**2)

                correct_Vdw = Vdw/(1-self.beta1**(i+1))
                correct_Vdb = Vdb/(1-self.beta1**(i+1))

                correct_Sdw = Sdw/(1-self.beta2**(i+1))
                correct_Sdb = Sdb/(1-self.beta2**(i+1))

                W = W - self.alpha * (correct_Vdw/(np.sqrt(correct_Sdw)+self.epsilon))
                b = b - self.alpha * (correct_Vdb/(np.sqrt(correct_Sdb)+self.epsilon))
                A.extend(a.ravel())
            if self.print_loss:
                print(i, k, self.loss(Y, np.array(A)))
        return W, b

    def get_gb(self, gb_name):
        gb_dict = {
            "gb": self.gd,
            "mini": self.mini_batch_gd,
            "momentum": self.momentum,
            "RMSprop": self.RMSprop,
            "Adam": self.Adam
        }
        return gb_dict.get(gb_name)

    def lr_fit(self, X, Y):
        m, nx = X.shape
        self.m = m
        self.nx = nx
        W = np.random.random((1, nx))*0.1
        b = 0
        self.W, self.b = self.get_gb(self.method)(X, Y, W, b)

    def predict(self, samples):
        """
        给出输入数据的预测结果
        :param samples:
        :return: list
        """
        result_proba = self.sigmoid(np.dot(self.W, samples.T) + self.b)
        result_list = (result_proba > self.threshold)
        return result_list

    def score(self, samples, y, print_error=False):
        """
        计算给定数据预测准确率
        :param samples: 给定数据特征
        :param y: 实际结果
        :param print_error: 是否打印错误信息
        :return:
        """
        ym = self.predict(samples)
        # yes = 0
        # y = list(y)
        # for i in range(len(ym)):
        #     if y[i] == ym[i]:
        #         yes += 1
        #     elif print_error:
        #         print(i, samples[i], y[i], ym[i])
        yes = np.dot(y, ym.T) + np.dot(1-y, 1-ym.T)
        return float(yes)/ym.size


# 本实验预测是否是label=1，取50~150共100个数据，50~100是正例，100~150是负例
data = datasets.load_iris()
X = data['data'][50:150]
Y = data['target'][50:150]
# 将label写到Y2中
Y2 = list(Y[0:50])
# 由于是二分类前50label是1，需要修改后50个label变为0
for item in list(Y[50:100]):
    Y2.append(0)
# 固定随机种子，比较各梯度下降算法
np.random.seed(1)
model = MyLr(method='Adam', alpha=0.1, epoch=100, batch=32, print_loss=False)
model.lr_fit(X, np.array(Y2))
print(model.method, model.score(X, np.array(Y2), print_error=False))

