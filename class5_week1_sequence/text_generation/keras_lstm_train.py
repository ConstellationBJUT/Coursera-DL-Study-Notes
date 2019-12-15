"""
@Time : 2019/12/15 4:35 PM 
@Author : bjjoy2009
keras框架实现生成shakespeare文本
整体思想：用连续40个字符生成1个字符，直到生成指定长度字符串，作为句子文本。
训练模型，保存模型
参考 https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
"""

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np

# 加载数据文件
with open('../shakespeare.txt', 'r') as f:
    text = f.read()

print('corpus length:', len(text))

# 生成字符_id字典，有标点符号和a~z组成
chars = sorted(list(set(text)))
char_num = len(chars)
print('total chars:', char_num)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40  # 每个句子由40个字符组成，也就是有40个cell
step = 3  # 每隔3个字符，获取maxlen个字符作为一个句子
sentences = []  # 存放句子的数组，每个句子是一个输入x
next_chars = []  # label，每个句子后边紧跟的字符作为输出y
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
# 训练数据转换为模型认识的格式，每个句子对应一个输出字符
# 训练数据x，维度(句子数，cell数，xi维度)
x = np.zeros((len(sentences), maxlen, char_num), dtype=np.bool)
# 训练label y，维度(句子数，yi维度)
y = np.zeros((len(sentences), char_num), dtype=np.bool)
# 将x,y转为one_hot，每个输入句子由maxlen个one_hot向量组成
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, char_num)))
model.add(Dense(char_num, activation='softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    """
    helper function to sample an index from a probability array
    :param preds: 模型正向传播计算得到的向量a，维度(char_num, 1)
    :param temperature: 多样性控制，值越大，随机性越强
    :return:
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    # 下面个两行做了softmax
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # 按照概率preds做一次随机抽样试验，返回一个和preds维度相同0，1向量，1表示该位置选中
    probas = np.random.multinomial(1, preds, 1)
    # 返回1所在位置，根据位置可以去找到对应字符
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    None


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=10,
          callbacks=[print_callback])

# 存储模型
model.save('shakes_model.h5')