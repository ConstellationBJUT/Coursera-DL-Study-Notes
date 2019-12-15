"""
@Time : 2019/12/15 5:09 PM 
@Author : bjjoy2009
加载模型，生成文本
"""

from keras.models import load_model
import numpy as np
import random

with open('../shakespeare.txt', 'r') as f:
    text = f.read()

print('corpus length:', len(text))

chars = sorted(list(set(text)))
char_num = len(chars)
print('total chars:', char_num)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40  # 需要和kears_lstm_train.py里边一样

# build the model: a single LSTM
print('load model...')
model = load_model('shakes_model.h5')


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


def generate_output(text_len=10):
    """
    生成文本
    :param text_len: 生成的text文本长度
    :return:
    """
    diversity = 1.0
    generated = ''  # 最终生成的文本
    # 从文本中随机选取位置截取maxlen字符作为初始输入句子
    start_index = random.randint(0, len(text) - maxlen - 1)

    sentence = text[start_index: start_index + maxlen]

    for i in range(text_len):
        # 输入句子x_pred
        x_pred = np.zeros((1, maxlen, len(chars)))
        # 生成one_hot向量
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.
        # 预测输出
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
        # 将句子字符串左移一位，新字符加载句子末尾，作为新的输入
        sentence = sentence[1:] + next_char
        generated += next_char
    return generated


text_len = random.randint(4, 200)
print('text len:', text_len)
new_text = generate_output(text_len)
print(new_text)