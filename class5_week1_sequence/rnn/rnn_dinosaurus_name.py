"""
@Time : 2019/12/11 4:40 PM 
@Author : bjjoy2009
rnn实现生成恐龙名字，batch_size需要设置为1，不同长度名字，T_x不同
"""
import numpy as np
from class5_week1_sequence.rnn.rnn import RNN, get_initial_loss, smooth, softmax


def sample(parameters, char_to_ix, seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN
    生成名称字符串，每生成一个字符作为下一个cell的输入，直到字符是\n或者长度到50结束
    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b.
    char_to_ix -- python dictionary mapping each character to an index.
    seed -- used for grading purposes. Do not worry about it.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """

    # Retrieve parameters and relevant shapes from "parameters" dictionary
    Waa, Wax, Wya, by, ba = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['ba']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    # Step 1: Create the one-hot vector x for the first character (initializing the sequence generation). (≈1 line)
    x = np.zeros((vocab_size, 1))
    # Step 1': Initialize a_prev as zeros (≈1 line)
    a_prev = np.zeros((n_a, 1))

    # Create an empty list of indices, this is the list which will contain the list of indices of the characters to generate (≈1 line)
    indices = []

    # Idx is a flag to detect a newline character, we initialize it to -1
    idx = -1

    # Loop over time-steps t. At each time-step, sample a character from a probability distribution and append
    # its index to "indices". We'll stop if we reach 50 characters (which should be very unlikely with a well
    # trained model), which helps debugging and prevents entering an infinite loop.
    counter = 0
    newline_character = char_to_ix['\n']

    while idx != newline_character and counter != 50:

        # Step 2: Forward propagate x using the equations (1), (2) and (3)
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + ba)
        z = np.dot(Wya, a) + by
        y = softmax(z)

        # for grading purposes
        np.random.seed(counter+seed)

        # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
        idx = np.random.choice(range(vocab_size), p=y.ravel())

        # Append the index to "indices"
        indices.append(idx)

        # Step 4: Overwrite the input character as the one corresponding to the sampled index.
        x = np.zeros((vocab_size, 1))
        x[idx] = 1

        # Update "a_prev" to be "a"
        a_prev = a

        # for grading purposes
        seed += 1
        counter += 1

    if counter == 50:
        indices.append(char_to_ix['\n'])

    return indices


def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # capitalize first character
    print('%s' % (txt, ), end='')


def model(data, ix_to_char, char_to_ix, n_a=50, iter_num=35000, dino_names=7, vocab_size=27):
    """
    Trains the model and generates dinosaur names.

    Arguments:
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration.
    vocab_size -- number of unique characters found in the text, size of the vocabulary

    Returns:
    parameters -- learned parameters
    """
    rnn = RNN(n_a=n_a, batch_size=1)
    # Retrieve n_x and n_y from vocab_size
    n_x, n_y = vocab_size, vocab_size

    # Initialize parameters
    parameters = rnn.initialize_parameters(n_a, n_x, n_y)

    # Initialize loss (this is required because we want to smooth our loss, don't worry about it)
    loss = get_initial_loss(vocab_size, dino_names)

    # Build list of all dinosaur names (training examples).
    with open("../dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    # Shuffle list of all dinosaur names
    np.random.seed(0)
    np.random.shuffle(examples)

    # Initialize the hidden state of rnn
    a_prev = np.zeros((n_a, 1))

    # Optimization loop
    for j in range(iter_num):

        # Use the hint above to define one training example (X,Y) (≈ 2 lines)
        index = j % len(examples)
        x = [None] + [char_to_ix[ch] for ch in examples[index]]  # 输入的名字example是名字list
        y = x[1:] + [char_to_ix["\n"]]  # 对应的输出名字，x左移一位后补\n
        X_batch = np.zeros((n_x, 1, len(x)))  # x转为输入矩阵
        Y_batch = np.zeros((n_y, 1, len(x)))  # y转为label
        # 字符对应位置补1
        for t in range(len(x)):
            if x[t] is not None:
                X_batch[x[t], 0, t] = 1
            Y_batch[y[t], 0, t] = 1

        # 每个序列输入初始化loss=0
        rnn.loss = 0
        # 送入rnn训练
        curr_loss, gradients, a_prev = rnn.optimize(X=X_batch, Y=Y_batch, a_prev=a_prev)

        # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
        loss = smooth(loss, curr_loss)

        # Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
        if j % 2000 == 0:

            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

            # The number of dinosaur names to print
            seed = 0
            for name in range(dino_names):

                # Sample indices and print them
                sampled_indices = sample(rnn.parameters, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)

                seed += 1  # To get the same result for grading purposed, increment the seed by one.

            print('\n')

    return parameters


data = open('../dinos.txt', 'r').read()
data = data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
parameters = model(data, ix_to_char, char_to_ix)
