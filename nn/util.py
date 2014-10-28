import numpy as np
import cPickle

def uniform(num_rows, num_cols, bound):
    '''Initialize a matrix shared variable with normally distributed
elements.'''
    return np.random.uniform(low=-bound, high=bound, size=(num_rows, num_cols))

def zeros(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return np.zeros(shape)

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_CIFAR_batches(files):
    X = []
    Y = []
    for f in files:
        dic = unpickle(f)
        X.append(dic['data'])
        Y += dic['labels']
    inputs = np.concatenate(X)
    outputs = np.zeros((len(inputs), 10), dtype=float)
    for i, y in enumerate(Y):
        assert y >= 0 and y <= 9
        outputs[i][y] = 1

    return inputs, outputs

