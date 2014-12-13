import numpy as np
import cPickle

import gflags
FLAGS = gflags.FLAGS
gflags.DEFINE_string('floatX', 'float32', 'use float32 (for speed) or float64 (for testing)')

def FLOAT():
    if FLAGS.floatX == 'float32':
        return np.float32
    else:
        assert FLAGS.floatX == 'float64'
        return np.float64


def normal(size, bound):
    '''Initialize a matrix shared variable with normally distributed
elements.'''
    return np.random.normal(0, bound, size=size).astype(FLOAT())

def uniform(size, bound):
    return np.random.uniform(-bound, bound, size=size).astype(FLOAT())


def zeros(shape):
    '''Initialize a vector shared variable with zero elements.'''
    return np.zeros(shape).astype(FLOAT())


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
    outputs = zeros((len(inputs), 10))
    for i, y in enumerate(Y):
        assert y >= 0 and y <= 9
        outputs[i][y] = 1

    return inputs.astype(FLOAT()), outputs.astype(FLOAT())


def max_argmax(array, start_i, end_i, start_j, end_j):
    ind = (start_i, start_j)
    ret = array[ind]
    for i in xrange(start_i, end_i):
        for j in xrange(start_j, end_j):
            if array[i][j] > ret:
                ret = array[i][j]
                ind = (i, j)
    return ret, ind


def softmax(a):
    logsum = np.log(np.sum(np.exp(a), axis=1))
    ret = np.exp(a - np.expand_dims(logsum, axis=1))
    return ret
