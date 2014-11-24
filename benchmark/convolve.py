import scipy.signal
import numpy as np
import time
import theano
import theano.tensor as T
from theano import config
from theano.tensor.nnet import conv

def benchmark_scipy(W, feature_maps, batch_size):
    t0 = time.time()

    after_filter = np.zeros((batch_size, 32, 28, 28)).astype(np.float32)
    for index in np.ndindex(batch_size, 32):
        i, q = index
        result = after_filter[index]
        for p in xrange(3):
            result += scipy.signal.convolve2d(feature_maps[i][p], W[q][p], mode='valid')

    t1 = time.time()
    print t1-t0

    return after_filter

def benchmark_theano(W, feature_maps, batch_size):
    inputs=T.tensor4('feature_maps')
    filters=T.tensor4('W')

    conv_out = conv.conv2d(
        input=inputs,
        filters=filters,
        filter_shape=(32, 3, 5, 5),
        image_shape=(batch_size, 3, 32, 32)
    )

    func = theano.function([inputs, filters], conv_out)

    t0 = time.time()

    after_filter = func(feature_maps, W)

    t1 = time.time()
    print t1-t0

    return after_filter

if __name__ == '__main__':
    batch_size = 100
    W = np.random.uniform(low=-0.01, high=0.01, size=(32, 3, 5, 5)).astype(np.float32)
    feature_maps = np.random.randint(0, 255, (batch_size, 3, 32, 32)).astype(np.float32)

    r1 = benchmark_scipy(W, feature_maps, batch_size)
    r2 = benchmark_theano(W, feature_maps, batch_size)
    np.testing.assert_array_almost_equal(r1, r2, decimal=5)

    
