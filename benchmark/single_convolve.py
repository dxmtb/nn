import scipy.signal
import numpy as np
import time
import theano
import theano.tensor as T
from theano import config
from theano.tensor.nnet import conv

TEST_NUM=1000

def benchmark_scipy(W, feature_maps):
    t0 = time.time()

    for _ in xrange(TEST_NUM):
        after_filter = scipy.signal.convolve2d(feature_maps[0][0], W[0][0], mode='valid')

    t1 = time.time()
    print t1-t0

    return after_filter

def benchmark_theano(W, feature_maps):
    inputs=T.tensor4('feature_maps')
    filters=T.tensor4('W')

    conv_out = conv.conv2d(
        input=inputs,
        filters=filters,
        filter_shape=(1, 1, 5, 5),
        image_shape=(1, 1, 32, 32)
    )

    func = theano.function([inputs, filters], conv_out)

    t0 = time.time()

    for _ in xrange(TEST_NUM):
        after_filter = func(feature_maps, W)

    t1 = time.time()
    print t1-t0

    return after_filter

if __name__ == '__main__':
    W = np.random.uniform(low=-0.01, high=0.01, size=(1, 1, 5, 5)).astype(np.float32)
    feature_maps = np.random.randint(0, 255, (1, 1, 32, 32)).astype(np.float32)

    r1 = benchmark_scipy(W, feature_maps)
    r2 = benchmark_theano(W, feature_maps)
    r2 = r2.reshape(r1.shape)
    np.testing.assert_array_almost_equal(r1, r2, decimal=5)

    

