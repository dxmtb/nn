import numpy as np
import util
from util import max_argmax
import logging
import scipy.signal

import gflags
FLAGS = gflags.FLAGS
gflags.DEFINE_bool('theano_conv', True, 'whether to use theano for convolve2d')

def upsample(error_before_pooling, error_output, pool_h, pool_w):
    tmp = np.kron(error_output, np.ones((1, 1, pool_h, pool_w)))
    tmp = tmp[:error_before_pooling.shape[0],
              :error_before_pooling.shape[1],
              :error_before_pooling.shape[2],
              :error_before_pooling.shape[3]]
    np.multiply(tmp, error_before_pooling, out=error_before_pooling)

def py_do_pooling(after_filter, poolsize):
    batch_size, n_output, n_height, n_width = after_filter.shape
    pool_h, pool_w = poolsize
    ret_h = int(float(n_height) / pool_h)
    ret_w = int(float(n_width) / pool_w)
    ret = util.zeros((batch_size, n_output, ret_h, ret_w))
    M = util.zeros((batch_size, n_output, n_height, n_width))
    ret.fill(-np.inf)

    for i, j, h, w in np.ndindex(ret.shape):
        ret[i][j][h][w], ind = max_argmax(after_filter[i][j],
                                          h*pool_h, (h+1)*pool_h,
                                          w*pool_w, (w+1)*pool_w)
        M[i][j][ind] = 1

    return M, ret

import ctypes as ct
import os

try:
    _DLL = np.ctypeslib.load_library('pooling.so',
            os.path.join(os.path.dirname(__file__)))
except Exception as error:
    raise error

_DLL.do_pooling.restype = _DLL.do_pooling.restype = None

def ct_do_pooling(after_filter, poolsize):
    batch_size, n_output, n_height, n_width = after_filter.shape
    pool_h, pool_w = poolsize
    ret_h = int(float(n_height) / pool_h)
    ret_w = int(float(n_width) / pool_w)
    M = np.ascontiguousarray(util.zeros((batch_size, n_output, n_height, n_width)))
    ret = np.ascontiguousarray(util.zeros((batch_size, n_output, ret_h, ret_w)))
    assert after_filter.dtype == M.dtype
    _DLL.do_pooling(ct.c_int(after_filter.itemsize),
                    ct.c_int(batch_size),
                    ct.c_int(n_output),
                    ct.c_int(n_height),
                    ct.c_int(n_width),
                    ct.c_int(pool_h),
                    ct.c_int(pool_w),
                    ct.c_int(ret_h),
                    ct.c_int(ret_w),
                    after_filter.ctypes.data_as(ct.c_void_p),
                    ret.ctypes.data_as(ct.c_void_p),
                    M.ctypes.data_as(ct.c_void_p))
    return M, ret

def ct_upsample(error_before_pooling, error_output, pool_h, pool_w):
    batch_size, n_output, n_height, n_width = error_before_pooling.shape
    ret_h = int(float(n_height) / pool_h)
    ret_w = int(float(n_width) / pool_w)
    assert error_before_pooling.flags['C_CONTIGUOUS']
    assert error_output.flags['C_CONTIGUOUS']
    assert error_before_pooling.dtype == error_output.dtype
    _DLL.upsample(ct.c_int(error_before_pooling.itemsize),
                  ct.c_int(batch_size),
                  ct.c_int(n_output),
                  ct.c_int(n_height),
                  ct.c_int(n_width),
                  ct.c_int(pool_h),
                  ct.c_int(pool_w),
                  ct.c_int(ret_h),
                  ct.c_int(ret_w),
                  error_before_pooling.ctypes.data_as(ct.c_void_p),
                  error_output.ctypes.data_as(ct.c_void_p))

def conv_valid(*args):
    return scipy.signal.convolve2d(*args, mode='valid')

def conv_full(*args):
    return scipy.signal.convolve2d(*args, mode='full')

def conv2d_func(image_shape, filter_shape, mode):
    import theano
    import theano.tensor as T
    if FLAGS.floatX == 'float32':
        inputs = T.ftensor4()
        filters = T.ftensor4()
    else:
        inputs = T.dtensor4()
        filters = T.dtensor4()
    conv_out = theano.tensor.nnet.conv2d(
        input=inputs,
        filters=filters,
        filter_shape=filter_shape,
        image_shape=image_shape,
        border_mode=mode
    )
    func = theano.function([inputs, filters], conv_out)
    return func

def rot90(tensor):
    assert len(tensor.shape) == 4
    ret = tensor[:,:,::-1,::-1]
    return ret

class ConvPoolLayer(object):
    def __init__(self, image_shape, filter_shape, poolsize, bound, activation, grad_activation):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        image_shape: (batch_size, # of input feature_maps, height, width)

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of output filters, num input feature maps,
                              filter height, filter width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """
        if not bound:
            fan_in = np.prod(filter_shape[1:])
            fan_out = ((filter_shape[0] * np.prod(filter_shape[2:])) /
                       (np.prod(poolsize)))
            bound = np.sqrt(6.0 / (fan_in + fan_out))
        self.W = util.normal(filter_shape, bound)
        self.b = util.zeros((filter_shape[0],))
        self.activation = activation
        self.grad_activation = grad_activation
        self.poolsize = poolsize
        self.filter_shape = filter_shape
        self.output_shape = (image_shape[0], filter_shape[0],
                             (image_shape[2] - filter_shape[2] + 1) / poolsize[0],
                             (image_shape[3] - filter_shape[3] + 1) / poolsize[1])
        if FLAGS.theano_conv:
            self._conv2d_activate_valid = conv2d_func(image_shape, filter_shape, 'valid')
            self._conv2d_grad_valid = conv2d_func(
                (image_shape[1], image_shape[0], image_shape[2], image_shape[3]),
                (filter_shape[0], image_shape[0],
                 image_shape[2] - filter_shape[2] + 1,
                 image_shape[3] - filter_shape[3] + 1), 'valid')
            self._conv2d_full = conv2d_func(
                (image_shape[0], filter_shape[0],
                 image_shape[2] - filter_shape[2] + 1,
                 image_shape[3] - filter_shape[3] + 1),
                (filter_shape[1], filter_shape[0], filter_shape[2], filter_shape[3]),
                'full')

        self.params = ['W', 'b']
        self.W_inc_before = util.zeros(self.W.shape)
        self.b_inc_before = util.zeros(self.b.shape)

    def do_pooling(self, after_filter, poolsize):
        ret = ct_do_pooling(after_filter, poolsize)
        return ret

    def conv2d_activate_valid(self, *args):
        # wrap for profiling
        return self._conv2d_activate_valid(*args)

    def conv2d_grad_valid(self, *args):
        # wrap for profiling
        return self._conv2d_grad_valid(*args)

    def conv2d_full(self, *args):
        # wrap for profiling
        return self._conv2d_full(*args)

    def activate(self, feature_maps):
        """
        feature_maps: (batch_size, # of input feature maps, height, width)
        after conv: (batch_size, # of filters, new h, new w)
        after pooling: (batch_size, # of filters,
                        new h / poolsize, new w / poolsize)
        """
        W = self.W
        b = self.b
        pool_h, pool_w = self.poolsize
        n_output, n_input, f_height, f_width = self.filter_shape
        batch_size, _, height, width = feature_maps.shape
        n_height, n_width = (height - f_height + 1, width - f_width + 1)

        assert feature_maps.shape[1] == n_input

        # do convolve2d
        if FLAGS.theano_conv:
            after_filter = self.conv2d_activate_valid(feature_maps, rot90(W))
        else:
            after_filter = util.zeros((batch_size, n_output, n_height, n_width))
            for index in np.ndindex(batch_size, n_output):
                i, q = index
                result = after_filter[index]
                for p in xrange(n_input):
                    result += conv_valid(feature_maps[i][p], np.rot90(W[q][p], 2))

        after_filter += b[np.newaxis, :, np.newaxis, np.newaxis]
        after_filter = self.activation(after_filter)

        # do pooling
        # borders are ignored
        self.M, ret = self.do_pooling(after_filter, self.poolsize)

        return ret

    def error(self, _, input):
        """
        error_output: Not used because we use error_before_pooling
                      calculated in grad()
        M, error_before_pooling: (batch_size, # of filters,
                                  h_after_filter, w_after_filter)
        input: (batch_size, # of filters, height, width)
        return: (batch_size, # of input feature_maps, height, width)
        """
        batch_size, _, height, width = input.shape
        n_output, n_input, _, _ = self.filter_shape

        assert input.shape[1] == n_input

        W = self.W
        grad_activation = self.grad_activation
        error_before_pooling = self.error_before_pooling

        if FLAGS.theano_conv:
            ret = self.conv2d_full(error_before_pooling, np.swapaxes(W, 0, 1))
        else:
            ret = util.zeros((batch_size, n_input, height, width))
            for i, j in np.ndindex(batch_size, n_input):
                result = ret[i][j]
                for k in xrange(n_output):
                    result += conv_full(error_before_pooling[i][k], W[k][j])

        ret = np.multiply(ret, grad_activation(input))

        return ret

    def grad(self, error_output, input):
        """
        input: (batch_size, # of input feature maps, height, width)
        error_output: (batch_size, # of filters,
                       h_after_filter, h_after_filter)
        """
        import time
        pool_h, pool_w = self.poolsize
        error_before_pooling = np.copy(self.M)
        #upsample(error_before_pooling, error_output, pool_h, pool_w)
        ct_upsample(error_before_pooling, error_output, pool_h, pool_w)

        error_output = error_before_pooling
        self.error_before_pooling = error_before_pooling

        batch_size = input.shape[0]

        if FLAGS.theano_conv:
            # (p, i, :, :) (q, i, :, :)
            # (p, q, :, :)
            rot_error_output = rot90(error_output)
            W_grad = self.conv2d_grad_valid(np.swapaxes(input, 0, 1), np.swapaxes(rot_error_output, 0, 1))
            W_grad = np.swapaxes(W_grad, 0, 1)
        else:
            W_grad = util.zeros(self.W.shape)
            for i, q, p in np.ndindex(batch_size, *self.W.shape[:2]):
                W_grad[q][p] = conv_valid(input[i][p], np.rot90(error_output[i][q], 2))

        b_grad = np.sum(error_output, axis=(0, 2, 3))

        self.W_grad = W_grad / input.shape[0]
        self.b_grad = b_grad / input.shape[0]

    def do_update(self, learning_rate, momentum=0.9, weight_decay=0.004):
        for param in self.params:
            p = getattr(self, param)
            p_inc = -learning_rate * getattr(self, param+'_grad') + \
                    momentum * getattr(self, param+'_inc_before')
            if param != 'b':
                p_inc += -learning_rate * weight_decay * p
            setattr(self, param, p + p_inc)
            setattr(self, param+'_inc_before', p_inc)
