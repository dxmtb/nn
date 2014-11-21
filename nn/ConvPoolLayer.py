import numpy as np
import util
from util import max_argmax
from scipy.signal import convolve2d as conv


def conv_valid(*args):
    return conv(*args, mode='valid')


def conv_full(*args):
    return conv(*args, mode='full')


class ConvPoolLayer(object):
    def __init__(self, filter_shape, poolsize, activation, grad_activation):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of output filters, num input feature maps,
                              filter height, filter width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """
        fan_in = np.prod(filter_shape[1:])
        fan_out = ((filter_shape[0] * np.prod(filter_shape[2:])) /
                   (np.prod(poolsize)))
        bound = np.sqrt(6.0 / (fan_in + fan_out))
        self.W = util.uniform(filter_shape, bound)
        self.b = util.zeros((filter_shape[0],))
        self.activation = activation
        self.grad_activation = grad_activation
        self.poolsize = poolsize
        self.filter_shape = filter_shape

        self.params = ['W', 'b']

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
        after_filter = util.zeros((batch_size, n_output, n_height, n_width))
        for index in np.ndindex(batch_size, n_output):
            i, q = index
            result = after_filter[index]
            for p in xrange(n_input):
                result += conv_valid(feature_maps[i][p], np.rot90(W[q][p], 2))
            result += b[q]
        after_filter = self.activation(after_filter)

        # do pooling
        # borders are ignored
        ret_h = int(float(n_height) / pool_h)
        ret_w = int(float(n_width) / pool_w)
        ret = util.zeros((batch_size, n_output, ret_h, ret_w))
        M = util.zeros((batch_size, n_output, n_height, n_width))

        for i, j, h, w in np.ndindex(ret.shape):
            ret[i][j][h][w], ind = max_argmax(after_filter[i][j],
                                              h*pool_h, (h+1)*pool_h,
                                              w*pool_w, (w+1)*pool_w)
            M[i][j][ind] = 1

        self.M = M

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

        ret = np.zeros((batch_size, n_input, height, width))
        for i, j in np.ndindex(batch_size, n_input):
            result = ret[i][j]
            for k in xrange(n_output):
                result += conv_full(error_before_pooling[i][k], W[k][j])
            ret[i][j] = np.multiply(result, grad_activation(input[i][j]))

        return ret

    def grad(self, error_output, input):
        """
        input: (batch_size, # of input feature maps, height, width)
        error_output: (batch_size, # of filters,
                       h_after_filter, h_after_filter)
        """
        pool_h, pool_w = self.poolsize
        error_before_pooling = np.copy(self.M)
        for i, j, h, w in np.ndindex(error_before_pooling.shape):
            if error_before_pooling[i][j][h][w] == 1:
                error_before_pooling[i][j][h][w] = \
                    error_output[i][j][h/pool_h][w/pool_w]

        error_output = error_before_pooling
        self.error_before_pooling = error_before_pooling

        batch_size = input.shape[0]

        W_grad = np.zeros(self.W.shape)
        for q, p in np.ndindex(self.W.shape[:2]):
            for i in xrange(batch_size):
                W_grad[q][p] += conv_valid(input[i][p],
                                           np.rot90(error_output[i][q], 2))

        b_grad = np.sum(error_output, axis=(0, 2, 3))

        self.W_grad = W_grad
        self.b_grad = b_grad

    def do_update(self, learning_rate):
        self.W -= learning_rate * self.W_grad
        self.b -= learning_rate * self.b_grad
