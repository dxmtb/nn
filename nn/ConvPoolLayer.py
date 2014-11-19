import numpy as np
import util
from util import pooling
from scipy.signal import convolve2d as conv


def conv_valid(*args):
    return conv(*args, mode='valid')


def conv_full(*args):
    return conv(*args, mode='full')


class ConvPoolLayer(object):
    def __init__(self, filter_shape, poolsize, activation, grad_activation, image_shape):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of output filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

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
        self.image_shape =  image_shape[2:]

    def activate(self, feature_maps):
        """
        feature_maps: (batch_size, # of input feature maps, height, width)
        after conv: (batch_size, # of filters, new h, new w)
        after pooling: (batch_size, # of filters, new h / poolsize, new w / poolsize)
        """
        W = self.W
        b = self.b
        pool_h, pool_w = self.poolsize
        batch_size, n_input, height, width = feature_maps.shape
        n_output, n_input_check, f_height, f_width = W.shape
        n_height = height - f_height + 1
        n_width = width - f_width + 1
        assert n_input == n_input_check
        assert (height, width) == self.image_shape

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
        ret_h = int(math.ceil(float(n_height) / poolsize[0]))
        ret_w = int(math.ceil(float(n_width) / poolsize[1]))
        ret = util.zeros((batch_size, n_output, ret_h, ret_w))
        M = util.zeros((batch_size, n_output, n_height, n_width))

        for i in xrange(batch_size):
            for j in xrange(n_output):
                for h in xrange(ret_h):
                    for w in xrange(ret_w):
                        ret[i][j][h][w], ind = \
                                max_argmax(after_filter[i][j], 
                                           h*pool_h, (h+1)*pool_h,
                                           w*pool_w, (w+1)*pool_w)
                        M[i][j][ind] = 1

        return (ret, after_filter, M)

    def error(self, error_output, output):
        """
        error_output: (batch_size, # of filters, h_after_pool, w_after_pool)
        output: (batch_size, # of filters, h_after_pool, w_after_pool)
        M, output_before_pooling: (batch_size, # of filters, h_after_filter, w_after_filter)
        return: (batch_size, # of input feature_maps, height, width)
        """
        output, output_before_pooling, M = output
        pool_h, pool_w = self.poolsize
        height, width = self.image_shape

        error_before_pooling = np.copy(M)
        for i, j in np.ndindex(error_before_pooling.shape[2:]):
            if error_before_pooling[i][j] == 1:
               error_before_pooling[i][j] = error_output[i/pool_h][j/pool_w]

        ret = np.zeros((batch_size, n_input, height, width))
        for i, j in np.ndindex(batch_size, n_input):
            result = ret[i][j]
            for k in xrange(n_output):
                result += conv_full(error_before_pooling[i][k], W[k][j])
            ret[i][j] = np.multiply(result, grad_activation(output_before_pooling[i][j]))

        return ret

    def grad(self, error_output, input):
        """
        input: (batch_size, # of input feature maps, height, width)
        error_output: (batch_size, # of filters, h_after_pool, w_after_pool)
        """
        batch_size = error_output.shape[0]
        W_grad = np.zeros(self.W.shape)
        error_output = np.sum(error_output, axis=0)
        for i, j in np.nditer(self.W.shape[2:]):
            W_grad[i][j] = conv_valid(input[j], np.rot90(error_output[i], 2))
        b_grad = np.sum(error_output, axis=(0, 2, 3))
        self.W_grad = W_grad
        self.b_grad = b_grad

    def do_update(self, learning_rate):
        self.W -= learning_rate * self.W_grad
        self.b -= learning_rate * self.b_grad

