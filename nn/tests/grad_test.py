import unittest
import numpy as np
from nn MLP import MLP
from nn.CNN import CNN
from nn import util

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')

import gflags
FLAGS = gflags.FLAGS

class GradientTestCase(unittest.TestCase):
    def testGrad(self):
        nn = self.nn
        EPSILON = self.EPSILON
        X_train = self.X_train
        y_train = self.y_train

        nn.train_batch(X_train, y_train, lr=None, update=False)

        mat, mat_grad, ind = nn.sample_parameter()
        grad = mat_grad[ind]

        mat[ind] += EPSILON
        nn.train_batch(X_train, y_train, lr=None, update=False)
        loss_plus = nn.loss(X_train, y_train)

        mat[ind] -= 2 * EPSILON
        nn.train_batch(X_train, y_train, lr=None, update=False)
        loss_minus = nn.loss(X_train, y_train)

        real_grad = (loss_plus - loss_minus) / (2 * EPSILON)

        np.testing.assert_approx_equal(grad, real_grad, significant=4)
        # logging.info('Expected %f BP %f' % (real_grad, grad))

class MLPGradientTestCase(GradientTestCase):
    def __init__(self, activation, loss_type, methodName='testGrad', EPSILON=1e-4):
        super(MLPGradientTestCase, self).__init__(methodName)
        self.activation = activation
        self.loss_type = loss_type
        self.EPSILON = EPSILON

    def setUp(self):
        in_dim, hidden_dim, out_dim = np.random.randint(3, 5, size=3)
        self.nn = MLP(in_dim, hidden_dim, out_dim, self.activation, self.loss_type)
        self.X_train = util.uniform((1, in_dim), 1.0)
        if self.loss_type == 'softmax':
            self.y_train = util.zeros((1, out_dim))
            self.y_train[0][np.random.randint(out_dim)] = 1.0
        else:
            self.y_train = util.uniform((1, out_dim), 1.0)
        for layer in nn.layers:
            layer.b = util.uniform(layer.b.shape, 0.5)

        self.X_train = self.X_train.astype(util.FLOAT())
        self.y_train = self.y_train.astype(util.FLOAT())

class CNNGradientTestCase(GradientTestCase):
    def __init__(self, activation, loss_type, methodName='testGrad', EPSILON=1e-7):
        super(CNNGradientTestCase, self).__init__(methodName)
        self.activation = activation
        self.loss_type = loss_type
        self.EPSILON = EPSILON

    def setUp(self):
        FLAGS.floatX = 'float64'
        out_dim = 10
        self.nn = CNN(10, self.activation, self.loss_type)
        self.X_train = util.uniform((1, 3, 32, 32), 10.0)
        if self.loss_type == 'softmax':
            self.y_train = util.zeros((1, out_dim))
            self.y_train[0][np.random.randint(out_dim)] = 1.0
        else:
            self.y_train = util.uniform((1, out_dim), 1.0)

        self.X_train = self.X_train.astype(util.FLOAT())
        self.y_train = self.y_train.astype(util.FLOAT())

def suite(activation, loss_type):
    suite = unittest.TestSuite()
    for _ in xrange(10):
        suite.addTest(CNNGradientTestCase(activation, loss_type))
        # suite.addTest(MLPGradientTestCase(activation, loss_type))
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite('tanh', 'mse'))
    runner.run(suite('sigmoid', 'mse'))
    runner.run(suite('tanh', 'softmax'))
    runner.run(suite('sigmoid', 'softmax'))
