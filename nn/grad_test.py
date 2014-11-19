import unittest
import numpy as np
from MLP import MLP
import util

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')

class GradientTestCase(unittest.TestCase):
    def __init__(self, nn_cls, activation, loss_type, methodName='runTest', EPSILON=1e-4):
        super(GradientTestCase, self).__init__(methodName)
        self.nn_cls = nn_cls
        self.activation = activation
        self.loss_type = loss_type
        self.EPSILON = EPSILON

    def setUp(self):
        in_dim, hidden_dim, out_dim = np.random.randint(3, 5, size=3)
        self.nn = self.nn_cls(in_dim, hidden_dim, out_dim, self.activation, self.loss_type)
        self.X_train = util.uniform((1, in_dim), 1.0)
        if self.loss_type == 'softmax':
            self.y_train = util.zeros((1, out_dim))
            self.y_train[0][np.random.randint(out_dim)] = 1.0
        else:
            self.y_train = util.uniform((1, out_dim), 1.0)

    def testGrad(self):
        nn = self.nn
        EPSILON = self.EPSILON
        X_train = self.X_train
        y_train = self.y_train

        for layer in nn.layers:
            layer.b = util.uniform(layer.b.shape, 0.5)

        nn.train_batch(X_train, y_train, lr=None, update=False)
        layer = nn.layers[np.random.randint(len(nn.layers))]
        a = np.random.randint(layer.W.shape[0])
        b = np.random.randint(layer.W.shape[1])
        grad = layer.W_grad[a][b]

        layer.W[a][b] += EPSILON
        nn.train_batch(X_train, y_train, lr=None, update=False)
        loss_plus = nn.loss(X_train, y_train)

        layer.W[a][b] -= 2 * EPSILON
        nn.train_batch(X_train, y_train, lr=None, update=False)
        loss_minus = nn.loss(X_train, y_train)

        real_grad = (loss_plus - loss_minus) / (2 * EPSILON)

        np.testing.assert_approx_equal(grad, real_grad, significant=4)

def suite(activation, loss_type):
   suite = unittest.TestSuite()
   for _ in xrange(100):
       suite.addTest(GradientTestCase(MLP, activation, loss_type, 'testGrad'))
   return suite

if __name__ == "__main__":
     runner = unittest.TextTestRunner()
     runner.run(suite('tanh', 'mse'))
     runner.run(suite('sigmoid', 'mse'))
     runner.run(suite('tanh', 'softmax'))
     runner.run(suite('sigmoid', 'softmax'))

