import unittest
import numpy as np
from MLP import MLP
from CNN import CNN
import util

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')


class LoadSaveTestCase(unittest.TestCase):
    def __init__(self, methodName='testGrad'):
        super(LoadSaveTestCase, self).__init__(methodName)
        self.activation = 'tanh'
        self.loss_type = 'mse'
        self.EPSILON = 1e-4

    def setUp(self):
        in_dim, hidden_dim, out_dim = np.random.randint(3, 5, size=3)
        self.args = in_dim, hidden_dim, out_dim, self.activation, self.loss_type
        self.X_train = util.uniform((1, in_dim), 1.0)
        if self.loss_type == 'softmax':
            self.y_train = util.zeros((1, out_dim))
            self.y_train[0][np.random.randint(out_dim)] = 1.0
        else:
            self.y_train = util.uniform((1, out_dim), 1.0)

    def runTest(self):
        nn = MLP(*self.args)
        for layer in nn.layers:
            layer.b = util.uniform(layer.b.shape, 0.5)
        X_train = self.X_train
        y_train = self.y_train

        loss_before = nn.loss(X_train, y_train)
        nn.dump('test.npz')

        nn = MLP(*self.args)
        nn.load('test.npz')
        loss_after = nn.loss(X_train, y_train)

        np.testing.assert_almost_equal(loss_after, loss_before)
        print loss_before, loss_after

if __name__ == "__main__":
    unittest.main()
