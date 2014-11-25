import unittest
import numpy as np
from nn.MLP import MLP
from nn import util
from nn.ConvPoolLayer import py_do_pooling, ct_do_pooling

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')

class UpsampleTestCase(unittest.TestCase):
    def runTest(self):
        images = util.uniform((100, 8, 28, 28), 255)
        poolsize = (3, 3)
        M1, ret1 = py_do_pooling(images, poolsize)
        M2, ret2 = ct_do_pooling(images, poolsize)
        np.testing.assert_array_equal(ret1, ret2)
        np.testing.assert_array_equal(M1, M2)
        print ret2, M2

if __name__ == "__main__":
    unittest.main()

