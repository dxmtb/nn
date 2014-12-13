import unittest
import numpy as np
from nn.ConvPoolLayer import rot90

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')

class Rot90TestCase(unittest.TestCase):
    def runTest(self):
        A = np.random.uniform(size=(10, 64, 32, 32))
        B = rot90(A)
        for i in xrange(10):
            for j in xrange(64):
                np.testing.assert_array_equal(np.rot90(A[i][j], 2), B[i][j])

if __name__ == "__main__":
    unittest.main()
