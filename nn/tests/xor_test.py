import unittest
import numpy as np
from nn.MLP import MLP
from nn import util

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')

class XORTestCase(unittest.TestCase):
    def runTest(self):
        inputs = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=util.FLOAT())
        outputs = np.array([[0, 1], [0, 1], [1, 0], [1, 0]], dtype=util.FLOAT()).reshape(4, 2)
        nn = MLP(2, 5, 2, 'sigmoid', loss_type='softmax')
        nn.fit(inputs, outputs, 1000, 3, 2)
        nn_outputs = nn.output(inputs)
        loss = abs(np.mean((nn_outputs-outputs)**2))
        print 'Outputs:', nn_outputs
        print 'Loss:', loss
        np.testing.assert_almost_equal(loss, 0, 2)

if __name__ == "__main__":
    unittest.main()
