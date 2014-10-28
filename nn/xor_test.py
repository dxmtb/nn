import unittest
import numpy as np
from MLP import MLP

class XORTestCase(unittest.TestCase):
    def runTest(self):
        inputs = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)
        outputs = np.array([[0, 1], [0, 1], [1, 0], [1, 0]], dtype=float).reshape(4, 2)
        nn = MLP(2, 5, 2, 'tanh')
        nn.fit(inputs, outputs, 1000, 3, 0.1)
        nn_outputs = nn.output(inputs)
        loss = abs(np.mean(nn_outputs-outputs))
        print 'Outputs:', nn_outputs
        print 'Loss:', loss
        self.assertTrue(loss < 0.1)

if __name__ == "__main__":
    unittest.main()
