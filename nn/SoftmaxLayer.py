import numpy as np
from util import softmax
from FullyConnectedLayer import FullyConnectedLayer


class SoftmaxLayer(FullyConnectedLayer):
    def activate(self, input):
        a = np.dot(input, self.W) + self.b
        ret = softmax(a)
        return ret
