import numpy as np
from util import softmax
from HiddenLayer import HiddenLayer

class SoftmaxLayer(HiddenLayer):
    def activate(self, input):
        a = np.dot(input, self.W) + self.b
        ret = softmax(a)
        return softmax(a)
