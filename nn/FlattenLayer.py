import numpy as np


class FlattenLayer(object):
    def __init__(self):
        self.params = []

    def activate(self, input):
        out_shape = (input.shape[0], np.prod(input.shape[1:]))
        return input.reshape(out_shape)

    def error(self, error_output, input):
        return error_output.reshape(input.shape)

    def grad(self, error_output, input):
        pass

    def do_update(self, learning_rate):
        pass
