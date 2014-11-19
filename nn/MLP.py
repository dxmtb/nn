import numpy as np
from HiddenLayer import HiddenLayer
from NeuralNetwork import NeuralNetwork
import logging

class MLP(NeuralNetwork):
    def __init__(self, in_dim, hidden_dim, out_dim, activation, loss_type, layer_num=0):
        NeuralNetwork.__init__(self, activation, loss_type)

        self.layers=[HiddenLayer(in_dim, hidden_dim, self.activation, self.grad_activation)]
        for _ in xrange(layer_num):
            self.layers.append(HiddenLayer(hidden_dim, hidden_dim, self.activation, self.grad_activation))
        if loss_type == 'mse':
            self.layers.append(HiddenLayer(hidden_dim, out_dim, self.activation, self.grad_activation))
        else:
            from SoftmaxLayer import SoftmaxLayer
            self.layers.append(SoftmaxLayer(hidden_dim, out_dim, self.activation, self.grad_activation))
