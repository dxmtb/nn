from NeuralNetwork import NeuralNetwork
from FullyConnectedLayer import FullyConnectedLayer
from ConvPoolLayer import ConvPoolLayer
from FlattenLayer import FlattenLayer
import numpy as np
import logging


class CNN(NeuralNetwork):
    def __init__(self, out_dim, activation, loss_type, poolsize=(2, 2)):
        NeuralNetwork.__init__(self, activation, loss_type)

        args = [self.activation, self.grad_activation]

        image_shape = (32, 32)
        filter_num = 64

        self.layers = []
        self.layers.append(ConvPoolLayer(image_shape,
                                         (filter_num, 3, 5, 5),
                                         (2, 2), None, *args))
        self.layers.append(ConvPoolLayer(self.layers[-1].output_shape,
                                         (filter_num, filter_num, 5, 5),
                                         (2, 2), None, *args))

        # (32-5+1)/2 = 14
        # (14-5+1)/2 = 5
        dim = np.prod(self.layers[-1].output_shape) * filter_num

        # logging.info('Output dim to FC %d' % dim)

        self.layers.append(FlattenLayer())

        if loss_type == 'mse':
            self.layers.append(FullyConnectedLayer(dim, out_dim, *args))
        else:
            from SoftmaxLayer import SoftmaxLayer
            self.layers.append(SoftmaxLayer(dim, out_dim, *args))

