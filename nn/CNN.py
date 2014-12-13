from NeuralNetwork import NeuralNetwork
from FullyConnectedLayer import FullyConnectedLayer
from ConvPoolLayer import ConvPoolLayer
from FlattenLayer import FlattenLayer
import numpy as np
import logging


class CNN(NeuralNetwork):
    def __init__(self, out_dim, activation, loss_type, batch_size, poolsize=(2, 2)):
        NeuralNetwork.__init__(self, activation, loss_type)

        args = [self.activation, self.grad_activation]

        image_shape = (batch_size, 3, 32, 32)
        filter_num = 32

        self.layers = []
        bound = [None, None]
        if activation == 'relu':
            bound = [0.0001, 0.0001]

        self.layers.append(ConvPoolLayer(image_shape,
                                         (filter_num, 3, 5, 5),
                                         (3, 3), bound[0], *args))
        self.layers.append(ConvPoolLayer(self.layers[-1].output_shape,
                                         (filter_num, filter_num, 5, 5),
                                         (3, 3), bound[1], *args))

        for layer in self.layers:
            logging.info("Layer output: %s" % (str(layer.output_shape)))
        # (32-5+1)/2 = 14
        # (14-5+1)/2 = 5
        dim = np.prod(self.layers[-1].output_shape[1:])

        logging.info('Output to FC: %d' % (dim))

        # logging.info('Output dim to FC %d' % dim)

        self.layers.append(FlattenLayer())

        if loss_type == 'mse':
            self.layers.append(FullyConnectedLayer(dim, out_dim, *args))
        else:
            from SoftmaxLayer import SoftmaxLayer
            self.layers.append(SoftmaxLayer(dim, out_dim, *args))

