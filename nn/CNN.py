from NeuralNetwork import NeuralNetwork
from FullyConnectedLayer import FullyConnectedLayer
from ConvPoolLayer import ConvPoolLayer
from FlattenLayer import FlattenLayer


class CNN(NeuralNetwork):
    def __init__(self, out_dim, activation, loss_type, poolsize=(2, 2)):
        NeuralNetwork.__init__(self, activation, loss_type)

        args = [self.activation, self.grad_activation]

        self.layers = []
        self.layers.append(ConvPoolLayer((64, 3, 5, 5), poolsize, *args))
        self.layers.append(ConvPoolLayer((64, 64, 5, 5), poolsize, *args))

        # (32-5+1)/2 = 14
        # (14-5+1)/2 = 5
        dim = 5 * 5 * 64

        self.layers.append(FlattenLayer())

        if loss_type == 'mse':
            self.layers.append(FullyConnectedLayer(dim, out_dim, *args))
        else:
            from SoftmaxLayer import SoftmaxLayer
            self.layers.append(SoftmaxLayer(dim, out_dim, *args))

