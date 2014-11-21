from FullyConnectedLayer import FullyConnectedLayer
from NeuralNetwork import NeuralNetwork


class MLP(NeuralNetwork):
    def __init__(self, in_dim, hidden_dim, out_dim, activation, loss_type,
                 layer_num=0):
        NeuralNetwork.__init__(self, activation, loss_type)

        args = [self.activation, self.grad_activation]
        self.layers = []
        self.layers.append(FullyConnectedLayer(in_dim, hidden_dim, *args))
        for _ in xrange(layer_num):
            self.layers.append(FullyConnectedLayer(hidden_dim, hidden_dim, *args))
        if loss_type == 'mse':
            self.layers.append(FullyConnectedLayer(hidden_dim, out_dim, *args))
        else:
            from SoftmaxLayer import SoftmaxLayer
            self.layers.append(SoftmaxLayer(hidden_dim, out_dim, *args))
