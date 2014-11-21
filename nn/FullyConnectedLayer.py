import numpy as np
import util


class FullyConnectedLayer(object):
    def __init__(self, in_dim, out_dim, activation, grad_activation):
        """
        output = f(x .* W + b)
        x: (n_input, in_dim)
        W: (in_dim, out_dim)
        b: (out_dim)
        output: (n_input, out_dim)
        """
        bound = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = util.uniform((in_dim, out_dim), bound)
        self.b = util.zeros((out_dim,))
        self.activation = activation
        self.grad_activation = grad_activation
        
        self.params = ['W', 'b']

    def activate(self, input):
        return self.activation(np.dot(input, self.W) + self.b)

    def error(self, error_output, input):
        """
        error_output: (n_input, out_dim)
        W_T: (out_dim, in_dim)
        input: (n_input, in_dim)
        """
        return np.multiply(np.dot(error_output, np.transpose(self.W)),
                           self.grad_activation(input))

    def grad(self, error_output, input):
        """
        error_output: (n_input, out_dim)
        input: (n_input, in_dim)
        W_grad: (in_dim, out_dim)
        """
        self.W_grad = np.dot(np.transpose(input), error_output)
        self.b_grad = np.sum(error_output, axis=0)

    def do_update(self, learning_rate):
        self.W -= learning_rate * self.W_grad
        self.b -= learning_rate * self.b_grad
