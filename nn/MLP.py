import numpy as np
from HiddenLayer import HiddenLayer
import logging

class MLP(object):
    def __init__(self, in_dim, hidden_dim, out_dim, activation, layer_num=0):
        if activation == 'tanh':
            self.activation = np.tanh
            # f(x)=tanh(x) f'(x)=1-(f(x))^2
            self.grad_activation = lambda x: 1. - x**2
        elif activation == 'sigmoid':
            from scipy.special import expit
            self.activation = expit
            self.grad_activation = lambda x: x*(1.-x)
        else:
            raise NotImplementedError('Unknown activation function: ' + activation)

        self.layers=[HiddenLayer(in_dim, hidden_dim, self.activation, self.grad_activation)]
        for _ in xrange(layer_num):
            self.layers.append(HiddenLayer(hidden_dim, hidden_dim, self.activation, self.grad_activation))
        self.layers.append(HiddenLayer(hidden_dim, out_dim, self.activation, self.grad_activation))

    def fit(self, X_train, y_train, n_epochs, batch_size, lr):
        logging.info('start fitting')
        N = len(X_train)
        batch_n = N / batch_size + 1
        if (batch_n-1) * batch_size >= N:
            batch_n = batch_n - 1

        for epoch in xrange(n_epochs):
            for batch in xrange(batch_n):
                start = batch * batch_size
                end = min((batch + 1) * batch_size, len(X_train))
                self.train_batch(X_train[start: end], y_train[start: end], lr)
            logging.info('Epoch %d Loss %lf' % (epoch, self.loss(X_train, y_train)))

    def test_fit(self, X_train, y_train, n_epochs, batch_size, lr):
        logging.info('start test fitting')
        N = len(X_train)
        batch_n = N / batch_size + 1
        if (batch_n-1) * batch_size >= N:
            batch_n = batch_n - 1

        for epoch in xrange(n_epochs):
            start = 0
            end = min(batch_size, len(X_train))
            self.train_batch(X_train[start: end], y_train[start: end], lr)
            logging.info('Epoch %d Loss %lf' % (epoch, self.loss(X_train, y_train)))

    def train_batch(self, X_train, y_train, lr):
        outputs = [X_train]
        for layer in self.layers:
            output = layer.activate(outputs[-1])
            outputs.append(output)

        # y_dim = len(y[0])
        # first (n_input, y_dim)
        # second (n_input, y_dim)
        # error (n_input, y_dim)) by element-wise multiply
        error = -np.multiply(y_train - outputs[-1], self.grad_activation(outputs[-1]))
        for i in reversed(xrange(len(self.layers))):
            layer = self.layers[i]
            layer.grad(error, outputs[i])
            if i != 0:
                error = layer.error(error, outputs[i])

        for layer in self.layers:
            layer.do_update(lr)

    def output(self, X):
        output = X
        for layer in self.layers:
            output = layer.activate(output)
        return output

    def loss(self, X_test, y_test):
        outputs = self.output(X_test)
        return np.sum((outputs-y_test)**2)

    def test(self, X_test, y_test):
        outputs = self.output(X_test)
        y_true = np.argmax(y_test, axis=1)
        y_pred = np.argmax(outputs, axis=1)
        from sklearn.metrics import confusion_matrix, accuracy_score
        return accuracy_score(y_true, y_pred), confusion_matrix(y_true, y_pred)
