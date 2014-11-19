import numpy as np
import logging
from util import softmax

class NeuralNetwork(object):
    def __init__(self, activation, loss_type):
        """
        loss_type: mse/softmax
        """
        if activation == 'tanh':
            self.activation = np.tanh
            self.grad_activation = lambda x: 1. - x**2
        elif activation == 'sigmoid':
            from scipy.special import expit
            self.activation = expit
            self.grad_activation = lambda x: x*(1.-x)
        else:
            raise NotImplementedError('Unknown activation function: ' + activation)

        if loss_type == 'mse':
            self.error_last_layer = lambda std_outputs, outputs: \
                    np.multiply(outputs - std_outputs, self.grad_activation(outputs))
            self.loss_func = lambda std_outputs, outputs: \
                    0.5 * np.sum((std_outputs-outputs)**2)
        elif loss_type == 'softmax':
            self.error_last_layer = lambda std_outputs, outputs: \
                    (outputs - std_outputs)
            self.loss_func = lambda std_outputs, outputs: \
                    -(np.sum(np.multiply(std_outputs, np.log(outputs))))
        else:
            raise NotImplementedError('Unknown loss type: ' + loss_type)

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
        for epoch in xrange(n_epochs):
            start = 0
            end = min(batch_size, len(X_train))
            self.train_batch(X_train[start: end], y_train[start: end], lr)
            logging.info('Epoch %d Loss %lf' % (epoch, self.loss(X_train, y_train)))

    def train_batch(self, X_train, y_train, lr, update=True):
        outputs = self.forward(X_train)
        self.backward(y_train, outputs)

        if update:
            for layer in self.layers:
                layer.do_update(lr)

    def forward(self, X_train):
        outputs = [X_train]
        for layer in self.layers:
            output = layer.activate(outputs[-1])
            outputs.append(output)
        return outputs

    def backward(self, y_train, outputs):
        error = self.error_last_layer(y_train, self.filter_output(outputs[-1]))
        for i in reversed(xrange(len(self.layers))):
            # outputs[i] is input to layer i
            layer = self.layers[i]
            layer.grad(error, outputs[i])
            if i != 0:
                error = layer.error(error, outputs[i])

    def _output(self, X):
        output = X
        for layer in self.layers:
            output = layer.activate(output)
        return output

    def loss(self, X_test, y_test):
        outputs = self.output(X_test)
        return self.loss_func(y_test, outputs)

    def output(self, X):
        return self.filter_output(self._output(X))

    def test(self, X_test, y_test):
        outputs = self.output(X_test)
        y_true = np.argmax(y_test, axis=1)
        y_pred = np.argmax(outputs, axis=1)
        from sklearn.metrics import confusion_matrix, accuracy_score
        return accuracy_score(y_true, y_pred), confusion_matrix(y_true, y_pred)

    # overload
    def filter_output(self, output):
        return output
