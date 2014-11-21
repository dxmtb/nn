import numpy as np
import logging

import gflags
import datetime

gflags.DEFINE_integer('batch_report_sec', 60, 'report batch loss every n seconds')
gflags.DEFINE_string('dump_prefix', 'theta', 'prefix of dump filename')

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
            raise NotImplementedError('Unknown activation function: ' +
                                      activation)

        if loss_type == 'mse':
            self.error_last_layer = lambda std_outputs, outputs: \
                np.multiply(outputs - std_outputs,
                            self.grad_activation(outputs))
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
            epoch_loss = 0.0
            end_time = datetime.datetime.now()
            for batch in xrange(batch_n):
                start = batch * batch_size
                end = min((batch + 1) * batch_size, len(X_train))
                batch_loss = self.train_batch(X_train[start: end], y_train[start: end], lr)
                epoch_loss += batch_loss

                if datetime.datetime.now() > end_time:
                    logging.info('Epoch %d Batch %d Avg loss %lf' % 
                                 (epoch, batch, epoch_loss/end))
                    end_time = datetime.datetime.now() + \
                        datetime.timedelta(0, gflags.FLAGS.batch_report_sec)

            logging.info('Epoch %d Loss %lf' % (epoch, epoch_loss/N))

    def test_fit(self, X_train, y_train, n_epochs, batch_size, lr):
        logging.info('start test fitting')
        for epoch in xrange(n_epochs):
            start = 0
            end = min(batch_size, len(X_train))
            batch_loss = self.train_batch(X_train[start: end], y_train[start: end], lr)
            logging.info('Epoch %d Loss %lf' % (epoch, batch_loss))

    def train_batch(self, X_train, y_train, lr, update=True):
        outputs = self.forward(X_train)
        self.backward(y_train, outputs)

        if update:
            for layer in self.layers:
                layer.do_update(lr)

        return self.loss_func(y_train, outputs[-1])

    def forward(self, X_train):
        outputs = [X_train]
        for layer in self.layers:
            output = layer.activate(outputs[-1])
            outputs.append(output)
        return outputs

    def backward(self, y_train, outputs):
        error = self.error_last_layer(y_train, outputs[-1])
        for i in reversed(xrange(len(self.layers))):
            # outputs[i] is input to layer i
            layer = self.layers[i]
            layer.grad(error, outputs[i])
            if i != 0:
                error = layer.error(error, outputs[i])

    def output(self, X):
        output = X
        for layer in self.layers:
            output = layer.activate(output)
        return output

    def loss(self, X_test, y_test):
        outputs = self.output(X_test)
        return self.loss_func(y_test, outputs)

    def test(self, X_test, y_test):
        logging.info('Start testing: len %d' % (len(X_test)))
        outputs = self.output(X_test)
        y_true = np.argmax(y_test, axis=1)
        y_pred = np.argmax(outputs, axis=1)
        from sklearn.metrics import confusion_matrix, accuracy_score
        return accuracy_score(y_true, y_pred), confusion_matrix(y_true, y_pred)

    def sample_parameter(self):
        import random
        layer = random.choice(self.layers)
        while len(layer.params) == 0:
            layer = random.choice(self.layers)

        param_name = random.choice(layer.params)

        param = getattr(layer, param_name)
        param_grad = getattr(layer, param_name + '_grad')
        
        ind = []
        for s in param.shape:
            ind.append(np.random.randint(s))

        return param, param_grad, tuple(ind)

    def dump(self, fname):
        theta = {}
        for layer_ind, layer in enumerate(self.layers):
            for param_name in layer.params:
                param = getattr(layer, param_name)
                theta['%d_%s' % (layer_ind, param_name)] = param
        np.savez(open(fname, 'wb'), **theta)

    def load(self, fname):
        theta = np.load(fname)
        for layer_ind, layer in enumerate(self.layers):
            for param_name in layer.params:
                param = theta['%d_%s' % (layer_ind, param_name)]
                old_param = getattr(layer, param_name)
                assert param.shape == old_param.shape
                setattr(layer, param_name, param)
