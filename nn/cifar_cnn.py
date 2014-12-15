from CNN import CNN
from util import load_CIFAR_batches

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s[%(lineno)d] %(levelname)s %(message)s', datefmt='%H:%M:%S')

import gflags
FLAGS = gflags.FLAGS
# model
gflags.DEFINE_string('loss_type', 'softmax', 'final loss type(mse or softmax)')
gflags.DEFINE_string('activation', 'relu', 'activation function')
# train
gflags.DEFINE_integer('epoch', 100, 'Epoch number')
gflags.DEFINE_integer('batch', 100, 'batch size')
# data
gflags.DEFINE_string('datapath', '../database/', 'path to CIFAR-10 data')


def load_CIFAR_train(path):
    inputs, outputs = load_CIFAR_batches([path+'data_batch_%d' % (i+1) for i in xrange(5)])
    return to4d(inputs), outputs

def load_CIFAR_test(path):
    inputs, outputs = load_CIFAR_batches([path+'test_batch'])
    return to4d(inputs), outputs

def to4d(data):
    data_size = data.shape[0]
    return data.reshape(data_size, 3, 32, 32)

def main(argv):
    argv = FLAGS(argv)
    inputs, outputs = load_CIFAR_train(FLAGS.datapath)
    X_test, y_test = load_CIFAR_test(FLAGS.datapath)
    nn = CNN(10, FLAGS.activation, FLAGS.loss_type, FLAGS.batch)
    nn.fit(inputs, outputs, FLAGS.epoch, FLAGS.batch, [0.00001, 0.00002],
            X_test, y_test)
    print nn.test(X_test, y_test)


if __name__ == '__main__':
    import sys
    main(sys.argv)
