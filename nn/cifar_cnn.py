from CNN import CNN
from util import load_CIFAR_batches

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

import gflags
FLAGS = gflags.FLAGS
# model
gflags.DEFINE_integer('hidden_dim', 1000, 'hidden dim')
gflags.DEFINE_string('loss_type', 'mse', 'final loss type(mse or softmax)')
gflags.DEFINE_string('activation', 'tanh', 'activation function')
# train
gflags.DEFINE_float('lr', 0.0001, 'learning rate')
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
    nn = CNN(10, FLAGS.activation, FLAGS.loss_type)
    nn.fit(inputs , outputs, FLAGS.epoch, FLAGS.batch, FLAGS.lr)
    print nn.test(*load_CIFAR_test(FLAGS.datapath))


if __name__ == '__main__':
    import sys
    main(sys.argv)
