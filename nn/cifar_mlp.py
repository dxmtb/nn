from MLP import MLP
from util import load_CIFAR_batches

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

import gflags
FLAGS = gflags.FLAGS
# model
gflags.DEFINE_integer('hidden_dim', 500, 'hidden dim')
gflags.DEFINE_string('loss_type', 'mse', 'final loss type(mse or softmax)')
gflags.DEFINE_integer('layer_num', 0, 'additional layer number')
gflags.DEFINE_string('activation', 'tanh', 'activation function')
# train
gflags.DEFINE_float('lr', 0.0001, 'learning rate')
gflags.DEFINE_integer('epoch', 500, 'Epoch number')
gflags.DEFINE_integer('batch', 100, 'batch size')
# data
gflags.DEFINE_string('datapath', '../database/', 'path to CIFAR-10 data')


def load_CIFAR_train(path):
    return load_CIFAR_batches([path+'data_batch_%d' % (i+1) for i in xrange(5)])


def load_CIFAR_test(path):
    return load_CIFAR_batches([path+'test_batch'])


def main(argv):
    argv = FLAGS(argv)
    inputs, outputs = load_CIFAR_train(FLAGS.datapath)
    nn = MLP(3072, FLAGS.hidden_dim, 10, FLAGS.activation, FLAGS.loss_type, FLAGS.layer_num)
    nn.fit(inputs, outputs, FLAGS.epoch, FLAGS.batch, FLAGS.lr)
    print nn.test(*load_CIFAR_test(FLAGS.datapath))


if __name__ == '__main__':
    import sys
    main(sys.argv)
