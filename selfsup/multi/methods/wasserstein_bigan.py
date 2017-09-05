# IN PROGRESS AND EXPERIMENTAL
from __future__ import division, print_function, absolute_import
import selfsup
import tensorflow as tf
import os
from .base import Method


if os.uname()[1] == 'kiriyama':
    TRAIN = os.path.expandvars("$IMAGENET_10K_DIR/fullpath_imagenet_tr_nolabels.txt")
else:
    TRAIN = "/share/data/vision-greg/larsson/data/fullpath_imagenet_and_places_train_shuffled.txt"


class WassersteinBiGAN(Method):
    def __init__(self, basenet, batch_size):
        self.basenet = basenet
        self.batch_size = batch_size

    @property
    def basenet_settings(self):
        return {'convolutional': False}

    def batch_loader(self):
        x, imgshape, imgname = selfsup.datasets.unlabeled_batching(
                TRAIN,
                batch_size=self.batch_size,
                input_size=self.basenet.canonical_input_size,
                random_mirror=True)

        return x, x, imgshape, imgname

    def build_network(self, network, y, phase_test):
        info = selfsup.info.create(scale_summary=True)

        z = network['activations']['top']

        sh = z.get_shape().as_list()

        W_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.0)

        with tf.variable_scope('rotation'):
            c_o = 4
            fc8W = tf.get_variable('weights', [sh[1], c_o], dtype=tf.float32,
                                initializer=W_init)
            fc8b = tf.get_variable('biases', [c_o], dtype=tf.float32,
                                initializer=b_init)
        z = tf.nn.xw_plus_b(z, fc8W, fc8b)

        with tf.variable_scope('primary_loss'):
            loss_each = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z)
            primary_loss = tf.reduce_mean(loss_each)

        with tf.name_scope('weight_decay'):
            wd = 0.0005
            l2_loss = tf.nn.l2_loss(fc8W)
            weight_decay = wd * l2_loss

        with tf.name_scope('loss'):
            loss = weight_decay + primary_loss

        variables = info['vars']

        info['activations']['primary_loss'] = primary_loss
        info['activations']['loss'] = loss
        info['activations']['weight_decay'] = weight_decay
        return info
