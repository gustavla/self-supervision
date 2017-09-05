from __future__ import division, print_function, absolute_import
import selfsup
import tensorflow as tf
import os
from .base import Method
from collections import OrderedDict
import deepdish as dd
import numpy as np

# http://stackoverflow.com/questions/36904298/how-to-implement-multi-class-hinge-loss-in-tensorflow/36928137#36928137
def multi_class_hinge_loss(logits, labels, n_classes):
    batch_size = logits.get_shape().as_list()[0]
    # get the correct logit
    flat_logits = tf.reshape(logits, (-1,))
    correct_id = tf.range(0, batch_size) * n_classes + labels
    correct_logit = tf.gather(flat_logits, correct_id)

    # get the wrong maximum logit
    max_label = tf.to_int32(tf.argmax(logits, 1))
    top2, _ = tf.nn.top_k(logits, k=2, sorted=True)
    first, second = tf.unstack(top2, axis=1)
    wrong_max_logit = tf.where(tf.equal(max_label, labels), second, first)

    # calculate multi-class hinge loss
    return tf.maximum(0., 1.0 + wrong_max_logit - correct_logit), wrong_max_logit, correct_logit


class Supervised(Method):
    def __init__(self, name, basenet, loader, loss_mult=1.0, loss='kl', weight_decay=1e-6,
            segment=0, num_segments=1, apply_fix=True):
        self.name = name
        self.basenet = basenet
        self._loader = loader
        self._loss_mult = loss_mult
        self._loss = loss
        self._weight_decay = weight_decay
        self._segment = segment
        self._num_segments = num_segments
        self._apply_fix = apply_fix

    @property
    def basenet_settings(self):
        return {'convolutional': False}

    def batch(self):
        x, extra = self._loader.batch()
        assert 'y' in extra and 'num_classes' in extra
        y = extra['y']
        self._labels = extra['y']
        self._num_classes = extra['num_classes']

        return x, extra

    def build_network(self, network, extra, phase_test, global_step):
        info = selfsup.info.create(scale_summary=True)

        z = network['activations']['top']

        z = tf.reshape(z, [z.get_shape().as_list()[0], -1])

        if self._num_segments > 1:
            C = z.get_shape().as_list()[-1] // self._num_segments
            z = z[:, self._segment*C:(self._segment+1)*C]

        W_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.constant_initializer(0.0)


        if self._apply_fix:
            z = selfsup.ops.dropout(z, 0.5, phase_test=phase_test)

        with tf.variable_scope('reduction'):
            c_o = self._num_classes
            task_W = tf.get_variable('weights', [z.get_shape().as_list()[1], c_o], dtype=tf.float32,
                                initializer=W_init)
            task_b = tf.get_variable('biases', [c_o], dtype=tf.float32,
                                initializer=b_init)
            z = tf.nn.xw_plus_b(z, task_W, task_b)

        if not self._apply_fix:
            z = selfsup.ops.dropout(z, 0.5, phase_test=phase_test)

        self.logits = z
        self.labels = self._labels

        with tf.variable_scope('primary_loss'):
            if self._loss == 'kl':
                loss_each = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._labels, logits=z)
            elif self._loss == 'l2':
                y_onehot = tf.one_hot(self._labels, self._num_classes)
                loss_each = (y_onehot - z)**2
            elif self._loss == 'l1':
                y_onehot = tf.one_hot(self._labels, self._num_classes)
                loss_each = tf.abs(y_onehot - z)
            elif self._loss == 'multilabel':
                y_onehot = tf.one_hot(self._labels, self._num_classes)
                loss_each = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_onehot, logits=z)
            elif self._loss == 'multihinge':
                #loss_each = multi_class_hinge_loss(labels=self._labels, logits=z, n_classes=self._num_classes)
                y_onehot = tf.one_hot(self._labels, self._num_classes)
                loss_each = tf.losses.hinge_loss(labels=y_onehot, logits=z)
            elif self._loss == 'multihinge2':
                #loss_each = multi_class_hinge_loss(labels=self._labels, logits=z, n_classes=self._num_classes)
                y_onehot = tf.one_hot(self._labels, self._num_classes)
                loss_each = tf.losses.hinge_loss(labels=y_onehot, logits=z)
            elif self._loss == 'hinge':
                #z *= 10
                loss_each, a, b = multi_class_hinge_loss(labels=self._labels, logits=z, n_classes=self._num_classes)
                #y_onehot = tf.one_hot(self._labels, self._num_classes)
                #loss_each = tf.losses.hinge_loss(labels=y_onehot, logits=z)
            elif self._loss == 'l2b':
                y_onehot = tf.one_hot(self._labels, self._num_classes)
                loss_each = (y_onehot - z)**2 * (y_onehot * (self._num_classes - 1) + 1)
            elif self._loss.startswith('kl:'):
                mult = float(self._loss.split(':')[1])
                loss_each = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._labels, logits=z * mult)
            else:
                raise ValueError('Unknown loss')

            primary_loss = tf.reduce_mean(loss_each) * self._loss_mult

        with tf.name_scope('weight_decay'):
            l2_loss = tf.nn.l2_loss(task_W)
            weight_decay = self._weight_decay * l2_loss

        with tf.name_scope('loss'):
            loss = weight_decay + primary_loss

        self.predictions = z

        variables = info['vars']

        self.losses = OrderedDict([
            ('main', primary_loss),
            ('+weight_decay', weight_decay),
        ])
        self.primary_loss = primary_loss
        self.loss = loss
        self.feedback_variables = []#[z, a, b]

        info['activations']['primary_loss'] = primary_loss
        info['activations']['loss'] = loss
        info['activations']['weight_decay'] = weight_decay
        return info

    def feedback(self, variables, iteration):
        pass
        #print(iteration, '-----')
        #for v in variables:
            #print(v)
