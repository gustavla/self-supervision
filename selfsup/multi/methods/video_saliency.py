from __future__ import division, print_function, absolute_import
import selfsup
import tensorflow as tf
import os
from .base import Method
from collections import OrderedDict


class VideoSaliency(Method):
    def __init__(self, name, basenet, loader):
        self.name = name
        self.basenet = basenet
        self._loader = loader
        self._classes = 32

    @property
    def basenet_settings(self):
        return {'convolutional': False}

    def batch(self):
        x, extra = self._loader.batch()
        assert 'saliency' in extra
        return x, extra

    def build_network(self, network, extra, phase_test, global_step):
        info = selfsup.info.create(scale_summary=True)

        z = network['activations']['top']

        logits = self.basenet.decoder(z, channels=self._classes, multiple=4)
        y = tf.image.resize_bilinear(extra['saliency'], logits.get_shape().as_list()[1:3])

        labels = tf.to_int32(tf.floor(y[..., 0] * self._classes * 0.99999))

        with tf.variable_scope('primary_loss'):
            loss_each = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            primary_loss = tf.reduce_mean(loss_each)

        #with tf.name_scope('weight_decay'):
            #wd = 0.0005
            #l2_loss = tf.nn.l2_loss(fc8W)
            #weight_decay = wd * l2_loss

        with tf.name_scope('loss'):
            loss = primary_loss

        variables = info['vars']

        self.losses = OrderedDict([('main', primary_loss)])
        self.primary_loss = primary_loss
        self.loss = loss
        self.feedback_variables = []

        info['activations']['primary_loss'] = primary_loss
        info['activations']['loss'] = loss
        #info['activations']['weight_decay'] = weight_decay
        return info

    def feedback(self, variables, iteration):
        pass
