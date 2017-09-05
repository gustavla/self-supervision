from __future__ import division, print_function, absolute_import
import selfsup
import tensorflow as tf
import os
from .base import Method
from autocolorize.tensorflow.sparse_extractor import sparse_extractor
from autocolorize.extraction import calc_rgb_from_hue_chroma
from collections import OrderedDict
import functools


class VideoSaliencyHypercolumn(Method):
    def __init__(self, name, basenet, loader):
        self.name = name
        self.basenet = basenet
        self._loader = loader
        self._classes = 32
        self.locations = 128
        self.edge_buffer = 32

    @property
    def basenet_settings(self):
        return {'convolutional': True}

    def batch(self):
        x, extra = self._loader.batch()

        input_size = x.get_shape().as_list()[1]

        edge_buffer = int(self.edge_buffer * self._loader.scale_factor)
        self.centroids = tf.to_float(tf.random_uniform(
                shape=[self._loader.batch_size, self.locations, 2],
                minval=edge_buffer,
                maxval=input_size - edge_buffer,
                name='centroids',
                dtype=tf.int32))

        assert 'saliency' in extra
        self.saliency = extra['saliency']
        return x, extra

    def build_network(self, basenet_info, extra, phase_test, global_step):
        info = selfsup.info.create(scale_summary=True)

        hyper = []

        dropout = functools.partial(selfsup.ops.dropout, phase_test=phase_test, info=info)

        with tf.name_scope('hypercolumn'):
            for name, scale in self.basenet.hypercolumn_layers:
                sparse_layer = sparse_extractor(self.centroids,
                                                basenet_info['activations'][name],
                                                scale, [0.0, 0.0])
                hyper.append(sparse_layer)

            flat_x = tf.concat(hyper, 1, name='concat')
            info['activations']['hypercolumn'] = flat_x

        flat_h = selfsup.ops.inner(flat_x, 1028, stddev=0.0001, info=info, name='pre_h_fc1')
        flat_h = dropout(flat_h, 0.5, name='h_fc1')

        z = selfsup.ops.inner(flat_h, self._classes, activation=None, info=info, name='saliency')

        with tf.name_scope('sparse_y'):
            y = sparse_extractor(self.centroids, self.saliency, 1 / self._loader.scale_factor, [0.0, 0.0])
            y = tf.to_int32(tf.floor(y[..., 0] * self._classes * 0.99999))

        with tf.variable_scope('primary_loss'):
            loss_each = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z)
            saliency_loss = tf.reduce_mean(loss_each)
            primary_loss = saliency_loss

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
