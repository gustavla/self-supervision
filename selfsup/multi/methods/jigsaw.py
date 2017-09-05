from __future__ import division, print_function, absolute_import
import selfsup
import tensorflow as tf
import os
from .base import Method
from collections import OrderedDict
import deepdish as dd
import numpy as np
import itertools
import selfsup.jigsaw


PERMUTATIONS = selfsup.jigsaw.load_permutations(selfsup.res('jigsaw/permutations_100_max.bin'))


def _make_random_patches(x, y, patch_size, permutations, size=3):
    batch_size = x.get_shape().as_list()[0]
    crop_size = x.get_shape().as_list()[1]

    perm_idx = tf.expand_dims(y, 1)
    perm = tf.gather_nd(permutations, perm_idx)


    WINDOW_SIZE = crop_size // size

    N = x.get_shape().as_list()[0]
    C = x.get_shape().as_list()[3]

    patches = []

    for i, j in dd.multi_range(size, size):
        #tf.slice(x, [

        M = WINDOW_SIZE - patch_size + 1
        assert M > 0, f'Jigsaw: Window size ({WINDOW_SIZE}) and patch size ({patch_size}) not compatible'
        limit = np.array([1, M, M, 1])
        offset = np.array([0, i * WINDOW_SIZE, j * WINDOW_SIZE, 0]) + tf.random_uniform(
                [4], dtype=tf.int32,
                maxval=M,
                ) % limit
        patch = tf.slice(x, offset, [N, patch_size, patch_size, C])
        patches.append(patch)

    patches1 = tf.stack(patches, axis=1)
    xyz = np.arange(batch_size)[:, np.newaxis] * size**2 + (perm - 1)
    #import ipdb
    #ipdb.set_trace()
    perm0 = tf.reshape(xyz, [-1])

    patches_flat = tf.reshape(patches1, [-1] + patches1.get_shape().as_list()[2:])
    #import ipdb
    ##ipdb.set_trace()
    patches2 = tf.gather(patches_flat, perm0)

    #return tf.reshape(patches2, [-1, PATCH_SIZE, PATCH_SIZE, C])
    return patches2


class Jigsaw(Method):
    def __init__(self, name, basenet, loader, patch_size=75, size=3,
                 reduce_channels=128, use_scalers=False):
        self.name = name
        self.basenet = basenet
        self._size = size
        self._patch_size = patch_size
        self._loader = loader
        self._reduce_channels = reduce_channels
        if size == 3:
            self._permutations = PERMUTATIONS
        elif size == 2:
            # They are 1-based due to the permutations file
            self._permutations = 1 + np.array(list(itertools.permutations(range(size**2))))
        self._use_scalers = use_scalers

    @property
    def basenet_settings(self):
        return {'convolutional': False}

    def batch(self):
        x, _ = self._loader.batch()
        y = tf.random_uniform([self._loader.batch_size], dtype=tf.int32, maxval=len(self._permutations))
        patches = _make_random_patches(x, y, self._patch_size, self._permutations, size=self._size)

        pad_both = self.basenet.canonical_input_size - self._patch_size
        pad_lo = pad_both // 2
        pad_up = pad_both - pad_lo

        #paddings = [[0, 0], [pad_lo, pad_up], [pad_lo, pad_up], [0, 0]]
        #pad_patches = tf.pad(patches, paddings=paddings, mode='REFLECT')
        pad_patches = patches

        self._y = y
        extra = {'permutation': y}
        return pad_patches, extra

    def build_network(self, network, extra, phase_test, global_step):
        info = selfsup.info.create(scale_summary=True)

        if self._size == 3:
            z = network['activations']['pool5']
        else:
            z = network['activations']['top']

        #z = tf.squeeze(z, [1, 2])
        z = tf.reshape(z, (z.get_shape().as_list()[0], -1))

        if self._use_scalers:
            z = selfsup.ops.scale(z, name='scale')

        #W_init = tf.contrib.layers.variance_scaling_initializer()
        W_init = tf.random_normal_initializer(0.0, 0.0001)
        b_init = tf.constant_initializer(0.0)

        reduce_ch = self._reduce_channels

        with tf.variable_scope('reduction'):
            c_o = reduce_ch
            reduce_W = tf.get_variable('weights', [z.get_shape().as_list()[1], c_o], dtype=tf.float32,
                                initializer=W_init)
            reduce_b = tf.get_variable('biases', [c_o], dtype=tf.float32,
                                initializer=b_init)
        z = tf.nn.xw_plus_b(z, reduce_W, reduce_b)
        z = tf.nn.relu(z)

        z = tf.reshape(z, [self._loader.batch_size, -1, z.get_shape().as_list()[-1]])
        z = tf.concat(tf.unstack(z, axis=1), 1)

        with tf.variable_scope('jigsaw'):
            c_o = len(self._permutations)
            jigsaw_W = tf.get_variable('weights', [z.get_shape().as_list()[1], c_o], dtype=tf.float32,
                                initializer=W_init)
            jigsaw_b = tf.get_variable('biases', [c_o], dtype=tf.float32,
                                initializer=b_init)
        z = tf.nn.xw_plus_b(z, jigsaw_W, jigsaw_b)

        with tf.variable_scope('primary_loss'):
            loss_each = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._y, logits=z)
            primary_loss = tf.reduce_mean(loss_each)

        with tf.name_scope('weight_decay'):
            wd = 1e-6
            l2_loss = tf.nn.l2_loss(reduce_W) + tf.nn.l2_loss(jigsaw_W)
            weight_decay = wd * l2_loss

        with tf.name_scope('loss'):
            loss = weight_decay + primary_loss

        variables = info['vars']

        self.losses = OrderedDict([
            ('main', primary_loss),
            ('+weight_decay', weight_decay),
        ])
        self.primary_loss = primary_loss
        self.loss = loss
        self.feedback_variables = []

        info['activations']['primary_loss'] = primary_loss
        info['activations']['loss'] = loss
        info['activations']['weight_decay'] = weight_decay
        return info

    def feedback(self, variables, iteration):
        pass
