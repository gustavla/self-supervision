import tensorflow as tf
import selfsup
import os
from .base import Method
from autocolorize.tensorflow.sparse_extractor import sparse_extractor
from autocolorize.extraction import calc_rgb_from_hue_chroma
import functools
from collections import OrderedDict
import numpy as np

def kl_divergence(labels, logits):
    #return tf.contrib.distributions.kl(p, q)
    return (tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits) -
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=tf.log(labels+1e-5)))


class ColorizeHypercolumn3(Method):
    def __init__(self, name, basenet, loader, use_scalers=False, version=1):
        self.name = name
        self._loader = loader
        self.basenet = basenet
        self.locations = 128
        self.edge_buffer = 32
        self._bins = 8
        self._use_scalers = use_scalers
        self._version = version
        self._hist_win_radius = 3
        self._hist_win_size = self._hist_win_radius * 2 + 1

    @property
    def basenet_settings(self):
        return {'convolutional': True}

    def batch(self):
        #input_size = self.basenet.canonical_input_size
        x, _ = self._loader.batch()

        input_size = x.get_shape().as_list()[1]

        self.mult = 1
        self.centroids = tf.to_float(tf.random_uniform(
                shape=[self._loader.batch_size, self.locations, 2],
                minval=self.edge_buffer // self.mult,
                maxval=(input_size - self.edge_buffer) // self.mult,
                name='centroids',
                dtype=tf.int32))

        gray_single = tf.reduce_mean(x, 3, keep_dims=True)
        gray = tf.tile(gray_single, [1, 1, 1, 3])

        w = input_size // self.mult
        if self.mult > 1:
            x4 = tf.image.resize_bilinear(x, (w, w))
        else:
            x4 = x

        hsv = tf.image.rgb_to_hsv(x4)

        self.hue = tf.expand_dims(hsv[..., 0], -1)
        self.chr = tf.reduce_max(x4, 3, keep_dims=True) - tf.reduce_min(x4, 3, keep_dims=True)

        #self.chr = tf.reduce_mean(x, 3, keep_dims=True)

        self.huechr = tf.concat([self.hue, self.chr], 3)

        if self._version == 4:
            self.huechr_onehot = tf.one_hot(tf.to_int32(0.9999 * self._bins * self.huechr), self._bins)
            self.hue_onehot, self.chr_onehot = tf.unstack(self.huechr_onehot, axis=3)

        elif self._version == 3:
            self.huechr_onehot = tf.one_hot(tf.to_int32(0.9999 * self._bins * self.huechr), self._bins)
            self.hue_onehot, self.chr_onehot = tf.unstack(self.huechr_onehot, axis=3)

            #self.hue_onehot = tf.zeros([8, 227, 227, 32], dtype=tf.float32)
            #self.chr_onehot = tf.zeros([8, 227, 227, 32], dtype=tf.float32)

            self.hue_integral = tf.cumsum(tf.cumsum(self.hue_onehot, 1), 2)
            self.chr_integral = tf.cumsum(tf.cumsum(self.chr_onehot, 1), 2)

            #import ipdb; ipdb.set_trace()

            #self.hue_integral = self.hue_onehot
            #self.chr_integral = self.chr_onehot

            #self.hue_integral = tf.zeros([8, 227, 227, 32], dtype=tf.float32)
            #self.chr_integral = tf.zeros([8, 227, 227, 32], dtype=tf.float32)

            #import ipdb; ipdb.set_trace()

            """
            Wa = tf.ones([self._hist_win_size, self._hist_win_size, self._bins, 1]) / self._hist_win_size**2
            Wb = tf.expand_dims(tf.expand_dims(tf.eye(self._bins), 0), 0)
            self.hue_bin = tf.nn.separable_conv2d(self.hue_onehot,
                    Wa,
                    Wb,
                    strides=[1, 1, 1, 1],
                    padding='SAME')


            #self.chr_bin = tf.nn.conv2d(self.chr_onehot, W, strides=[1, 1, 1, 1], padding='SAME')
            self.chr_bin = tf.nn.separable_conv2d(self.chr_onehot,
                    Wa,
                    Wb,
                    strides=[1, 1, 1, 1],
                    padding='SAME')
            #self.hue_bin = self.hue_onehot
            #self.chr_bin = self.chr_onehot

            self.huechr_bin = tf.concat([
                tf.expand_dims(self.hue_bin, 3),
                tf.expand_dims(self.chr_bin, 3)
                ], 3)

            sh = self.huechr_bin.get_shape().as_list()[:3] + [-1]
            self.huechr_bin = tf.reshape(self.huechr_bin, sh)

            #self.hue = tf.nn.conv2d(self.hue_onehot, W0, strides=[1, 1, 1, 1], padding='SAME')
            #self.hue = tf.nn.conv2d(self.hue, W1, strides=[1, 1, 1, 1], padding='SAME')
            #self.chr = tf.nn.conv2d(self.chr_onehot, W0, strides=[1, 1, 1, 1], padding='SAME')
            #self.chr = tf.nn.conv2d(self.chr, W1, strides=[1, 1, 1, 1], padding='SAME')
            """

        return gray, dict(grayscale=gray_single, hsv=hsv), #, imgshape, imgname

    def build_network(self, basenet_info, extra, phase_test, global_step):
        info = selfsup.info.create(scale_summary=True)

        hyper = []

        dropout = functools.partial(selfsup.ops.dropout, phase_test=phase_test, info=info)

        with tf.name_scope('hypercolumn'):
            layers = self.basenet.hypercolumn_layers
            if self._version == 2:
                layers = layers[4:]
            for name, scale in layers:
                sparse_layer = sparse_extractor(self.centroids,
                                                basenet_info['activations'][name],
                                                scale, [0.0, 0.0])
                if self._use_scalers:
                    sparse_layer = selfsup.ops.scale(sparse_layer, name=name+'/scale')
                hyper.append(sparse_layer)

            flat_x = tf.concat(hyper, 1, name='concat')
            info['activations']['hypercolumn'] = flat_x

        flat_h = selfsup.ops.inner(flat_x, 1024, stddev=0.0001, info=info, name='pre_h_fc1')
        flat_h = dropout(flat_h, 0.5, name='h_fc1')

        z_hue = selfsup.ops.inner(flat_h, self._bins, activation=None, info=info, name='hue')
        z_chr = selfsup.ops.inner(flat_h, self._bins, activation=None, info=info, name='chroma')

        with tf.name_scope('sparse_y'):
            if self._version == 3:
                w = self._hist_win_radius

                cc = tf.concat([
                    self.centroids + [w+1, w+1],
                    self.centroids + [w+1, -w],
                    self.centroids + [-w, w+1],
                    self.centroids + [-w, -w],
                ], 1)

                yy = sparse_extractor(cc, self.hue_integral, 1.0, [0.0, 0.0])
                #yy = tf.zeros([4096, 32], dtype=tf.float32)
                yy = tf.reshape(yy, [self.hue_integral.shape[0].value, self.centroids.shape[1].value*4, self._bins])
                y11, y10, y01, y00 = tf.split(yy, 4, axis=1)

                y_hue = tf.reshape((y11 - y10 - y01 + y00) / self._hist_win_size**2, [-1, self._bins])

                yy = sparse_extractor(cc, self.chr_integral, 1.0, [0.0, 0.0])
                #yy = tf.zeros([4096, 32], dtype=tf.float32)
                yy = tf.reshape(yy, [self.chr_integral.shape[0].value, self.centroids.shape[1].value*4, self._bins])
                y11, y10, y01, y00 = tf.split(yy, 4, axis=1)

                y_chr = tf.reshape((y11 - y10 - y01 + y00) / self._hist_win_size**2, [-1, self._bins])
            else:
                #y_huechr = sparse_extractor(self.centroids, self.huechr, self.mult, [0.0, 0.0])
                y_huechr = sparse_extractor(self.centroids, self.huechr, self.mult, [0.0, 0.0])
                y_hue, y_chr = tf.unstack(tf.to_int32(y_huechr * self._bins * 0.99999), axis=1)

        with tf.variable_scope('primary_loss'):
            if self._version == 3:
                loss_chr_each = kl_divergence(labels=y_chr, logits=z_chr)
                loss_hue_each = kl_divergence(labels=y_hue, logits=z_hue)
            #y_chr = sparse_extractor(
            else:
                loss_chr_each = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_chr, logits=z_chr)
                loss_hue_each = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_hue, logits=z_hue)
            #loss_hue_each = tf.multiply(5 * y_huechr[..., 1], loss_hue_each)
            hue_loss = tf.reduce_mean(loss_hue_each)
            chr_loss = tf.reduce_mean(loss_chr_each)
            primary_loss = hue_loss + chr_loss

        l2_layers = [
            info['weights']['pre_h_fc1:weights'],
            info['weights']['hue:weights'],
            info['weights']['chroma:weights'],
        ]
        with tf.name_scope('weight_decay'):
            wd = 1e-6
            l2_loss = tf.add_n([
                tf.nn.l2_loss(v) for v in l2_layers
            ])
            weight_decay = wd * l2_loss

        with tf.name_scope('loss'):
            loss = weight_decay + primary_loss


        self._losses = OrderedDict([
            ('hue', hue_loss),
            ('chr', chr_loss),
            ('+weight_decay', weight_decay),
        ])
        self._primary_loss = primary_loss
        self._loss = loss

        variables = info['vars']

        info['activations']['y_hue'] = y_hue
        info['activations']['y_chr'] = y_chr
        info['activations']['z_hue'] = z_hue
        info['activations']['z_chr'] = z_chr

        self.feedback_variables = [
            #self.hue_integral,
            #self.hue,
            #self.centroids,
            #y_hue,
            #extra['grayscale'],
            #tf.nn.softmax(z_hue),
            #tf.nn.softmax(z_chr),
        ]

        info['activations']['hue'] = self.huechr[..., 0]
        info['activations']['chr'] = self.huechr[..., 1]
        info['activations']['centroids'] = self.centroids
        info['activations']['primary_loss'] = primary_loss
        info['activations']['loss'] = loss
        info['activations']['weight_decay'] = weight_decay
        return info

    def feedback(self, variables, iteration):
        """
        rgb = calc_rgb_from_hue_chroma(variables[0], variables[1], variables[2])

        vzlog.ColorImageGrid([
            rgb[:3]
        ], rows=1, vmin=0, vmax=1).save(self.make_path('hue', 'png'))
        """

    @property
    def losses(self):
        return self._losses

    @property
    def primary_loss(self):
        return self._primary_loss

    @property
    def loss(self):
        return self._loss

