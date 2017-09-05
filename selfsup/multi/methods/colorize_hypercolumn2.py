import tensorflow as tf
import selfsup
import os
from .base import Method
from autocolorize.tensorflow.sparse_extractor import sparse_extractor
from autocolorize.extraction import calc_rgb_from_hue_chroma
import functools
from collections import OrderedDict



class ColorizeHypercolumn2(Method):
    def __init__(self, name, basenet, loader, use_scalers=True):
        self.name = name
        self._loader = loader
        self.basenet = basenet
        self.locations = 128
        self.edge_buffer = 32
        self.bins = 32
        self._use_scalers = use_scalers

    @property
    def basenet_settings(self):
        return {'convolutional': True}

    def batch(self):
        #input_size = self.basenet.canonical_input_size
        x, _ = self._loader.batch()

        input_size = x.get_shape().as_list()[1]

        self.centroids = tf.to_float(tf.random_uniform(
                shape=[self._loader.batch_size, self.locations, 2],
                minval=self.edge_buffer//4,
                maxval=(input_size - self.edge_buffer)//4,
                name='centroids',
                dtype=tf.int32))

        gray_single = tf.reduce_mean(x, 3, keep_dims=True)
        gray = tf.tile(gray_single, [1, 1, 1, 3])
        self.grayscale = gray_single

        w = input_size // 4
        x4 = tf.image.resize_bilinear(x, (w, w))

        hsv = tf.image.rgb_to_hsv(x4)

        self.hue = tf.expand_dims(hsv[..., 0], -1)
        self.chr = tf.reduce_max(x4, 3, keep_dims=True) - tf.reduce_min(x4, 3, keep_dims=True)

        #self.chr = tf.reduce_mean(x, 3, keep_dims=True)

        self.huechr = tf.concat([self.hue, self.chr], 3)

        return gray, dict(grayscale=gray_single, hsv=hsv), #, imgshape, imgname

    def build_network(self, basenet_info, extra, phase_test, global_step):
        info = selfsup.info.create(scale_summary=True)

        hyper = []

        dropout = functools.partial(selfsup.ops.dropout, phase_test=phase_test, info=info)

        with tf.name_scope('hypercolumn'):
            for name, scale in self.basenet.hypercolumn_layers:
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

        z_hue = selfsup.ops.inner(flat_h, self.bins, activation=None, info=info, name='hue')
        z_chr = selfsup.ops.inner(flat_h, self.bins, activation=None, info=info, name='chroma')

        with tf.name_scope('sparse_y'):
            y_huechr = sparse_extractor(self.centroids, self.huechr, 4.0, [0.0, 0.0])
            L = sparse_extractor(self.centroids, self.grayscale, 1.0, [0.0, 0.0])
            y_hue, y_chr = tf.unstack(tf.to_int32(y_huechr * self.bins * 0.99999), axis=1)

        with tf.variable_scope('primary_loss'):
            chr_weights = tf.minimum(1.0, 3/2 * (1 - tf.abs(2 * L - 1)))
            loss_chr_each = tf.multiply(chr_weights, tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_chr, logits=z_chr))
            loss_hue_each = tf.multiply(5 * y_huechr[..., 1], tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_hue, logits=z_hue), name='foo')
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

