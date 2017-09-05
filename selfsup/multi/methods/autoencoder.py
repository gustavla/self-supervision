import tensorflow as tf
import os
import selfsup
from .base import Method
from collections import OrderedDict


class Autoencoder(Method):
    def __init__(self, name, basenet, loader, embedding_size=1024, method='gustav'):
        self.name = name
        self.basenet = basenet
        self._loader = loader
        self._embedding_size = embedding_size
        self._method = method

    @property
    def basenet_settings(self):
        return {'convolutional': False}

    def batch(self):
        x, extra = self._loader.batch()
        extra['x'] = x
        return x, extra

    def build_network(self, basenet_info, extra, phase_test, global_step):
        x = extra['x']
        batch_size = x.get_shape().as_list()[0]
        info = selfsup.info.create(scale_summary=True)
        info['config']['save_pre'] = True

        #z = basenet_info['activations']['top']
        z = basenet_info['activations']['fc7']

        """
        W_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.0)

        with tf.variable_scope('rotation'):
            c_o = 4
            fc8W = tf.get_variable('weights', [sh[1], c_o], dtype=tf.float32,
                                initializer=W_init)
            fc8b = tf.get_variable('biases', [c_o], dtype=tf.float32,
                                initializer=b_init)
        z = tf.nn.xw_plus_b(z, fc8W, fc8b)
        """

        ladder_losses = []
        ys = []

        if self._method == 'variational':
            assert self._embedding_size is not None

            mu = selfsup.ops.inner(z, self._embedding_size, name='imu', info=info, activation=None)
            log_sigma = selfsup.ops.inner(z, self._embedding_size, name='log_sigma', info=info, activation=None)

            noise = tf.random_normal([batch_size, self._embedding_size], 0, 1, dtype=tf.float32)

            y = self.basenet.decoder(mu + tf.exp(0.5 * log_sigma) * noise, from_name='fc7', channels=3, multiple=1, info=info)
            import ipdb; ipdb.set_trace()

            ys.append(y)

            primary_loss = 50 * tf.reduce_mean(0.25 * (x - y) ** 2)

            # Add loss to encoding
            latent_loss = tf.reduce_mean(
                0.5 * tf.reduce_sum(1 + log_sigma - tf.square(mu) - tf.exp(log_sigma), 1)
            )

            # 0.5 * T.sum(1 + log_sigma - mu**2 - T.exp(log_sigma), axis=1)
            ladder_loss = tf.constant(0.0, dtype=tf.float32)

        elif self._method == 'gustav':
            with tf.variable_scope('decoder') as scope:
                layers =  [('conv2', 'conv1', 0.25), ('conv4', 'conv3', 0.25), ('fc6', 'conv5', 0.25), ('fc7', 'fc6', 0.25)][::-1]

                if self._embedding_size is None:
                    layers = layers[1:]
                    y = self.basenet.decoder(z, from_name='fc6', channels=3, multiple=1, info=info)
                    scope.reuse_variables()
                    layers 
                else:
                    e = selfsup.ops.inner(z, self._embedding_size, name='embedding', info=info)
                    y = self.basenet.decoder(e, from_name='fc7', channels=3, multiple=1, info=info)
                    scope.reuse_variables()
                ys.append(y)

                primary_loss = 50 * tf.reduce_mean(0.25 * (x - y) ** 2)

                for l, l2, mult in layers:
                    z = basenet_info['activations'][l]
                    print(l, z.get_shape().as_list())
                    y = self.basenet.decoder(z, from_name=l2, channels=3, multiple=1, info=info)
                    y = tf.nn.sigmoid(y)

                    ys.append(y)

                    #enc = basenet_info['activations']['pre:' + l]
                    #dec = info['activations']['pre:up' + l]
                    loss = 50 * tf.reduce_mean(mult * (x - y)**2)
                    ladder_losses.append(loss)
                    scope.reuse_variables()

            #primary_loss = ladder_losses.pop(0)
            #primary_loss 
            ladder_loss = tf.add_n(ladder_losses)
        else:
            if self._embedding_size is None:
                y = self.basenet.decoder(z, from_name='fc6', channels=3, multiple=1, info=info)
            else:
                e = seflsup.ops.inner(z, self._embedding_size, name='embedding', info=info)
                y = self.basenet.decoder(e, from_name='fc7', channels=3, multiple=1, info=info)
            ys.append(y)

            primary_loss = 50 * tf.reduce_mean(0.25 * (x - y) ** 2)

            if self._method == 'ladder':
                ladder_losses = []
                with tf.variable_scope('decoder') as scope:
                    for l in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6'][::-1]:
                        e_z = basenet_info['activations']['pre:' + l]
                        d_z = info['activations']['pre:up' + l]

                        d_z = tf.squeeze(d_z)

                        #enc = basenet_info['activations']['pre:' + l]
                        #dec = info['activations']['pre:up' + l]
                        loss = tf.reduce_mean(0.1 * (e_z - d_z)**2)
                        ladder_losses.append(loss)
                ladder_loss = tf.add_n(ladder_losses)
            else:
                ladder_loss = tf.constant(0.0, dtype=tf.float32)

        with tf.name_scope('loss'):
            loss = primary_loss + ladder_loss


        variables = info['vars']

        self.losses = OrderedDict([
            ('main', primary_loss),
            ('lad', ladder_loss),
        ])
        if self._method == 'variational':
            self.losses['latent'] = latent_loss
            loss += latent_loss

        self.primary_loss = primary_loss
        self.loss = loss
        self.feedback_variables = [
            x,
            ys,
            info['weights']['upconv0:weights'],
        ]

        info['activations']['primary_loss'] = primary_loss
        info['activations']['ladder_loss'] = ladder_loss
        info['activations']['loss'] = loss
        #info['activations']['weight_decay'] = weight_decay
        return info

    def feedback(self, variables, iteration):
        import vzlog
        fn = self.make_path('reconstruction', 'png', iteration)
        vzlog.ColorImageGrid([variables[0]] + variables[1]).save(fn)

        fn = self.make_path('upconv1', 'png', iteration)
        vzlog.ColorImageGrid(variables[2].transpose(3, 0, 1, 2),
                vmin=None, vmax=None).save(fn)
