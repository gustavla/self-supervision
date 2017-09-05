import tensorflow as tf
from selfsup.util import DummyDict
from selfsup import ops
from .alex_bn import batch_norm

LAYERS = ['conv0', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']

def decoder(y, from_name='fc7', to_name='conv0', info=DummyDict(), use_batch_norm=False, phase_test=None,
        global_step=None):
    BATCH_SIZE = y.get_shape().as_list()[0]

    if use_batch_norm:
        assert global_step is not None
        def bn(z, name):
            return batch_norm(z, global_step=global_step, phase_test=phase_test, name=name)
    else:
        def bn(z, name):
            return z

    if len(y.get_shape().as_list()) == 2:
        y = tf.expand_dims(tf.expand_dims(y, 1), 1)

    def check(name):
        return name in LAYERS[LAYERS.index(to_name):LAYERS.index(from_name)+1]

    if check('fc7'):
        sh = [BATCH_SIZE, 1, 1, 4096]
        y = ops.upconv(y, sh[-1], size=1, strides=1, info=info, activation=None, name='upfc7', output_shape=sh)
        y = bn(y, 'upfc7')
        info['activations']['pre:upfc7'] = y
        y = tf.nn.relu(y)
        info['activations']['upfc7'] = y

    if check('fc6'):
        sh = [BATCH_SIZE, 1, 1, 4096]
        y = ops.upconv(y, sh[-1], size=1, strides=1, info=info, activation=None, name='upfc6', output_shape=sh)
        y = bn(y, 'upfc6')
        info['activations']['pre:upfc6'] = y
        y = tf.nn.relu(y)
        info['activations']['upfc6'] = y

    if check('conv5'):
        sh = [BATCH_SIZE, 6, 6, 256]
        y = ops.upconv(y, sh[-1], size=6, strides=2, info=info, activation=None, name='upconv5_pre', output_shape=sh, padding='VALID')

        sh = [BATCH_SIZE, 13, 13, 256]
        y = ops.upconv(y, sh[-1], size=3, strides=2, info=info, activation=None, name='upconv5', output_shape=sh, padding='VALID')
        y = bn(y, 'upconv5')
        info['activations']['pre:upconv5'] = y
        y = tf.nn.relu(y)
        info['activations']['upconv5'] = y

    if check('conv4'):
        sh = [BATCH_SIZE, 13, 13, 384]
        y = ops.conv(y, sh[-1], size=3, strides=1, info=info, activation=None, name='upconv4', output_shape=sh, padding='SAME')
        y = bn(y, 'upconv4')
        info['activations']['pre:upconv4'] = y
        y = tf.nn.relu(y)
        info['activations']['upconv4'] = y

    if check('conv3'):
        sh = [BATCH_SIZE, 13, 13, 384]
        y = ops.conv(y, sh[-1], size=3, strides=1, info=info, activation=None, name='upconv3', output_shape=sh, padding='SAME')
        y = bn(y, 'upconv3')
        info['activations']['pre:upconv3'] = y
        y = tf.nn.relu(y)
        info['activations']['upconv3'] = y

    if check('conv2'):
        sh = [BATCH_SIZE, 27, 27, 256]
        y = ops.upconv(y, sh[-1], size=3, strides=2, info=info, activation=None, name='upconv2', output_shape=sh, padding='VALID')
        y = bn(y, 'upconv2')
        info['activations']['pre:upconv2'] = y
        y = tf.nn.relu(y)
        info['activations']['upconv2'] = y

    if check('conv1'):
        sh = [BATCH_SIZE, 57, 57, 96]
        y = ops.upconv(y, sh[-1], size=5, strides=2, info=info, activation=None, name='upconv1', output_shape=sh, padding='VALID')
        y = y[:, 1:-1, 1:-1]
        y = bn(y, 'upconv1')
        info['activations']['pre:upconv1'] = y
        y = tf.nn.relu(y)
        info['activations']['upconv1'] = y

    if check('conv0'):
        sh = [BATCH_SIZE, 227, 227, 3]
        y = ops.upconv(y, sh[-1], size=11, strides=4, info=info, activation=None, name='upconv0', output_shape=sh, padding='VALID')

    return y
