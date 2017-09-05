from __future__ import division, print_function, absolute_import
import tensorflow as tf
import functools
import numpy as np
from selfsup.util import DummyDict
from selfsup import ops, caffe
from selfsup.moving_averages import ExponentialMovingAverageExtended
import sys


def _pretrained_resnet_conv_weights_initializer(name, data, info=None, full_info=None, pre_adjust_batch_norm=False, bn_name=None, scale_name=None):
    shape = None
    #callback = lambda x: x
    if name in data and '0' in data[name]:
        W = data[name]['0'].copy()
        if W.ndim == 2 and name == 'fc1000':
            W = W.reshape((W.shape[0], -1, 1, 1))
        W = W.transpose(2, 3, 1, 0)
        init_type = 'file'
        if name == 'conv1' and W.shape[2] == 3:
            W = W[:, :, ::-1]
            init_type += ':bgr-flipped'
        init = tf.constant_initializer(W)
        #if full_info['config']['return_weights']:
            #full_info['weights'][name+':weights'] = W
        shape = W.shape
    else:
        init_type = 'init'
        init = tf.contrib.layers.variance_scaling_initializer()
    if info is not None:
        info[name + '/weights'] = init_type
    return init, shape


def _pretrained_resnet_inner_weights_initializer(name, data, info=DummyDict(), full_info=DummyDict(), pre_adjust_batch_norm=False, bn_name=None):
    shape = None
    mu = 0.0
    sg = 1.0
    if name in data and '0' in data[name]:
        W = data[name]['0']
        W = W.T
        init_type = 'file'
        #if pre_adjust_batch_norm and bn_name is not None and bn_name in data:
        #    bn_data = data[bn_name]
        #    sigma = np.sqrt(1e-5 + bn_data['1'] / bn_data['2'])
        #    W /= sigma
        #    init_type += ':batch-adjusted'
        if pre_adjust_batch_norm and bn_name is not None and bn_name in data:
            bn_data = data[bn_name]
            bn_sigma = np.sqrt(1e-5 + bn_data['1'] / bn_data['2'])
            sc_sigma = data[scale_name]['0']
            #W /= bn_sigma / sc_sigma
            #callback = lambda x: x * sc_sigma / bn_sigma
            #mu = -bn_data['0'] / bn_data['2'] * sc_sigma / bn_sigma
            mu = data[scale_name]['1'] - bn_data['0'] / bn_data['2'] * sc_sigma / bn_sigma
            #mu = data[scale_name]['1']
            sg = sc_sigma / bn_sigma
            init_type += ':batch-adjusted'#(W*={})'.format(sc_sigma / bn_sigma)
        init = tf.constant_initializer(W.copy())
        #if full_info['config']['return_weights']:
            #full_info['weights'][name+':weights'] = W
        shape = W.shape
    else:
        init_type = 'init'
        init = tf.contrib.layers.variance_scaling_initializer()
    info[name + '/weights'] = init_type
    return init, shape, mu, sg


def _pretrained_resnet_biases_initializer(name, data, info=DummyDict(), full_info=DummyDict(), pre_adjust_batch_norm=False, bn_name=None, scale_name=None):
    shape = None
    #callback = lambda x: x
    if name in data and '1' in data[name]:
        init_type = 'file'
        sc_sigma = data[name]['0'].copy()
        sc_bias = data[name]['1'].copy()
        #if pre_adjust_batch_norm and scale_name is not None and bn_name is not None and bn_name in data:
        if pre_adjust_batch_norm and bn_name is not None and bn_name in data:
            bn_data = data[bn_name]
            bn_sigma = np.sqrt(1e-5 + bn_data['1'] / bn_data['2'])
            mu = bn_data['0'] / bn_data['2']
            #sc_bias = sc_bias - mu * sc_sigma / bn_sigma
            #callback = lambda x: x - mu * sc_sigma / bn_sigma
            #sc_bias = -mu / bn_sigma
            #sc_bias = -mu
            sc_bias = np.zeros_like(mu)
            init_type += ':batch-adjusted'#(b-={})'.format(mu*sc_sigma/bn_sigma)
        init = tf.constant_initializer(sc_bias)
        #if full_info['config']['return_weights']:
            #full_info['weights'][name+':biases'] = sc_bias
        shape = sc_bias.shape
    else:
        init_type = 'init'
        init = tf.constant_initializer(0.0)
    info[name + '/biases'] = init_type
    return init, shape#, callback


def resnet_conv(x, channels, size=3, padding='SAME', stride=1, batch_norm=False,
         phase_test=None, activation=tf.nn.relu, name=None,
         parameter_name=None, bn_name=None, scale_name=None, summarize_scale=False, info=DummyDict(), parameters={},
         pre_adjust_batch_norm=False, iteration=None):
    if parameter_name is None:
        parameter_name = name
    if scale_name is None:
        scale_name = parameter_name
    with tf.name_scope(name):
        features = int(x.get_shape()[3])
        f = channels
        shape = [size, size, features, f]

        W_init, W_shape = _pretrained_resnet_conv_weights_initializer(parameter_name, parameters,
                                                          info=info.get('init'),
                                                          full_info=info)

        #b_init, b_shape = _pretrained_resnet_biases_initializer(scale_name, parameters,
                                                    #info=info.get('init'),
                                                    #full_info=info,
                                                    #pre_adjust_batch_norm=pre_adjust_batch_norm,
                                                    #bn_name=bn_name)

        assert W_shape is None or tuple(W_shape) == tuple(shape), "Incorrect weights shape for {} (file: {}, spec: {})".format(name, W_shape, shape)

        with tf.variable_scope(name):
            W = tf.get_variable('weights', shape, dtype=tf.float32,
                                initializer=W_init)

        raw_conv0 = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)
        #conv0 = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
        if stride > 1:
            conv0 = tf.strided_slice(raw_conv0, [0, 0, 0, 0], raw_conv0.get_shape(), [1, 2, 2, 1])
        else:
            conv0 = raw_conv0

        z = conv0

        if True:
            assert phase_test is not None, "phase_test required for batch norm"
            if bn_name in parameters:
                bn_data = parameters[bn_name]
                bn_mean = bn_data['0'] / bn_data['2']
                bn_var = bn_data['1'] / bn_data['2']
            else:
                bn_mean = np.zeros(f, dtype=np.float32)
                bn_var = np.full(f, 0.5, dtype=np.float32)  # a bit strange, but we don't know

            if scale_name in parameters:
                mu = parameters[scale_name]['1']
                sg = parameters[scale_name]['0']
            else:
                mu = np.zeros(f, dtype=np.float32)
                sg = np.ones(f, dtype=np.float32)

            mm, vv = tf.nn.moments(z, [0, 1, 2], name='mommy')

            assert mu.size == f
            assert sg.size == f
            beta = tf.Variable(tf.constant(mu, shape=[f]), name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(sg, shape=[f]), name='gamma', trainable=True)
            ema = ExponentialMovingAverageExtended(decay=0.999, value=[bn_mean, bn_var],
                    num_updates=iteration)

            def mean_var_train():
                ema_apply_op = ema.apply([mm, vv])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(mm), tf.identity(vv)

            def mean_var_test():
                return ema.average(mm), ema.average(vv)

            mean, var = tf.cond(~phase_test,
                                mean_var_train,
                                mean_var_test)

            info['activations']['last_mean'] = mean
            info['activations']['last_var'] = var

            z = tf.nn.batch_normalization(z, mean, var, beta, gamma, 1e-5)


    info['activations']['preact_' + name] = z

    if activation is not None:
        z = activation(z)

    if info.get('scale_summary'):
        with tf.name_scope('activation'):
            tf.summary.scalar('activation/' + name, tf.sqrt(tf.reduce_mean(z**2)))

    info['activations'][name] = z
    if 'weights' in info:
        info['weights'][name + ':weights'] = W
        #info['weights'][name + ':biases'] = b
    return z


def resnet_atrous_conv(x, channels, size=3, padding='SAME', stride=1, hole=1, batch_norm=False,
         phase_test=None, activation=tf.nn.relu, name=None,
         parameter_name=None, bn_name=None, scale_name=None, summarize_scale=False, info=DummyDict(), parameters={},
         pre_adjust_batch_norm=False):
    if parameter_name is None:
        parameter_name = name
    if scale_name is None:
        scale_name = parameter_name
    with tf.name_scope(name):
        features = int(x.get_shape()[3])
        f = channels
        shape = [size, size, features, f]

        W_init, W_shape = _pretrained_resnet_conv_weights_initializer(parameter_name, parameters,
                                                          info=info.get('init'),
                                                          pre_adjust_batch_norm=pre_adjust_batch_norm,
                                                          bn_name=bn_name, scale_name=scale_name)
        b_init, b_shape = _pretrained_resnet_biases_initializer(scale_name, parameters,
                                                    info=info.get('init'),
                                                    pre_adjust_batch_norm=pre_adjust_batch_norm,
                                                    bn_name=bn_name)

        assert W_shape is None or tuple(W_shape) == tuple(shape), "Incorrect weights shape for {} (file: {}, spec: {})".format(name, W_shape, shape)
        assert b_shape is None or tuple(b_shape) == (f,), "Incorrect bias shape for {} (file: {}, spec; {})".format(name, b_shape, (f,))

        with tf.variable_scope(name):
            W = tf.get_variable('weights', shape, dtype=tf.float32,
                                initializer=W_init)
            b = tf.get_variable('biases', [f], dtype=tf.float32,
                                initializer=b_init)

        if hole == 1:
            raw_conv0 = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)
        else:
            assert stride == 1
            raw_conv0 = tf.nn.atrous_conv2d(x, W, rate=hole, padding=padding)
        #conv0 = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
        if stride > 1:
            conv0 = tf.strided_slice(raw_conv0, [0, 0, 0, 0], raw_conv0.get_shape(), [1, stride, stride, 1])
        else:
            conv0 = raw_conv0
        h1 = tf.reshape(tf.nn.bias_add(conv0, b), conv0.get_shape())

        z = h1

    if activation is not None:
        z = activation(z)

    if info.get('scale_summary'):
        with tf.name_scope('activation'):
            tf.summary.scalar('activation/' + name, tf.sqrt(tf.reduce_mean(z**2)))

    info['activations'][name] = z
    return z


def resnet_inner(x, channels, info=DummyDict(), stddev=None,
              activation=tf.nn.relu, name=None, parameters={},
              parameter_name=None):
    if parameter_name is None:
        parameter_name = name
    with tf.name_scope(name):
        f = channels
        features = np.prod(x.get_shape().as_list()[1:])
        xflat = tf.reshape(x, [-1, features])
        shape = [features, channels]

        W_init, W_shape, mu, sg = _pretrained_resnet_inner_weights_initializer(parameter_name, parameters, info=info.get('init'))
        b_init, b_shape = _pretrained_resnet_biases_initializer(parameter_name, parameters, info=info.get('init'))

        assert W_shape is None or tuple(W_shape) == tuple(shape), "Incorrect weights shape for {} (file: {}, spec: {})".format(name, W_shape, shape)
        assert b_shape is None or tuple(b_shape) == (f,), "Incorrect bias shape for {} (file: {}, spec; {})".format(name, b_shape, (f,))

        with tf.variable_scope(name):
            W = tf.get_variable('weights', shape, dtype=tf.float32,
                                initializer=W_init)
            #b = tf.get_variable('biases', [f], dtype=tf.float32,
                                #initializer=b_init)

        z = tf.matmul(xflat, W)
        z = z * sg + mu
        #z = tf.nn.bias_add(z, b)
    if activation is not None:
        z = activation(z)
    info['activations'][name] = z

    if info.get('scale_summary'):
        with tf.name_scope('activation'):
            tf.summary.scalar('activation/' + name, tf.sqrt(tf.reduce_mean(z**2)))

    return z


def build_network(x, info=DummyDict(), parameters={},
                  phase_test=None, convolutional=False, final_layer=True,
                  pre_adjust_batch_norm=False,
                  num_features_mult=1.0, iteration=None):

    # Set up VGG-16
    conv = functools.partial(resnet_conv, size=3, parameters=parameters,
                             info=info, pre_adjust_batch_norm=pre_adjust_batch_norm,
                             phase_test=phase_test, iteration=iteration)
    pool = functools.partial(ops.max_pool, info=info)
    avg_pool = functools.partial(ops.avg_pool, info=info)
    dropout = functools.partial(ops.dropout, phase_test=phase_test, info=info)

    def num(f):
        return int(f * num_features_mult)

    z = x
    conv1 = conv(z, num(64), size=7, stride=2, name='conv1', bn_name='bn_conv1',
                 scale_name='scale_conv1')

    pool1 = pool(conv1, 3, stride=2, name='pool1')

    res2a_branch1 = conv(pool1, num(256), size=1, name='res2a_branch1', bn_name='bn2a_branch1',
                 scale_name='scale2a_branch1', activation=None)

    res2a_branch2a = conv(pool1, num(64), size=1, name='res2a_branch2a', bn_name='bn2a_branch2a',
                scale_name='scale2a_branch2a')

    res2a_branch2b = conv(res2a_branch2a, num(64), size=3, name='res2a_branch2b', bn_name='bn2a_branch2b',
                          scale_name='scale2a_branch2b')
    res2a_branch2c = conv(res2a_branch2b, num(256), size=1, name='res2a_branch2c', bn_name='bn2a_branch2c',
                          scale_name='scale2a_branch2c', activation=None)

    res2a_preact = tf.add(res2a_branch1, res2a_branch2c)
    info['activations']['preact_res2a'] = res2a_preact
    res2a = tf.nn.relu(res2a_preact, name='res2a')
    info['activations']['res2a'] = res2a


    # ---
    """
:call nobias-conv 1 0 1 64 res2a res2b_branch2a
:call batch-norm res2b_branch2a bn2b_branch2a
:call bias res2b_branch2a scale2b_branch2a
:call relu res2b_branch2a
:#
:call nobias-conv 3 1 1 64 res2b_branch2a res2b_branch2b
:call batch-norm res2b_branch2b bn2b_branch2b
:call bias res2b_branch2b scale2b_branch2b
:call relu res2b_branch2b
:#
:call nobias-conv 1 0 1 256 res2b_branch2b res2b_branch2c
:call batch-norm res2b_branch2c bn2b_branch2c
:call bias res2b_branch2c scale2b_branch2c
:call add res2a res2b_branch2c res2b
:call relu res2b
    """

    def block(x, ch1, ch2, b):
        output = 'res{}'.format(b)
        branch2a = conv(x, num(ch1), size=1, name='res{}_branch2a'.format(b),
                              bn_name='bn{}_branch2a'.format(b),
                              scale_name='scale{}_branch2a'.format(b))
        branch2b = conv(branch2a, num(ch1), size=3, name='res{}_branch2b'.format(b),
                              bn_name='bn{}_branch2b'.format(b),
                              scale_name='scale{}_branch2b'.format(b))
        branch2c = conv(branch2b, num(ch2), size=1, name='res{}_branch2c'.format(b),
                              bn_name='bn{}_branch2c'.format(b),
                              scale_name='scale{}_branch2c'.format(b), activation=None)
        z0 = tf.add(x, branch2c)
        info['activations']['preact_' + output] = z0
        z = tf.nn.relu(z0, name=output)
        info['activations'][output] = z
        return z

    """
:call nobias-conv 1 0 2 ${ch2} res${a} res${b}_branch1
:call batch-norm res${b}_branch1 bn${b}_branch1
:call bias res${b}_branch1 scale${b}_branch1
:#
:call nobias-conv 1 0 2 ${ch1} res${a} res${b}_branch2a
:call batch-norm res${b}_branch2a bn${b}_branch2a
:call bias res${b}_branch2a scale${b}_branch2a
:call relu res${b}_branch2a
:#
:call nobias-conv 3 1 1 ${ch1} res${b}_branch2a res${b}_branch2b
:call batch-norm res${b}_branch2b bn${b}_branch2b
:call bias res${b}_branch2b scale${b}_branch2b
:call relu res${b}_branch2b
:#
:call nobias-conv 1 0 1 ${ch2} res${b}_branch2b res${b}_branch2c
:call batch-norm res${b}_branch2c bn${b}_branch2c
:call bias res${b}_branch2c scale${b}_branch2c
:call add res${b}_branch1 res${b}_branch2c res${b}
:call relu res${b}
    """
    def block_reduce(x, ch1, ch2, b, stride=2):
        output = 'res{}'.format(b)
        branch1 = conv(x, num(ch2), size=1, stride=stride,
                       name='res{}_branch1'.format(b),
                       bn_name='bn{}_branch1'.format(b),
                       scale_name='scale{}_branch1'.format(b),
                       activation=None)

        branch2a = conv(x, num(ch1), size=1, stride=stride,
                        name='res{}_branch2a'.format(b),
                        bn_name='bn{}_branch2a'.format(b),
                        scale_name='scale{}_branch2a'.format(b))
        branch2b = conv(branch2a, num(ch1), size=3,
                        name='res{}_branch2b'.format(b),
                        bn_name='bn{}_branch2b'.format(b),
                        scale_name='scale{}_branch2b'.format(b))
        branch2c = conv(branch2b, num(ch2), size=1,
                        name='res{}_branch2c'.format(b),
                        bn_name='bn{}_branch2c'.format(b),
                        scale_name='scale{}_branch2c'.format(b), activation=None)
        z0 = tf.add(branch1, branch2c)
        info['activations']['preact_' + output] = z0
        z = tf.nn.relu(z0, name=output)
        #z = tf.nn.relu(tf.add(branch1, branch2c), name=output)
        info['activations'][output] = z
        return z


    res2b = block(res2a, 64, 256, '2b')
    res2c = block(res2b, 64, 256, '2c')

    res3a = block_reduce(res2c, 128, 512, '3a')
    """
:call resnet 128 512 3a  3b1
:call resnet 128 512 3b1 3b2
:call resnet 128 512 3b2 3b3
:call resnet 128 512 3b3 3b4
:call resnet 128 512 3b4 3b5
:call resnet 128 512 3b5 3b6
:call resnet 128 512 3b6 3b7
    """
    res3b1 = block(res3a,  128, 512, '3b1')
    res3b2 = block(res3b1, 128, 512, '3b2')
    res3b3 = block(res3b2, 128, 512, '3b3')
    res3b4 = block(res3b3, 128, 512, '3b4')
    res3b5 = block(res3b4, 128, 512, '3b5')
    res3b6 = block(res3b5, 128, 512, '3b6')
    res3b7 = block(res3b6, 128, 512, '3b7')
    """
:call resnet-reduce 256 1024 3b7 4a
    """
    res4a = block_reduce(res3b7, 256, 1024, '4a')
    """
:call resnet 256 1024 4a 4b1
:call resnet 256 1024 4b1 4b2
:call resnet 256 1024 4b2 4b3
:call resnet 256 1024 4b3 4b4
:call resnet 256 1024 4b4 4b5
:call resnet 256 1024 4b5 4b6
:call resnet 256 1024 4b6 4b7
:call resnet 256 1024 4b7 4b8
:call resnet 256 1024 4b8 4b9
:call resnet 256 1024 4b9 4b10
:call resnet 256 1024 4b10 4b11
:call resnet 256 1024 4b11 4b12
:call resnet 256 1024 4b12 4b13
:call resnet 256 1024 4b13 4b14
:call resnet 256 1024 4b14 4b15
:call resnet 256 1024 4b15 4b16
:call resnet 256 1024 4b16 4b17
:call resnet 256 1024 4b17 4b18
:call resnet 256 1024 4b18 4b19
:call resnet 256 1024 4b19 4b20
:call resnet 256 1024 4b20 4b21
:call resnet 256 1024 4b21 4b22
:call resnet 256 1024 4b22 4b23
:call resnet 256 1024 4b23 4b24
:call resnet 256 1024 4b24 4b25
:call resnet 256 1024 4b25 4b26
:call resnet 256 1024 4b26 4b27
:call resnet 256 1024 4b27 4b28
:call resnet 256 1024 4b28 4b29
:call resnet 256 1024 4b29 4b30
:call resnet 256 1024 4b30 4b31
:call resnet 256 1024 4b31 4b32
:call resnet 256 1024 4b32 4b33
:call resnet 256 1024 4b33 4b34
:call resnet 256 1024 4b34 4b35
    """
    res4b1 = block(res4a,    256, 1024, '4b1')
    res4b2 = block(res4b1,   256, 1024, '4b2')
    res4b3 = block(res4b2,   256, 1024, '4b3')
    res4b4 = block(res4b3,   256, 1024, '4b4')
    res4b5 = block(res4b4,   256, 1024, '4b5')
    res4b6 = block(res4b5,   256, 1024, '4b6')
    res4b7 = block(res4b6,   256, 1024, '4b7')
    res4b8 = block(res4b7,   256, 1024, '4b8')
    res4b9 = block(res4b8,   256, 1024, '4b9')
    res4b10 = block(res4b9,  256, 1024, '4b10')
    res4b11 = block(res4b10, 256, 1024, '4b11')
    res4b12 = block(res4b11, 256, 1024, '4b12')
    res4b13 = block(res4b12, 256, 1024, '4b13')
    res4b14 = block(res4b13, 256, 1024, '4b14')
    res4b15 = block(res4b14, 256, 1024, '4b15')
    res4b16 = block(res4b15, 256, 1024, '4b16')
    res4b17 = block(res4b16, 256, 1024, '4b17')
    res4b18 = block(res4b17, 256, 1024, '4b18')
    res4b19 = block(res4b18, 256, 1024, '4b19')
    res4b20 = block(res4b19, 256, 1024, '4b20')
    res4b21 = block(res4b20, 256, 1024, '4b21')
    res4b22 = block(res4b21, 256, 1024, '4b22')
    res4b23 = block(res4b22, 256, 1024, '4b23')
    res4b24 = block(res4b23, 256, 1024, '4b24')
    res4b25 = block(res4b24, 256, 1024, '4b25')
    res4b26 = block(res4b25, 256, 1024, '4b26')
    res4b27 = block(res4b26, 256, 1024, '4b27')
    res4b28 = block(res4b27, 256, 1024, '4b28')
    res4b29 = block(res4b28, 256, 1024, '4b29')
    res4b30 = block(res4b29, 256, 1024, '4b30')
    res4b31 = block(res4b30, 256, 1024, '4b31')
    res4b32 = block(res4b31, 256, 1024, '4b32')
    res4b33 = block(res4b32, 256, 1024, '4b33')
    res4b34 = block(res4b33, 256, 1024, '4b34')
    res4b35 = block(res4b34, 256, 1024, '4b35')
    """
:call resnet-reduce 512 2048 4b35 5a
    """
    res5a = block_reduce(res4b35, 512, 2048, '5a')

    """
:call resnet 512 2048 5a 5b
:call resnet 512 2048 5b 5c
    """
    res5b = block(res5a, 512, 2048, '5b')
    res5c = block(res5b, 512, 2048, '5c')

    """
layer {
    bottom: "res5c"
    top: "pool5"
    name: "pool5"
    type: "Pooling"
    pooling_param {
        kernel_size: 7
        stride: 1
        pool: AVE
    }
}
    """
    if final_layer:
        pool5 = avg_pool(res5c, 7, stride=1, name='pool5', padding='VALID')
        if convolutional:
            z = conv(pool5, 1000, size=1, name='fc1000', activation=None)
        else:
            z = resnet_inner(pool5, 1000, info=info, parameters=parameters, activation=None, name='fc1000')
    else:
        z = res5c

    return z


def build_network_atrous2(x, info=DummyDict(), parameters={},
                  phase_test=None, convolutional=False, final_layer=True,
                  pre_adjust_batch_norm=False):

    # Set up VGG-16
    conv = functools.partial(resnet_conv, size=3, parameters=parameters,
                             info=info, pre_adjust_batch_norm=pre_adjust_batch_norm)
    aconv = functools.partial(resnet_atrous_conv, size=3, parameters=parameters,
                             info=info, pre_adjust_batch_norm=pre_adjust_batch_norm)
    pool = functools.partial(ops.max_pool, info=info)
    avg_pool = functools.partial(ops.avg_pool, info=info)
    dropout = functools.partial(ops.dropout, phase_test=phase_test, info=info)

    z = x
    conv1 = conv(z, 64, size=7, stride=2, name='conv1', bn_name='bn_conv1',
                 scale_name='scale_conv1')

    pool1 = pool(conv1, 3, stride=2, name='pool1')

    res2a_branch1 = conv(pool1, 256, size=1, name='res2a_branch1', bn_name='bn2a_branch1',
                 scale_name='scale2a_branch1', activation=None)

    res2a_branch2a = conv(pool1, 64, size=1, name='res2a_branch2a', bn_name='bn2a_branch2a',
                scale_name='scale2a_branch2a')

    res2a_branch2b = conv(res2a_branch2a, 64, size=3, name='res2a_branch2b', bn_name='bn2a_branch2b',
                          scale_name='scale2a_branch2b')
    res2a_branch2c = conv(res2a_branch2b, 256, size=1, name='res2a_branch2c', bn_name='bn2a_branch2c',
                          scale_name='scale2a_branch2c', activation=None)

    res2a = tf.nn.relu(tf.add(res2a_branch1, res2a_branch2c), name='res2a')
    info['activations']['res2a'] = res2a


    # ---
    """
:call nobias-conv 1 0 1 64 res2a res2b_branch2a
:call batch-norm res2b_branch2a bn2b_branch2a
:call bias res2b_branch2a scale2b_branch2a
:call relu res2b_branch2a
:#
:call nobias-conv 3 1 1 64 res2b_branch2a res2b_branch2b
:call batch-norm res2b_branch2b bn2b_branch2b
:call bias res2b_branch2b scale2b_branch2b
:call relu res2b_branch2b
:#
:call nobias-conv 1 0 1 256 res2b_branch2b res2b_branch2c
:call batch-norm res2b_branch2c bn2b_branch2c
:call bias res2b_branch2c scale2b_branch2c
:call add res2a res2b_branch2c res2b
:call relu res2b
    """

    def block(x, ch1, ch2, b, hole=1):
        output = 'res{}'.format(b)
        branch2a = aconv(x, ch1, size=1, hole=hole, name='res{}_branch2a'.format(b),
                              bn_name='bn{}_branch2a'.format(b),
                              scale_name='scale{}_branch2a'.format(b))
        branch2b = aconv(branch2a, ch1, size=3, hole=hole, name='res{}_branch2b'.format(b),
                              bn_name='bn{}_branch2b'.format(b),
                              scale_name='scale{}_branch2b'.format(b))
        branch2c = aconv(branch2b, ch2, size=1, hole=hole, name='res{}_branch2c'.format(b),
                              bn_name='bn{}_branch2c'.format(b),
                              scale_name='scale{}_branch2c'.format(b), activation=None)
        z = tf.nn.relu(tf.add(x, branch2c), name=output)
        info['activations'][output] = z
        return z


    """
:call nobias-conv 1 0 2 ${ch2} res${a} res${b}_branch1
:call batch-norm res${b}_branch1 bn${b}_branch1
:call bias res${b}_branch1 scale${b}_branch1
:#
:call nobias-conv 1 0 2 ${ch1} res${a} res${b}_branch2a
:call batch-norm res${b}_branch2a bn${b}_branch2a
:call bias res${b}_branch2a scale${b}_branch2a
:call relu res${b}_branch2a
:#
:call nobias-conv 3 1 1 ${ch1} res${b}_branch2a res${b}_branch2b
:call batch-norm res${b}_branch2b bn${b}_branch2b
:call bias res${b}_branch2b scale${b}_branch2b
:call relu res${b}_branch2b
:#
:call nobias-conv 1 0 1 ${ch2} res${b}_branch2b res${b}_branch2c
:call batch-norm res${b}_branch2c bn${b}_branch2c
:call bias res${b}_branch2c scale${b}_branch2c
:call add res${b}_branch1 res${b}_branch2c res${b}
:call relu res${b}
    """
    def block_reduce(x, ch1, ch2, b, stride=2):
        output = 'res{}'.format(b)
        branch1 = conv(x, ch2, size=1, stride=stride,
                       name='res{}_branch1'.format(b),
                       bn_name='bn{}_branch1'.format(b),
                       scale_name='scale{}_branch1'.format(b),
                       activation=None)

        branch2a = conv(x, ch1, size=1, stride=stride,
                        name='res{}_branch2a'.format(b),
                        bn_name='bn{}_branch2a'.format(b),
                        scale_name='scale{}_branch2a'.format(b))
        branch2b = conv(branch2a, ch1, size=3,
                        name='res{}_branch2b'.format(b),
                        bn_name='bn{}_branch2b'.format(b),
                        scale_name='scale{}_branch2b'.format(b))
        branch2c = conv(branch2b, ch2, size=1,
                        name='res{}_branch2c'.format(b),
                        bn_name='bn{}_branch2c'.format(b),
                        scale_name='scale{}_branch2c'.format(b), activation=None)
        z = tf.nn.relu(tf.add(branch1, branch2c), name=output)
        info['activations'][output] = z
        return z

    res2b = block(res2a, 64, 256, '2b')
    res2c = block(res2b, 64, 256, '2c')

    res3a = block_reduce(res2c, 128, 512, '3a')
    """
:call resnet 128 512 3a  3b1
:call resnet 128 512 3b1 3b2
:call resnet 128 512 3b2 3b3
:call resnet 128 512 3b3 3b4
:call resnet 128 512 3b4 3b5
:call resnet 128 512 3b5 3b6
:call resnet 128 512 3b6 3b7
    """
    res3b1 = block(res3a,  128, 512, '3b1')
    res3b2 = block(res3b1, 128, 512, '3b2')
    res3b3 = block(res3b2, 128, 512, '3b3')
    res3b4 = block(res3b3, 128, 512, '3b4')
    res3b5 = block(res3b4, 128, 512, '3b5')
    res3b6 = block(res3b5, 128, 512, '3b6')
    res3b7 = block(res3b6, 128, 512, '3b7')
    """
:call resnet-reduce 256 1024 3b7 4a
    """
    res4a = block_reduce(res3b7, 256, 1024, '4a')
    """
:call resnet 256 1024 4a 4b1
:call resnet 256 1024 4b1 4b2
:call resnet 256 1024 4b2 4b3
:call resnet 256 1024 4b3 4b4
:call resnet 256 1024 4b4 4b5
:call resnet 256 1024 4b5 4b6
:call resnet 256 1024 4b6 4b7
:call resnet 256 1024 4b7 4b8
:call resnet 256 1024 4b8 4b9
:call resnet 256 1024 4b9 4b10
:call resnet 256 1024 4b10 4b11
:call resnet 256 1024 4b11 4b12
:call resnet 256 1024 4b12 4b13
:call resnet 256 1024 4b13 4b14
:call resnet 256 1024 4b14 4b15
:call resnet 256 1024 4b15 4b16
:call resnet 256 1024 4b16 4b17
:call resnet 256 1024 4b17 4b18
:call resnet 256 1024 4b18 4b19
:call resnet 256 1024 4b19 4b20
:call resnet 256 1024 4b20 4b21
:call resnet 256 1024 4b21 4b22
:call resnet 256 1024 4b22 4b23
:call resnet 256 1024 4b23 4b24
:call resnet 256 1024 4b24 4b25
:call resnet 256 1024 4b25 4b26
:call resnet 256 1024 4b26 4b27
:call resnet 256 1024 4b27 4b28
:call resnet 256 1024 4b28 4b29
:call resnet 256 1024 4b29 4b30
:call resnet 256 1024 4b30 4b31
:call resnet 256 1024 4b31 4b32
:call resnet 256 1024 4b32 4b33
:call resnet 256 1024 4b33 4b34
:call resnet 256 1024 4b34 4b35
    """
    res4b1 = block(res4a,    256, 1024, '4b1')
    res4b2 = block(res4b1,   256, 1024, '4b2')
    res4b3 = block(res4b2,   256, 1024, '4b3')
    res4b4 = block(res4b3,   256, 1024, '4b4')
    res4b5 = block(res4b4,   256, 1024, '4b5')
    res4b6 = block(res4b5,   256, 1024, '4b6')
    res4b7 = block(res4b6,   256, 1024, '4b7')
    res4b8 = block(res4b7,   256, 1024, '4b8')
    res4b9 = block(res4b8,   256, 1024, '4b9')
    res4b10 = block(res4b9,  256, 1024, '4b10')
    res4b11 = block(res4b10, 256, 1024, '4b11')
    res4b12 = block(res4b11, 256, 1024, '4b12')
    res4b13 = block(res4b12, 256, 1024, '4b13')
    res4b14 = block(res4b13, 256, 1024, '4b14')
    res4b15 = block(res4b14, 256, 1024, '4b15')
    res4b16 = block(res4b15, 256, 1024, '4b16')
    res4b17 = block(res4b16, 256, 1024, '4b17')
    res4b18 = block(res4b17, 256, 1024, '4b18')
    res4b19 = block(res4b18, 256, 1024, '4b19')
    res4b20 = block(res4b19, 256, 1024, '4b20')
    res4b21 = block(res4b20, 256, 1024, '4b21')
    res4b22 = block(res4b21, 256, 1024, '4b22')
    res4b23 = block(res4b22, 256, 1024, '4b23')
    res4b24 = block(res4b23, 256, 1024, '4b24')
    res4b25 = block(res4b24, 256, 1024, '4b25')
    res4b26 = block(res4b25, 256, 1024, '4b26')
    res4b27 = block(res4b26, 256, 1024, '4b27')
    res4b28 = block(res4b27, 256, 1024, '4b28')
    res4b29 = block(res4b28, 256, 1024, '4b29')
    res4b30 = block(res4b29, 256, 1024, '4b30')
    res4b31 = block(res4b30, 256, 1024, '4b31')
    res4b32 = block(res4b31, 256, 1024, '4b32')
    res4b33 = block(res4b32, 256, 1024, '4b33')
    res4b34 = block(res4b33, 256, 1024, '4b34')
    res4b35 = block(res4b34, 256, 1024, '4b35')
    """
:call resnet-reduce 512 2048 4b35 5a
    """
    res5a = block_reduce(res4b35, 512, 2048, '5a', stride=1)

    """
:call resnet 512 2048 5a 5b
:call resnet 512 2048 5b 5c
    """
    res5b = block(res5a, 512, 2048, '5b', hole=2)
    res5c = block(res5b, 512, 2048, '5c', hole=2)

    """
layer {
    bottom: "res5c"
    top: "pool5"
    name: "pool5"
    type: "Pooling"
    pooling_param {
        kernel_size: 7
        stride: 1
        pool: AVE
    }
}
    """
    if final_layer:
        pool5 = avg_pool(res5c, 7, stride=1, name='pool5', padding='VALID')
        if convolutional:
            z = conv(pool5, 1000, size=1, name='fc1000', activation=None)
        else:
            z = resnet_inner(pool5, 1000, info=info, parameters=parameters, activation=None, name='fc1000')
    else:
        z = res5c

    return z

def build_network_atrous4(x, info=DummyDict(), parameters={},
                  phase_test=None, convolutional=False, final_layer=True,
                  pre_adjust_batch_norm=False):

    # Set up VGG-16
    conv = functools.partial(resnet_conv, size=3, parameters=parameters,
                             info=info, pre_adjust_batch_norm=pre_adjust_batch_norm)
    aconv = functools.partial(resnet_atrous_conv, size=3, parameters=parameters,
                             info=info, pre_adjust_batch_norm=pre_adjust_batch_norm)
    pool = functools.partial(ops.max_pool, info=info)
    avg_pool = functools.partial(ops.avg_pool, info=info)
    dropout = functools.partial(ops.dropout, phase_test=phase_test, info=info)

    z = x
    conv1 = conv(z, 64, size=7, stride=2, name='conv1', bn_name='bn_conv1',
                 scale_name='scale_conv1')

    pool1 = pool(conv1, 3, stride=2, name='pool1')

    res2a_branch1 = conv(pool1, 256, size=1, name='res2a_branch1', bn_name='bn2a_branch1',
                 scale_name='scale2a_branch1', activation=None)

    res2a_branch2a = conv(pool1, 64, size=1, name='res2a_branch2a', bn_name='bn2a_branch2a',
                scale_name='scale2a_branch2a')

    res2a_branch2b = conv(res2a_branch2a, 64, size=3, name='res2a_branch2b', bn_name='bn2a_branch2b',
                          scale_name='scale2a_branch2b')
    res2a_branch2c = conv(res2a_branch2b, 256, size=1, name='res2a_branch2c', bn_name='bn2a_branch2c',
                          scale_name='scale2a_branch2c', activation=None)

    res2a = tf.nn.relu(tf.add(res2a_branch1, res2a_branch2c), name='res2a')
    info['activations']['res2a'] = res2a


    # ---
    """
:call nobias-conv 1 0 1 64 res2a res2b_branch2a
:call batch-norm res2b_branch2a bn2b_branch2a
:call bias res2b_branch2a scale2b_branch2a
:call relu res2b_branch2a
:#
:call nobias-conv 3 1 1 64 res2b_branch2a res2b_branch2b
:call batch-norm res2b_branch2b bn2b_branch2b
:call bias res2b_branch2b scale2b_branch2b
:call relu res2b_branch2b
:#
:call nobias-conv 1 0 1 256 res2b_branch2b res2b_branch2c
:call batch-norm res2b_branch2c bn2b_branch2c
:call bias res2b_branch2c scale2b_branch2c
:call add res2a res2b_branch2c res2b
:call relu res2b
    """

    def block(x, ch1, ch2, b, hole=1):
        output = 'res{}'.format(b)
        branch2a = conv(x, ch1, size=1, name='res{}_branch2a'.format(b),
                              bn_name='bn{}_branch2a'.format(b),
                              scale_name='scale{}_branch2a'.format(b))
        branch2b = aconv(branch2a, ch1, size=3, hole=hole, name='res{}_branch2b'.format(b),
                              bn_name='bn{}_branch2b'.format(b),
                              scale_name='scale{}_branch2b'.format(b))
        branch2c = conv(branch2b, ch2, size=1, name='res{}_branch2c'.format(b),
                              bn_name='bn{}_branch2c'.format(b),
                              scale_name='scale{}_branch2c'.format(b), activation=None)
        z = tf.nn.relu(tf.add(x, branch2c), name=output)
        info['activations'][output] = z
        return z

    """
:call nobias-conv 1 0 2 ${ch2} res${a} res${b}_branch1
:call batch-norm res${b}_branch1 bn${b}_branch1
:call bias res${b}_branch1 scale${b}_branch1
:#
:call nobias-conv 1 0 2 ${ch1} res${a} res${b}_branch2a
:call batch-norm res${b}_branch2a bn${b}_branch2a
:call bias res${b}_branch2a scale${b}_branch2a
:call relu res${b}_branch2a
:#
:call nobias-conv 3 1 1 ${ch1} res${b}_branch2a res${b}_branch2b
:call batch-norm res${b}_branch2b bn${b}_branch2b
:call bias res${b}_branch2b scale${b}_branch2b
:call relu res${b}_branch2b
:#
:call nobias-conv 1 0 1 ${ch2} res${b}_branch2b res${b}_branch2c
:call batch-norm res${b}_branch2c bn${b}_branch2c
:call bias res${b}_branch2c scale${b}_branch2c
:call add res${b}_branch1 res${b}_branch2c res${b}
:call relu res${b}
    """
    def block_reduce(x, ch1, ch2, b, stride=2, hole=1):
        output = 'res{}'.format(b)
        branch1 = conv(x, ch2, size=1, stride=stride,
                       name='res{}_branch1'.format(b),
                       bn_name='bn{}_branch1'.format(b),
                       scale_name='scale{}_branch1'.format(b),
                       activation=None)

        branch2a = conv(x, ch1, size=1, stride=stride,
                        name='res{}_branch2a'.format(b),
                        bn_name='bn{}_branch2a'.format(b),
                        scale_name='scale{}_branch2a'.format(b))
        branch2b = aconv(branch2a, ch1, size=3, hole=hole,
                        name='res{}_branch2b'.format(b),
                        bn_name='bn{}_branch2b'.format(b),
                        scale_name='scale{}_branch2b'.format(b))
        branch2c = conv(branch2b, ch2, size=1,
                        name='res{}_branch2c'.format(b),
                        bn_name='bn{}_branch2c'.format(b),
                        scale_name='scale{}_branch2c'.format(b), activation=None)
        z = tf.nn.relu(tf.add(branch1, branch2c), name=output)
        info['activations'][output] = z
        return z

    res2b = block(res2a, 64, 256, '2b')
    res2c = block(res2b, 64, 256, '2c')

    res3a = block_reduce(res2c, 128, 512, '3a')
    """
:call resnet 128 512 3a  3b1
:call resnet 128 512 3b1 3b2
:call resnet 128 512 3b2 3b3
:call resnet 128 512 3b3 3b4
:call resnet 128 512 3b4 3b5
:call resnet 128 512 3b5 3b6
:call resnet 128 512 3b6 3b7
    """
    res3b1 = block(res3a,  128, 512, '3b1')
    res3b2 = block(res3b1, 128, 512, '3b2')
    res3b3 = block(res3b2, 128, 512, '3b3')
    res3b4 = block(res3b3, 128, 512, '3b4')
    res3b5 = block(res3b4, 128, 512, '3b5')
    res3b6 = block(res3b5, 128, 512, '3b6')
    res3b7 = block(res3b6, 128, 512, '3b7')
    """
:call resnet-reduce 256 1024 3b7 4a
    """
    res4a = block_reduce(res3b7, 256, 1024, '4a', stride=1, hole=2)
    """
:call resnet 256 1024 4a 4b1
:call resnet 256 1024 4b1 4b2
:call resnet 256 1024 4b2 4b3
:call resnet 256 1024 4b3 4b4
:call resnet 256 1024 4b4 4b5
:call resnet 256 1024 4b5 4b6
:call resnet 256 1024 4b6 4b7
:call resnet 256 1024 4b7 4b8
:call resnet 256 1024 4b8 4b9
:call resnet 256 1024 4b9 4b10
:call resnet 256 1024 4b10 4b11
:call resnet 256 1024 4b11 4b12
:call resnet 256 1024 4b12 4b13
:call resnet 256 1024 4b13 4b14
:call resnet 256 1024 4b14 4b15
:call resnet 256 1024 4b15 4b16
:call resnet 256 1024 4b16 4b17
:call resnet 256 1024 4b17 4b18
:call resnet 256 1024 4b18 4b19
:call resnet 256 1024 4b19 4b20
:call resnet 256 1024 4b20 4b21
:call resnet 256 1024 4b21 4b22
:call resnet 256 1024 4b22 4b23
:call resnet 256 1024 4b23 4b24
:call resnet 256 1024 4b24 4b25
:call resnet 256 1024 4b25 4b26
:call resnet 256 1024 4b26 4b27
:call resnet 256 1024 4b27 4b28
:call resnet 256 1024 4b28 4b29
:call resnet 256 1024 4b29 4b30
:call resnet 256 1024 4b30 4b31
:call resnet 256 1024 4b31 4b32
:call resnet 256 1024 4b32 4b33
:call resnet 256 1024 4b33 4b34
:call resnet 256 1024 4b34 4b35
    """
    res4b1 = block(res4a,    256, 1024, '4b1', hole=2)
    res4b2 = block(res4b1,   256, 1024, '4b2', hole=2)
    res4b3 = block(res4b2,   256, 1024, '4b3', hole=2)
    res4b4 = block(res4b3,   256, 1024, '4b4', hole=2)
    res4b5 = block(res4b4,   256, 1024, '4b5', hole=2)
    res4b6 = block(res4b5,   256, 1024, '4b6', hole=2)
    res4b7 = block(res4b6,   256, 1024, '4b7', hole=2)
    res4b8 = block(res4b7,   256, 1024, '4b8', hole=2)
    res4b9 = block(res4b8,   256, 1024, '4b9', hole=2)
    res4b10 = block(res4b9,  256, 1024, '4b10', hole=2)
    res4b11 = block(res4b10, 256, 1024, '4b11', hole=2)
    res4b12 = block(res4b11, 256, 1024, '4b12', hole=2)
    res4b13 = block(res4b12, 256, 1024, '4b13', hole=2)
    res4b14 = block(res4b13, 256, 1024, '4b14', hole=2)
    res4b15 = block(res4b14, 256, 1024, '4b15', hole=2)
    res4b16 = block(res4b15, 256, 1024, '4b16', hole=2)
    res4b17 = block(res4b16, 256, 1024, '4b17', hole=2)
    res4b18 = block(res4b17, 256, 1024, '4b18', hole=2)
    res4b19 = block(res4b18, 256, 1024, '4b19', hole=2)
    res4b20 = block(res4b19, 256, 1024, '4b20', hole=2)
    res4b21 = block(res4b20, 256, 1024, '4b21', hole=2)
    res4b22 = block(res4b21, 256, 1024, '4b22', hole=2)
    res4b23 = block(res4b22, 256, 1024, '4b23', hole=2)
    res4b24 = block(res4b23, 256, 1024, '4b24', hole=2)
    res4b25 = block(res4b24, 256, 1024, '4b25', hole=2)
    res4b26 = block(res4b25, 256, 1024, '4b26', hole=2)
    res4b27 = block(res4b26, 256, 1024, '4b27', hole=2)
    res4b28 = block(res4b27, 256, 1024, '4b28', hole=2)
    res4b29 = block(res4b28, 256, 1024, '4b29', hole=2)
    res4b30 = block(res4b29, 256, 1024, '4b30', hole=2)
    res4b31 = block(res4b30, 256, 1024, '4b31', hole=2)
    res4b32 = block(res4b31, 256, 1024, '4b32', hole=2)
    res4b33 = block(res4b32, 256, 1024, '4b33', hole=2)
    res4b34 = block(res4b33, 256, 1024, '4b34', hole=2)
    res4b35 = block(res4b34, 256, 1024, '4b35', hole=2)
    """
:call resnet-reduce 512 2048 4b35 5a
    """
    res5a = block_reduce(res4b35, 512, 2048, '5a', stride=1, hole=4)

    """
:call resnet 512 2048 5a 5b
:call resnet 512 2048 5b 5c
    """
    res5b = block(res5a, 512, 2048, '5b', hole=4)
    res5c = block(res5b, 512, 2048, '5c', hole=4)

    """
layer {
    bottom: "res5c"
    top: "pool5"
    name: "pool5"
    type: "Pooling"
    pooling_param {
        kernel_size: 7
        stride: 1
        pool: AVE
    }
}
    """

    #res5c = 
    #res5c = tf.strided_slice(res5c, [0, 0, 0, 0], res5c.get_shape(), [1, 4, 4, 1])

    if final_layer:
        pool5 = ops.atrous_avg_pool(res5c, 7, rate=4, name='pool5', padding='SAME' if convolutional else 'VALID')
        info['activations']['pool5'] = pool5
        ##pool5 = avg_pool(res5c, 7 * 4, stride=1, name='pool5', padding='SAME' if convolutional else 'VALID')
        #pool5 = res5c
        if convolutional:
            z = conv(pool5, 1000, size=1, name='fc1000', activation=None)
        else:
            z = resnet_inner(pool5, 1000, info=info, parameters=parameters, activation=None, name='fc1000')
    else:
        z = res5c

    return z
