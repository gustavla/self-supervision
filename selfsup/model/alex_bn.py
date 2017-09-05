from __future__ import division, print_function, absolute_import
import tensorflow as tf
import functools
import numpy as np
from selfsup import extra
from selfsup.moving_averages import ExponentialMovingAverageExtended
from selfsup.util import DummyDict
from selfsup import ops, caffe

import sys


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])


def init_batch_norm_vars(name, sh):
    bn_mean = tf.get_variable(name + '/bn_mean', shape=sh, trainable=False,
            dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    bn_var = tf.get_variable(name + '/bn_var', shape=sh, trainable=False,
            dtype=tf.float32, initializer=tf.constant_initializer(1.0))
    return bn_mean, bn_var


def batch_norm(z, global_step, phase_test, name, bn_mean=None, bn_var=None):
    mm, vv = extra.moments(z, list(range(z.get_shape().ndims-1)), keep_dims=False, name=name + '_moments')

    beta = 0.0
    gamma = 1.0

    sh = mm.get_shape().as_list()[-1:]
    if bn_mean is None and bn_var is None:
        bn_mean, bn_var = init_batch_norm_vars(name, sh)

    alpha0 = 0.999
    N = 1000
    alpha = tf.to_float(tf.minimum(global_step, N) / N  * alpha0)
    def mean_var_train():
        apply_op_mm = tf.assign(bn_mean, bn_mean * alpha + mm * (1 - alpha))
        apply_op_vv = tf.assign(bn_var, bn_var * alpha + vv * (1 - alpha))

        with tf.control_dependencies([apply_op_mm, apply_op_vv]):
            return tf.identity(mm), tf.identity(vv)
            #return tf.identity(bn_mean), tf.identity(bn_var)

    def mean_var_test():
        return bn_mean, bn_var

    mean, var = tf.cond(tf.logical_not(phase_test),
                        mean_var_train,
                        mean_var_test)

    z = tf.nn.batch_normalization(z, mean, var, beta, gamma, 1e-5)
    return z


def build_network(x, info=DummyDict(), parameters={},
                  phase_test=None, convolutional=False, final_layer=True,
                  activation=tf.nn.relu,
                  well_behaved_size=False,
                  global_step=None,
                  use_lrn=True, prefix='', use_dropout=True):

    # Set up AlexNet
    #conv = functools.partial(alex_conv, size=3, parameters=parameters,
                             #info=info, pre_adjust_batch_norm=pre_adjust_batch_norm)
    #pool = functools.partial(ops.max_pool, info=info)
    if use_dropout:
        dropout = functools.partial(ops.dropout, phase_test=phase_test, info=info)
    else:
        def dropout(x, *args, **kwargs):
            return x

    def add_info(name, z, pre=None, info=DummyDict()):
        info['activations'][name] = z
        if info['config'].get('save_pre'):
            info['activations']['pre:' + name] = pre
        if info.get('scale_summary'):
            with tf.name_scope('activation'):
                tf.summary.scalar('activation/' + name, tf.sqrt(tf.reduce_mean(z**2)))

    if activation is None:
        activation = lambda x: x

    W_init = tf.contrib.layers.xavier_initializer_conv2d()
    W_init_fc = tf.contrib.layers.xavier_initializer()
    b_init = tf.constant_initializer(0.0)

    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4; padding='VALID'
    if convolutional or well_behaved_size:
        padding = 'SAME'
    name = prefix + 'conv1'
    with tf.variable_scope(name):
        sh = [k_h, k_w, x.get_shape().as_list()[3], c_o]
        conv1W = tf.get_variable('weights', sh, dtype=tf.float32,
                            initializer=W_init)
        conv1b = tf.get_variable('biases', [c_o], dtype=tf.float32,
                            initializer=b_init)
    if 'weights' in info:
        info['weights'][name + ':weights'] = conv1W
        info['weights'][name + ':biases'] = conv1b
    info['weights'][name + ':weights'] = conv1W
    info['weights'][name + ':biases'] = conv1b
    conv1 = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding=padding, group=1)
    conv1 = batch_norm(conv1, global_step=global_step, phase_test=phase_test, name=name)
    pre = conv1
    conv1 = activation(conv1)
    add_info(name, conv1, pre=pre, info=info)

    c_o_old = c_o

    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    if use_lrn:
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
        info['activations']['lrn1'] = lrn1
    else:
        lrn1 = conv1


    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    if convolutional or well_behaved_size:
        padding = 'SAME'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    info['activations']['maxpool1'] = maxpool1

    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    #conv2W = tf.Variable(net_data["conv2"][0])
    #conv2b = tf.Variable(net_data["conv2"][1])
    name = prefix + 'conv2'
    with tf.variable_scope(name):
        sh = [k_h, k_w, c_o_old // group, c_o]
        conv2W = tf.get_variable('weights', sh, dtype=tf.float32,
                            initializer=W_init)
        conv2b = tf.get_variable('biases', [c_o], dtype=tf.float32,
                            initializer=b_init)
    if 'weights' in info:
        info['weights'][name + ':weights'] = conv2W
        info['weights'][name + ':biases'] = conv2b

    conv2 = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = batch_norm(conv2, global_step=global_step, phase_test=phase_test, name=name)
    pre = conv2
    conv2 = activation(conv2)
    add_info(name, conv2, pre=pre, info=info)


    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    if use_lrn:
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
    else:
        lrn2 = conv2

    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    if convolutional or well_behaved_size:
        padding = 'SAME'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    info['activations'][prefix+'pool2'] = maxpool2

    c_o_old = c_o

    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    name = prefix + 'conv3'
    with tf.variable_scope(name):
        sh = [k_h, k_w, c_o_old // group, c_o]
        conv3W = tf.get_variable('weights', sh, dtype=tf.float32,
                            initializer=W_init)
        conv3b = tf.get_variable('biases', [c_o], dtype=tf.float32,
                            initializer=b_init)
    if 'weights' in info:
        info['weights'][name + ':weights'] = conv3W
        info['weights'][name + ':biases'] = conv3b
    conv3 = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = batch_norm(conv3, global_step=global_step, phase_test=phase_test, name=name)
    pre = conv3
    conv3 = activation(conv3)
    add_info(name, conv3, pre=pre, info=info)

    c_o_old = c_o

    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    name = prefix + 'conv4'
    with tf.variable_scope(name):
        sh = [k_h, k_w, c_o_old // group, c_o]
        conv4W = tf.get_variable('weights', sh, dtype=tf.float32,
                            initializer=W_init)
        conv4b = tf.get_variable('biases', [c_o], dtype=tf.float32,
                            initializer=b_init)
    if 'weights' in info:
        info['weights'][name + ':weights'] = conv4W
        info['weights'][name + ':biases'] = conv4b
    conv4 = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = batch_norm(conv4, global_step=global_step, phase_test=phase_test, name=name)
    pre = conv4
    conv4 = activation(conv4)
    add_info(name, conv4, pre=pre, info=info)

    c_o_old = c_o

    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    name = prefix + 'conv5'
    with tf.variable_scope(name):
        sh = [k_h, k_w, c_o_old // group, c_o]
        conv5W = tf.get_variable('weights', sh, dtype=tf.float32,
                            initializer=W_init)
        conv5b = tf.get_variable('biases', [c_o], dtype=tf.float32,
                            initializer=b_init)
    if 'weights' in info:
        info['weights'][name + ':weights'] = conv5W
        info['weights'][name + ':biases'] = conv5b
    conv5 = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = batch_norm(conv5, global_step=global_step, phase_test=phase_test, name=name)
    pre = conv5
    conv5 = activation(conv5)
    add_info(name, conv5, pre=pre, info=info)

    #maxpool5
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    if convolutional or well_behaved_size:
        padding = 'SAME'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    info['activations']['pool5'] = maxpool5

    c_o_old = np.prod(maxpool5.get_shape().as_list()[1:])

    channels = maxpool5.get_shape().as_list()[-1]

    info['activations'][prefix+'conv1'] = conv1
    info['activations'][prefix+'conv2'] = conv2
    info['activations'][prefix+'conv3'] = conv3
    info['activations'][prefix+'conv4'] = conv4
    info['activations'][prefix+'conv5'] = conv5

    # Set up weights and biases for fc6/fc7, so that if they are not used, they
    # are still set up (otherwise reuse=True will fail)
    name = prefix + 'fc6'
    with tf.variable_scope(name):
        c_o = 4096
        sh = [6, 6, channels, c_o]
        fc6W = tf.get_variable('weights', sh, dtype=tf.float32,
                            initializer=W_init_fc)
        fc6b = tf.get_variable('biases', [c_o], dtype=tf.float32,
                            initializer=b_init)
    fc6_bn_mean, fc6_bn_var = init_batch_norm_vars(name, [c_o])
    if 'weights' in info:
        info['weights'][name + ':weights'] = fc6W
        info['weights'][name + ':biases'] = fc6b

    name = prefix + 'fc7'
    with tf.variable_scope(name):
        c_old_o = c_o
        c_o = 4096
        sh = [1, 1, c_old_o, c_o]
        fc7W = tf.get_variable('weights', sh, dtype=tf.float32,
                            initializer=W_init_fc)
        fc7b = tf.get_variable('biases', [c_o], dtype=tf.float32,
                            initializer=b_init)
    fc7_bn_mean, fc7_bn_var = init_batch_norm_vars(name, [c_o])
    if 'weights' in info:
        info['weights'][name + ':weights'] = fc7W
        info['weights'][name + ':biases'] = fc7b

    if maxpool5.get_shape().as_list()[1:3] != [6, 6] and not convolutional:
        print('RETURNING PRE-FULLY-CONNECTED')
        return maxpool5

    if convolutional:
        name = prefix + 'fc6'
        #fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
        #fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
        conv6 = tf.nn.conv2d(maxpool5, fc6W, strides=[1, 1, 1, 1], padding='SAME')
        fc6_in = tf.nn.bias_add(conv6, fc6b)
        fc6 = fc6_in
        fc6 = batch_norm(fc6, global_step=global_step, phase_test=phase_test, name=name,
                bn_mean=fc6_bn_mean, bn_var=fc6_bn_var)
        pre = fc6
        fc6 = tf.nn.relu(fc6)
        add_info(name+':nodropout', fc6, pre=fc6_in, info=info)
        fc6 = dropout(fc6, 0.5)
        add_info(name, fc6, pre=pre, info=info)

        c_o_old = c_o

        c_o = 4096

        name = prefix + 'fc7'
        #fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
        conv7 = tf.nn.conv2d(fc6, fc7W, strides=[1, 1, 1, 1], padding='SAME')
        fc7_in = tf.nn.bias_add(conv7, fc7b)
        fc7 = fc7_in
        fc7 = batch_norm(fc7, global_step=global_step, phase_test=phase_test, name=name,
                bn_mean=fc7_bn_mean, bn_var=fc7_bn_var)
        pre = fc7
        fc7 = tf.nn.relu(fc7)
        add_info(name+':nodropout', fc7, pre=fc7_in, info=info)
        fc7 = dropout(fc7, 0.5)
        add_info(name, fc7, pre=pre, info=info)

        c_o_old = c_o

        if final_layer:
            c_o = 1000

            name = prefix + 'fc8'
            with tf.variable_scope(name):
                sh = [1, 1, c_o_old, c_o]
                fc8W = tf.get_variable('weights', sh, dtype=tf.float32,
                                    initializer=W_init_fc)
                fc8b = tf.get_variable('biases', [c_o], dtype=tf.float32,
                                    initializer=b_init)
            if 'weights' in info:
                info['weights'][name + ':weights'] = fc8W
                info['weights'][name + ':biases'] = fc8b
            #fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
            conv8 = tf.nn.conv2d(fc7, fc8W, strides=[1, 1, 1, 1], padding='SAME')
            fc8 = tf.nn.bias_add(conv8, fc8b)
            info['activations'][name] = fc8
        else:
            fc8 = fc7

    else:
        sh_fc = [c_o_old, c_o]
        fc6W = tf.reshape(fc6W, sh_fc)

        name = prefix + 'fc6'
        maxpool5_flat = tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))])
        #fc6 = tf.nn.relu_layer(maxpool5_flat, fc6W, fc6b, name=name)
        fc6_in = tf.nn.bias_add(tf.matmul(maxpool5_flat, fc6W), fc6b)
        fc6 = fc6_in
        fc6 = batch_norm(fc6, global_step=global_step, phase_test=phase_test, name=name,
                bn_mean=fc6_bn_mean, bn_var=fc6_bn_var)
        pre = fc6
        fc6 = tf.nn.relu(fc6, name=name)
        add_info(name+':nodropout', fc6, pre=fc6_in, info=info)
        fc6 = dropout(fc6, 0.5)
        add_info(name, fc6, pre=pre, info=info)

        c_o_old = c_o

        c_o = 4096

        name = prefix + 'fc7'
        fc7W = tf.squeeze(fc7W, [0, 1])

        #fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b, name=name)
        fc7_in = tf.nn.bias_add(tf.matmul(fc6, fc7W), fc7b)
        fc7 = fc7_in
        fc7 = batch_norm(fc7, global_step=global_step, phase_test=phase_test, name=name,
                bn_mean=fc7_bn_mean, bn_var=fc7_bn_var)
        pre = fc7
        fc7 = tf.nn.relu(fc7, name=name)
        add_info(name+':nodropout', fc7, pre=fc7_in, info=info)
        fc7 = dropout(fc7, 0.5)
        add_info(name, fc7, pre=pre, info=info)

        c_o_old = c_o
        c_o = 1000

        if final_layer:
            name = prefix+'fc8'
            with tf.variable_scope(name):
                sh = [c_o_old, c_o]
                fc8W = tf.get_variable('weights', sh, dtype=tf.float32,
                                    initializer=W_init_fc)
                fc8b = tf.get_variable('biases', [c_o], dtype=tf.float32,
                                    initializer=b_init)
            if 'weights' in info:
                info['weights'][name + ':weights'] = fc8W
                info['weights'][name + ':biases'] = fc8b
            fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
            info['activations'][name] = fc8
        else:
            fc8 = fc7

    info['activations'][prefix+'fc6'] = fc6
    info['activations'][prefix+'fc7'] = fc7


    return fc8


LAYERS = [
    'conv1', 'conv2', 'conv3', 'conv4', 'conv5',
    'fc6', 'fc7'
]

CONV_FC_TRANSITIONALS = {
    'fc6': (4096, 256, 6, 6)
}

def save_caffemodel(path, session, prefix='', verbose=True, extra_layers=[]):
    return caffe.save_caffemodel(path, session, LAYERS+extra_layers,
                                 prefix=prefix,
                                 save_batch_norm=True,
                                 #lax_naming=True,
                                 conv_fc_transitionals=CONV_FC_TRANSITIONALS,
                                 verbose=verbose)


def load_caffemodel(path, session, prefix='', ignore=set(), verbose=True):
    return caffe.load_caffemodel(path, session,
                                 prefix=prefix,
                                 ignore=ignore,
                                 conv_fc_transitionals=CONV_FC_TRANSITIONALS,
                                 verbose=verbose)
