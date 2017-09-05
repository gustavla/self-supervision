from __future__ import division, print_function, absolute_import
from . import util
from . import ops
from .util import DummyDict
import tensorflow as tf
import functools
import numpy as np
import struct
import sys
import deepdish as dd
from .caffe import from_caffe, to_caffe

def load_permutations(fn):
    with open(fn, 'rb') as f:
        N, P = struct.unpack('<II', f.read(8))
        perms = np.zeros((N, P), dtype=np.int32)
        for i in range(N):
            perm = struct.unpack('<IIIIIIIII', f.read(4*P))
            assert len(set(perm)) == P, perm
            perms[i] = perm

    return perms

def _pretrained_alex_conv_weights_initializer(name, data, info=None, pre_adjust_batch_norm=False, prefix=''):
    shape = None
    if name in data and '0' in data[name]:
        W = data[name]['0'].copy()
        if W.ndim == 2 and name == 'fc6':
            #assert 0, "6x6 or 7x7? Confirm."
            W = W.reshape((W.shape[0], 256, 6, 6))
        elif W.ndim == 2 and name == 'fc7':
            W = W.reshape((W.shape[0], -1, 1, 1))
        elif W.ndim == 2 and name == 'fc8':
            W = W.reshape((W.shape[0], -1, 1, 1))
        W = W.transpose(2, 3, 1, 0)
        init_type = 'file'
        if name == 'conv1' and W.shape[2] == 3:
            W = W[:, :, ::-1]
            init_type += ':bgr-flipped'
        bn_name = 'batch_' + name
        if pre_adjust_batch_norm and bn_name in data:
            bn_data = data[bn_name]
            sigma = np.sqrt(1e-5 + bn_data['1'] / bn_data['2'])
            W /= sigma
            init_type += ':batch-adjusted'
        init = tf.constant_initializer(W)
        shape = W.shape
    else:
        init_type = 'init'
        init = tf.contrib.layers.variance_scaling_initializer()
    if info is not None:
        info[prefix + ':' + name + '/weights'] = init_type
    return init, shape


def _pretrained_alex_inner_weights_initializer(name, data, info=DummyDict(), pre_adjust_batch_norm=False, prefix=''):
    shape = None
    if name in data and '0' in data[name]:
        W = data[name]['0']
        if W.ndim != 2:
            W = W.transpose(2, 3, 1, 0)
            W = W.reshape((-1, W.shape[-1]))
        else:
            # We have to fix the permutation
            if name == 'fc6':
                W = W.reshape(W.shape[0], 256, 6, 6).transpose(0, 2, 3, 1).reshape(4096, -1).T
                #W = W.reshape(W.shape[0], 256, 6, 6).transpose(2, 3, 1, 0).reshape(-1, 4096)
            else:
                W = W.T
        init_type = 'file'
        bn_name = 'batch_' + name
        if pre_adjust_batch_norm and bn_name in data:
            bn_data = data[bn_name]
            sigma = np.sqrt(1e-5 + bn_data['1'] / bn_data['2'])
            W /= sigma
            init_type += ':batch-adjusted'
        init = tf.constant_initializer(W.copy())
        shape = W.shape
    else:
        init_type = 'init'
        init = tf.contrib.layers.variance_scaling_initializer()
    info[prefix + ':' + name + '/weights'] = init_type
    return init, shape


def _pretrained_alex_biases_initializer(name, data, info=DummyDict(), pre_adjust_batch_norm=False, prefix=''):
    shape = None
    if name in data and '1' in data[name]:
        init_type = 'file'
        bias = data[name]['1'].copy()
        bn_name = 'batch_' + name
        if pre_adjust_batch_norm and bn_name in data:
            bn_data = data[bn_name]
            sigma = np.sqrt(1e-5 + bn_data['1'] / bn_data['2'])
            mu = bn_data['0'] / bn_data['2']
            bias = (bias - mu) / sigma
            init_type += ':batch-adjusted'
        init = tf.constant_initializer(bias)
        shape = bias.shape
    else:
        init_type = 'init'
        init = tf.constant_initializer(0.0)
    info[prefix + ':' + name + '/biases'] = init_type
    return init, shape


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1, name=''):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k, g: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding, name=name + f'_g{g}')


    if group==1:
        conv = convolve(input, kernel, 0)
    else:
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k, g) for g, (i, k) in enumerate(zip(input_groups, kernel_groups))]
        conv = tf.concat(output_groups, 3)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])


def make_patches(x, patch_size):
    batch_size = x.get_shape().as_list()[0]
    crop_size = x.get_shape().as_list()[1]

    WINDOW_SIZE = crop_size // 3

    N = x.get_shape().as_list()[0]
    C = x.get_shape().as_list()[3]

    patches = []

    for i, j in dd.multi_range(3, 3):
        #tf.slice(x, [

        M = WINDOW_SIZE - patch_size + 1
        HALF = M // 2
        limit = np.array([1, M, M, 1])
        offset = np.array([0, i * WINDOW_SIZE + HALF, j * WINDOW_SIZE + HALF, 0])
        patch = tf.slice(x, offset, [N, patch_size, patch_size, C])
        patches.append(patch)

    patches1 = tf.stack(patches, axis=1)

    return tf.reshape(patches1, [-1, patch_size, patch_size, C])
    #return patches2, -1


def make_random_patches(x, y, patch_size, permutations):
    batch_size = x.get_shape().as_list()[0]
    crop_size = x.get_shape().as_list()[1]

    perm_idx = tf.expand_dims(y, 1)
    perm = tf.gather_nd(permutations, perm_idx)


    WINDOW_SIZE = crop_size // 3

    N = x.get_shape().as_list()[0]
    C = x.get_shape().as_list()[3]

    patches = []

    for i, j in dd.multi_range(3, 3):
        #tf.slice(x, [

        M = WINDOW_SIZE - patch_size + 1
        limit = np.array([1, M, M, 1])
        offset = np.array([0, i * WINDOW_SIZE, j * WINDOW_SIZE, 0]) + tf.random_uniform(
                [4], dtype=tf.int32,
                maxval=M,
                ) % limit
        patch = tf.slice(x, offset, [N, patch_size, patch_size, C])
        patches.append(patch)

    patches1 = tf.stack(patches, axis=1)
    perm0 = tf.reshape(np.arange(batch_size)[:, np.newaxis] * 9 + (perm - 1), [-1])

    patches_flat = tf.reshape(patches1, [-1] + patches1.get_shape().as_list()[2:])
    patches2 = tf.gather(patches_flat, perm0)

    #return tf.reshape(patches2, [-1, PATCH_SIZE, PATCH_SIZE, C])
    return patches2


def build_network(raw_x, y, info=DummyDict(), parameters={}, permutations=None, hole=1,
                  patch_size=None,
                  phase_test=None, convolutional=False, final_layer=True,
                  activation=tf.nn.relu,
                  pre_adjust_batch_norm=False, well_behaved_size=False,
                  use_lrn=True, prefix='', use_dropout=True):

    batch_size = raw_x.get_shape().as_list()[0]
    if permutations is not None:
        x = make_random_patches(raw_x, y, permutations=permutations,
                patch_size=patch_size)
    else:
        x = make_patches(raw_x, patch_size=patch_size)

    info['activations']['patches'] = x
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
            #with tf.name_scope('activation'):
            tf.summary.scalar('activation/' + name, tf.sqrt(tf.reduce_mean(z**2)))

    if activation is None:
        activation = lambda x: x

    W_init = tf.contrib.layers.variance_scaling_initializer()
    b_init = tf.constant_initializer(0.0)

    k_h = 11; k_w = 11; c_o = 96; s_h = 2; s_w = 2; padding='VALID'
    if convolutional or well_behaved_size:
        padding = 'SAME'
    #conv1W = tf.Variable(net_data["conv1"][0])
    #conv1b = tf.Variable(net_data["conv1"][1])
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
        conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding=padding, group=1, name=name)
        conv1 = activation(conv1_in)
        add_info(name, conv1, pre=conv1_in, info=info)

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

        conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group, name=name)
        conv2 = activation(conv2_in)
        add_info(name, conv2, pre=conv2_in, info=info)


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

    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    #conv3W = tf.Variable(net_data["conv3"][0])
    #conv3b = tf.Variable(net_data["conv3"][1])
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
        conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group, name=name)
        conv3 = activation(conv3_in)
        add_info(name, conv3, pre=conv3_in, info=info)

    c_o_old = c_o

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    #conv4W = tf.Variable(net_data["conv4"][0])
    #conv4b = tf.Variable(net_data["conv4"][1])
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
        conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group, name=name)
        conv4 = activation(conv4_in)
        add_info(name, conv4, pre=conv4_in, info=info)

    c_o_old = c_o

    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    #conv5W = tf.Variable(net_data["conv5"][0])
    #conv5b = tf.Variable(net_data["conv5"][1])
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
        conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group, name=name)
        conv5 = activation(conv5_in)
        add_info(name, conv5, pre=conv5_in, info=info)

    #maxpool5
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'SAME'
    if convolutional or well_behaved_size:
        padding = 'SAME'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    info['activations']['pool5'] = maxpool5

    c_o_old = np.prod(maxpool5.get_shape().as_list()[1:])
    c_o = 1024

    name = prefix + 'fc6'
    with tf.variable_scope(name):
        sh = [c_o_old, c_o]
        fc6W = tf.get_variable('weights', sh, dtype=tf.float32,
                            initializer=W_init)
        fc6b = tf.get_variable('biases', [c_o], dtype=tf.float32,
                            initializer=b_init)
        if 'weights' in info:
            info['weights'][name + ':weights'] = fc6W
            info['weights'][name + ':biases'] = fc6b
        maxpool5_flat = tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))])
        #fc6 = tf.nn.relu_layer(maxpool5_flat, fc6W, fc6b, name=name)
        mini_fc6_in = tf.nn.bias_add(tf.matmul(maxpool5_flat, fc6W), fc6b)
        mini_fc6 = tf.nn.relu(mini_fc6_in, name=name)
        add_info(name+':nodropout', mini_fc6, pre=mini_fc6_in, info=info)
        mini_fc6 = dropout(mini_fc6, 0.5)
        add_info(name, mini_fc6, pre=mini_fc6_in, info=info)

        batch_fc6 = tf.reshape(mini_fc6, [batch_size, -1, c_o])
        fc6 = tf.concat(tf.unstack(batch_fc6, axis=1), 1)

    c_o_old = fc6.get_shape().as_list()[1]

    c_o = 4096

    name = prefix + 'fc7'
    with tf.variable_scope(name):
        fc7W = tf.get_variable('weights', [c_o_old, c_o], dtype=tf.float32,
                            initializer=W_init)
        fc7b = tf.get_variable('biases', [c_o], dtype=tf.float32,
                            initializer=b_init)
        if 'weights' in info:
            info['weights'][name + ':weights'] = fc7W
            info['weights'][name + ':biases'] = fc7b
        #fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b, name=name)
        fc7_in = tf.nn.bias_add(tf.matmul(fc6, fc7W), fc7b)
        fc7 = tf.nn.relu(fc7_in, name=name)
        add_info(name+':nodropout', fc7, pre=fc7_in, info=info)
        fc7 = dropout(fc7, 0.5)
        add_info(name, fc7, pre=fc7_in, info=info)

    c_o_old = c_o
    c_o = 1000

    if final_layer:
        W_init, W_shape = _pretrained_alex_inner_weights_initializer('fc8', parameters,
                                                          info=info.get('init'),
                                                          pre_adjust_batch_norm=pre_adjust_batch_norm,
                                                          prefix=prefix)
        b_init, b_shape = _pretrained_alex_biases_initializer('fc8', parameters,
                                                    info=info.get('init'),
                                                    pre_adjust_batch_norm=pre_adjust_batch_norm,
                                                    prefix=prefix)
        #fc8
        #fc(1000, relu=False, name='fc8')
        #fc8W = tf.Variable(net_data["fc8"][0])
        #fc8b = tf.Variable(net_data["fc8"][1])
        name = prefix+'fc8'
        with tf.variable_scope(name):
            sh = [c_o_old, c_o]
            assert W_shape is None or tuple(sh) == tuple(W_shape), (sh, W_shape)
            fc8W = tf.get_variable('weights', sh, dtype=tf.float32,
                                initializer=W_init)
            fc8b = tf.get_variable('biases', [c_o], dtype=tf.float32,
                                initializer=b_init)
            if 'weights' in info:
                info['weights'][name + ':weights'] = fc8W
                info['weights'][name + ':biases'] = fc8b
            fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
            info['activations'][name] = fc8
    else:
        fc8 = fc7

    info['activations'][prefix+'conv1'] = conv1
    info['activations'][prefix+'conv2'] = conv2
    info['activations'][prefix+'conv3'] = conv3
    info['activations'][prefix+'conv4'] = conv4
    info['activations'][prefix+'conv5'] = conv5
    info['activations'][prefix+'fc6'] = fc6
    info['activations'][prefix+'fc7'] = fc7


    return fc8


def build_regular_network(x, info=DummyDict(),
                  phase_test=None, convolutional=False,
                  activation=tf.nn.relu,
                  well_behaved_size=False,
                  use_lrn=True, prefix='', use_dropout=True):

    batch_size = x.get_shape().as_list()[0]

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
            #with tf.name_scope('activation'):
            tf.summary.scalar('activation/' + name, tf.sqrt(tf.reduce_mean(z**2)))

    if activation is None:
        activation = lambda x: x

    W_init = tf.contrib.layers.variance_scaling_initializer()
    b_init = tf.constant_initializer(0.0)

    k_h = 11; k_w = 11; c_o = 96; s_h = 2; s_w = 2; padding='VALID'
    if convolutional or well_behaved_size:
        padding = 'SAME'
    #conv1W = tf.Variable(net_data["conv1"][0])
    #conv1b = tf.Variable(net_data["conv1"][1])
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
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding=padding, group=1, name=name)
    conv1 = activation(conv1_in)
    add_info(name, conv1, pre=conv1_in, info=info)

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


    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    if convolutional or well_behaved_size:
        padding = 'SAME'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    info['activations']['maxpool1'] = maxpool1

    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
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

    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group, name=name)
    conv2 = activation(conv2_in)
    add_info(name, conv2, pre=conv2_in, info=info)


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

    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    #conv3W = tf.Variable(net_data["conv3"][0])
    #conv3b = tf.Variable(net_data["conv3"][1])
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
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group, name=name)
    conv3 = activation(conv3_in)
    add_info(name, conv3, pre=conv3_in, info=info)

    c_o_old = c_o

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    #conv4W = tf.Variable(net_data["conv4"][0])
    #conv4b = tf.Variable(net_data["conv4"][1])
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
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group, name=name)
    conv4 = activation(conv4_in)
    add_info(name, conv4, pre=conv4_in, info=info)

    c_o_old = c_o

    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    #conv5W = tf.Variable(net_data["conv5"][0])
    #conv5b = tf.Variable(net_data["conv5"][1])
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
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=group, name=name)
    conv5 = activation(conv5_in)
    add_info(name, conv5, pre=conv5_in, info=info)

    #maxpool5
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 5; k_w = 5; s_h = 4; s_w = 4; padding = 'VALID'
    if convolutional or well_behaved_size:
        padding = 'SAME'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    info['activations']['pool5'] = maxpool5


    info['activations'][prefix+'conv1'] = conv1
    info['activations'][prefix+'conv2'] = conv2
    info['activations'][prefix+'conv3'] = conv3
    info['activations'][prefix+'conv4'] = conv4
    info['activations'][prefix+'conv5'] = conv5
    return maxpool5


def build_noroozi_network(x, info=DummyDict(),
                  phase_test=None, convolutional=False,
                  activation=tf.nn.relu,
                  well_behaved_size=False,
                  use_lrn=True, prefix='', use_dropout=True,
                  final_layer=False):

    batch_size = x.get_shape().as_list()[0]

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
            #with tf.name_scope('activation'):
            tf.summary.scalar('activation/' + name, tf.sqrt(tf.reduce_mean(z**2)))

    if activation is None:
        activation = lambda x: x

    W_init = tf.contrib.layers.variance_scaling_initializer()
    b_init = tf.constant_initializer(0.0)

    k_h = 11; k_w = 11; c_o = 96; s_h = 2; s_w = 2; padding='VALID'
    if convolutional or well_behaved_size:
        padding = 'SAME'
    #conv1W = tf.Variable(net_data["conv1"][0])
    #conv1b = tf.Variable(net_data["conv1"][1])
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
    #x = tf.pad(x, [[0, 0], [0, 1], [0, 1], [0, 0]])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding=padding, group=1, name=name)
    conv1 = activation(conv1_in)
    add_info(name, conv1, pre=conv1_in, info=info)

    c_o_old = c_o


    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    if convolutional or well_behaved_size:
        padding = 'SAME'
    maxpool1 = tf.nn.max_pool(conv1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    info['activations']['maxpool1'] = maxpool1


    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    if use_lrn:
        lrn1 = tf.nn.local_response_normalization(maxpool1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
        info['activations']['lrn1'] = lrn1
    else:
        lrn1 = maxpool1

    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
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

    conv2_in = conv(lrn1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group, name=name)
    conv2 = activation(conv2_in)
    add_info(name, conv2, pre=conv2_in, info=info)


    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    if convolutional or well_behaved_size:
        padding = 'SAME'
    lrn2 = tf.pad(conv2, [[0, 0], [0, 1], [0, 1], [0, 0]])
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    info['activations'][prefix+'pool2'] = maxpool2

    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    if use_lrn:
        lrn2 = tf.nn.local_response_normalization(maxpool2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
    else:
        lrn2 = maxpool2

    c_o_old = c_o

    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    #conv3W = tf.Variable(net_data["conv3"][0])
    #conv3b = tf.Variable(net_data["conv3"][1])
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
    conv3_in = conv(lrn2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group, name=name)
    conv3 = activation(conv3_in)
    add_info(name, conv3, pre=conv3_in, info=info)

    c_o_old = c_o

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    #conv4W = tf.Variable(net_data["conv4"][0])
    #conv4b = tf.Variable(net_data["conv4"][1])
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
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group, name=name)
    conv4 = activation(conv4_in)
    add_info(name, conv4, pre=conv4_in, info=info)

    c_o_old = c_o

    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    #conv5W = tf.Variable(net_data["conv5"][0])
    #conv5b = tf.Variable(net_data["conv5"][1])
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
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=group, name=name)
    conv5 = activation(conv5_in)
    add_info(name, conv5, pre=conv5_in, info=info)

    #maxpool5
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 5; k_w = 5; s_h = 4; s_w = 4; padding = 'VALID'
    if convolutional or well_behaved_size:
        padding = 'SAME'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    info['activations']['pool5'] = maxpool5

    c_o_old = np.prod(maxpool5.get_shape().as_list()[1:])

    c_o = 4096

    if convolutional:
        c_o_old = maxpool5.get_shape().as_list()[-1]

        name = prefix + 'fc6'
        with tf.variable_scope(name):
            sh = [6, 6, c_o_old, c_o]
            fc6W = tf.get_variable('weights', sh, dtype=tf.float32,
                                initializer=W_init)
            fc6b = tf.get_variable('biases', [c_o], dtype=tf.float32,
                                initializer=b_init)
            if 'weights' in info:
                info['weights'][name + ':weights'] = fc6W
                info['weights'][name + ':biases'] = fc6b
            #fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
            #fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
            conv6 = tf.nn.conv2d(maxpool5, fc6W, strides=[1, 1, 1, 1], padding='SAME')
            fc6_in = tf.nn.bias_add(conv6, fc6b)
            fc6 = tf.nn.relu(fc6_in)
            add_info(name+':nodropout', fc6, pre=fc6_in, info=info)
            fc6 = dropout(fc6, 0.5)
            add_info(name, fc6, pre=fc6_in, info=info)

        c_o_old = c_o

        c_o = 4096

        #fc7
        #fc(4096, name='fc7')
        #fc7W = tf.Variable(net_data["fc7"][0])
        #fc7b = tf.Variable(net_data["fc7"][1])
        name = prefix + 'fc7'
        with tf.variable_scope(name):
            sh = [1, 1, c_o_old, c_o]
            fc7W = tf.get_variable('weights', sh, dtype=tf.float32,
                                initializer=W_init)
            fc7b = tf.get_variable('biases', [c_o], dtype=tf.float32,
                                initializer=b_init)
            if 'weights' in info:
                info['weights'][name + ':weights'] = fc7W
                info['weights'][name + ':biases'] = fc7b
            #fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
            conv7 = tf.nn.conv2d(fc6, fc7W, strides=[1, 1, 1, 1], padding='SAME')
            fc7_in = tf.nn.bias_add(conv7, fc7b)
            fc7 = tf.nn.relu(fc7_in)
            add_info(name+':nodropout', fc7, pre=fc7_in, info=info)
            fc7 = dropout(fc7, 0.5)
            add_info(name, fc7, pre=fc7_in, info=info)

        c_o_old = c_o

        if final_layer:
            c_o = 1000

            name = prefix + 'fc8'
            with tf.variable_scope(name):
                sh = [1, 1, c_o_old, c_o]
                fc8W = tf.get_variable('weights', sh, dtype=tf.float32,
                                    initializer=W_init)
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
        name = prefix + 'fc6'
        with tf.variable_scope(name):
            sh = [c_o_old, c_o]
            fc6W = tf.get_variable('weights', sh, dtype=tf.float32,
                                initializer=W_init)
            fc6b = tf.get_variable('biases', [c_o], dtype=tf.float32,
                                initializer=b_init)
            if 'weights' in info:
                info['weights'][name + ':weights'] = fc6W
                info['weights'][name + ':biases'] = fc6b
            maxpool5_flat = tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))])
            #fc6 = tf.nn.relu_layer(maxpool5_flat, fc6W, fc6b, name=name)
            fc6_in = tf.nn.bias_add(tf.matmul(maxpool5_flat, fc6W), fc6b)
            fc6 = tf.nn.relu(fc6_in, name=name)
            add_info(name+':nodropout', fc6, pre=fc6_in, info=info)
            fc6 = dropout(fc6, 0.5)
            add_info(name, fc6, pre=fc6_in, info=info)

        c_o_old = c_o

        c_o = 4096

        name = prefix + 'fc7'
        with tf.variable_scope(name):
            fc7W = tf.get_variable('weights', [c_o_old, c_o], dtype=tf.float32,
                                initializer=W_init)
            fc7b = tf.get_variable('biases', [c_o], dtype=tf.float32,
                                initializer=b_init)
            if 'weights' in info:
                info['weights'][name + ':weights'] = fc7W
                info['weights'][name + ':biases'] = fc7b
            #fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b, name=name)
            fc7_in = tf.nn.bias_add(tf.matmul(fc6, fc7W), fc7b)
            fc7 = tf.nn.relu(fc7_in, name=name)
            add_info(name+':nodropout', fc7, pre=fc7_in, info=info)
            fc7 = dropout(fc7, 0.5)
            add_info(name, fc7, pre=fc7_in, info=info)

        c_o_old = c_o
        c_o = 1000

        if final_layer:
            name = prefix+'fc8'
            with tf.variable_scope(name):
                sh = [c_o_old, c_o]
                fc8W = tf.get_variable('weights', sh, dtype=tf.float32,
                                    initializer=W_init)
                fc8b = tf.get_variable('biases', [c_o], dtype=tf.float32,
                                    initializer=b_init)
            if 'weights' in info:
                info['weights'][name + ':weights'] = fc8W
                info['weights'][name + ':biases'] = fc8b
            fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
            info['activations'][name] = fc8
        else:
            fc8 = fc7

    info['activations'][prefix+'conv1'] = conv1
    info['activations'][prefix+'conv2'] = conv2
    info['activations'][prefix+'conv3'] = conv3
    info['activations'][prefix+'conv4'] = conv4
    info['activations'][prefix+'conv5'] = conv5
    #return maxpool5
    return fc8


def load_caffemodel(path, session, info=DummyDict(), prefix='', ignore=set()):
    def find_weights(name, which='weights'):
        for tw in tf.trainable_variables():
            if tw.name.split(':')[0] == name + '/' + which:
                return tw
        return None

    data = dd.io.load(path, '/data')

    conv_fc_transitionals = {'fc6_s1': (1024, 256, 3, 3)}

    assigns = []
    loaded = []
    info = {}
    for key in data:
        local_key = prefix + key.replace('_s1', '')

        if local_key not in ignore:
            if '0' in data[key]:
                weights = find_weights(local_key, 'weights')
                if weights is not None:
                    W = from_caffe(data[key]['0'], color_layer='conv1_s1', name=key, conv_fc_transitionals=conv_fc_transitionals, info=info)
                    assigns.append(weights.assign(W))
                    loaded.append('{}:0 -> {}:weights {}'.format(key, local_key, info.get(key, '')))
            if '1' in data[key]:
                biases = find_weights(local_key, 'biases')
                if biases is not None:
                    assigns.append(biases.assign(data[key]['1']))
                    loaded.append('{}:1 -> {}:biases'.format(key, local_key))

    session.run(assigns)
    return loaded
