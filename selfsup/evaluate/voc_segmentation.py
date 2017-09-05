import sys
import os
import tensorflow as tf
import deepdish as dd
import itertools as itr
import numpy as np
import datetime
import glob
import re
from autocolorize.tensorflow.sparse_extractor import sparse_extractor
from selfsup.util import DummyDict
import time
import functools
import selfsup.model.resnet152_bn
import selfsup.model.vgg16
import selfsup.snapshot
import selfsup.ops
import selfsup.datasets
import selfsup.printing

def now():
    return datetime.datetime.utcnow()

CLASSES = 21
DEV = '/gpu:0'
BATCH_SIZE = 10

SEED = 1234

CROP_SIZE = 448
EDGE_BUFFER = 32
LOCATIONS = 128


PASCAL_TRAIN = os.path.expandvars('$DATA/pascal/train_aug.txt')

PASCAL_90_TRAIN = os.path.expandvars('$DATA/pascal/90_10/train_aug_train.txt')
PASCAL_90_VAL = os.path.expandvars('$DATA/pascal/90_10/train_aug_val.txt')

TEST_EVERY = 10
SNAPSHOT_EVERY = 1000
WINDOW = 5000 // TEST_EVERY
STAGE_MIN_ITER = 0
INIT_MIN_ITER = 35000
STAGES = 3

TIMELIMIT = 3.7


def build_network(rgb01, y, dense=False, input_size=100, batch_size=10, locations=128,
                  edge_buffer=32, classes=CLASSES):
    outputs = []
    activations = {}

    x01 = tf.reduce_mean(rgb01, 3, keep_dims=True)

    phase_test = tf.placeholder(tf.bool, name='phase_test')

    info = selfsup.info.create(scale_summary=True)

    if CAFFE_MODEL is not None:
        resnet = dd.io.load(CAFFE_MODEL, '/data')
    else:
        resnet = {}

    x = x01 - 114.451 / 255
    z = selfsup.model.resnet152_bn.build_network(x, info=info, parameters=resnet,
                                convolutional=True, final_layer=False,
                                phase_test=phase_test, pre_adjust_batch_norm=True)

    conv = functools.partial(selfsup.model.vgg16.vgg_conv, size=3, parameters=resnet,
                             info=info)
    pool = functools.partial(selfsup.ops.max_pool, info=info)
    dropout = functools.partial(selfsup.ops.dropout, phase_test=phase_test, info=info)

    # Add convolutional layers
    z = pool(z, 2, name='pool8')
    z = conv(z, 1024, name='conv8_1')
    z = pool(z, 2, name='pool9')
    z = conv(z, 1024, name='conv9_1')

    # Create hypercolumn
    activations = info['activations']

    LAY = [
        ('conv1', 2.0),
        ('res2a', 4.0),
        ('res2b', 4.0),
        ('res2c', 4.0),
        ('res3b1', 8.0),
        ('res3b4', 8.0),
        ('res3b7', 8.0),
        ('res4b5', 16.0),
        ('res4b10', 16.0),
        ('res4b15', 16.0),
        ('res4b20', 16.0),
        ('res4b25', 16.0),
        ('res4b30', 16.0),
        ('res4b35', 16.0),
        ('res5c', 32.0),
        ('conv8_1', 64.0),
        ('conv9_1', 128.0),
    ]

    if not dense:
        #centroids = tf.placeholder(tf.float32, shape=[batch_size, locations, 2], name='centroids')
        centroids = tf.cast(tf.random_uniform(shape=[batch_size, locations, 2],
                                              minval=edge_buffer,
                                              maxval=input_size - edge_buffer,
                                              name='centroids',
                                              dtype=tf.int32), tf.float32)

        hyper = []

        with tf.name_scope('hypercolumn'):
            for name, scale in LAY:
                sparse_layer = sparse_extractor(centroids, activations[name],
                                                scale, [0.0, 0.0])
                hyper.append(sparse_layer)

            flat_x = tf.concat(hyper, 1, name='concat')
            activations['hypercolumn'] = flat_x
            activations['mean_hypercolumn'] = tf.reduce_mean(flat_x, 0)

        with tf.name_scope('sparse_y'):
            flat_y = tf.squeeze(tf.cast(sparse_extractor(centroids, tf.cast(tf.expand_dims(y, -1), tf.float32), 1.0, [0.0, 0.0]), tf.int32))

        activations['centroids'] = centroids

    else:
        ref_scale_name = 'res2a'
        REF_SCALE = 4
        shape = tuple(np.asarray(activations[ref_scale_name].get_shape()[1:3]).astype(int))

        hyper = []

        with tf.name_scope('hypercolumn'):
            for name, scale in LAY:
                ss = scale / REF_SCALE

                if ss > 1.0:
                    layer = tf.image.resize_bilinear(activations[name], shape)
                elif ss < 1.0:
                    layer = tf.image.resize_nearest_neighbor(activations[name], shape)
                else:
                    layer = activations[name]

                hyper.append(layer)

            hyper_x = tf.concat(hyper, 3, name='concat')
            flat_x = tf.reshape(hyper_x, [-1, int(hyper_x.get_shape()[-1])])

        ss = 1.0 / REF_SCALE
        y4 = tf.expand_dims(y, 3)
        if ss != 1.0:
            layer4 = tf.image.resize_nearest_neighbor(y4, shape)
        else:
            layer4 = y4
        layer = tf.squeeze(layer4, [3])

        flat_y = tf.reshape(layer, [-1])

    flat_h = selfsup.ops.inner(flat_x, 1024, stddev=0.0001, info=info, name='pre_h_fc1')
    flat_h = dropout(flat_h, 0.5, name='h_fc1')

    flat_z = selfsup.ops.inner(flat_h, CLASSES, activation=None, info=info, name='h_fc2')

    if dense:
        img_z = tf.reshape(flat_z, layer.get_shape().as_list() + [-1])

        full_z = tf.image.resize_bilinear(img_z, [input_size, input_size])
        activations['z'] = img_z
        activations['full_z'] = full_z

    #kl_div = tf.reduce_mean(tf.reduce_sum(-y * tf.log(yhat), 1))
    with tf.name_scope('kl_div'):
        kl_div_each = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flat_z, labels=flat_y)
        weight_each = tf.cast(tf.not_equal(flat_y, 255), tf.float32)
        kl_div = tf.reduce_sum(kl_div_each * weight_each) / tf.reduce_sum(weight_each)

        activations['kl_div_each'] = kl_div_each
        activations['weight_each'] = weight_each
        #kl_div = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(flat_z, flat_y))

    with tf.name_scope('weight_decay'):
        wd = 0.0005
        l2_loss = tf.add_n([
            tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name
        ])

        weight_decay = wd * l2_loss

    with tf.name_scope('loss'):
        loss = kl_div + weight_decay

    #pred_entropy = tf.reduce_mean(tf.reduce_sum(-sparse_yhat * tf.log(yhat), 1))

    global_step = tf.Variable(0, trainable=False, name="global_step")

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(flat_y, tf.cast(tf.argmax(flat_z, 1), tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    activations['z'] = z  # TEMP
    activations['y'] = y
    activations['x'] = x
    #activations['yhat'] = yhat
    activations['accuracy'] = accuracy
    activations['global_step'] = global_step
    #activations['entropy'] = pred_entropy
    activations['loss'] = loss
    activations['kl_div'] = kl_div
    activations['phase_test'] = phase_test
    activations['weight_decay'] = weight_decay

    activations['flat_y'] = flat_y
    activations['flat_z'] = flat_z

    info['layer_info'] = LAY

    with tf.name_scope('summaries'):
        tf.summary.scalar('loss', kl_div)
        tf.summary.scalar('weight_decay', weight_decay)
        tf.summary.scalar('objective', loss)
        tf.summary.scalar('accuracy', accuracy)

    return activations, info


def train(model_filename, output_dir, device='/gpu:0', time_limit=None,
          steps=None):
    rs = np.random.RandomState(0)
    start = time.time()

    for path in ['img', 'snapshots']:
        os.makedirs(path, exist_ok=True)

    with tf.device('/cpu'):
        pick_val = tf.placeholder(tf.bool, name='pick_val')

        tr_x, tr_y, tr_imgshape, tr_imgname = selfsup.datasets.voc_seg_batching(PASCAL_TRAIN,
                                                                           batch_size=BATCH_SIZE,
                                                                           input_size=CROP_SIZE,
                                                                           seed=SEED,
                                                                           random_mirror=True)

    with tf.device(DEV):
        x = tr_x
        y = tr_y
        imgshape = tr_imgshape
        imgname = tr_imgname
        variables, info = build_network(x, y, dense=False,
                                        input_size=CROP_SIZE,
                                        batch_size=BATCH_SIZE,
                                        locations=LOCATIONS,
                                        edge_buffer=EDGE_BUFFER)


        selfsup.printing.print_init(info)

        variables['pick_val'] = pick_val

    lr0 = 0.001
    #epoch = len(imgs) / BATCH_SIZE
    #steps = list((np.array([250, 300, 350, 400]) * epoch).astype(np.int32))
    variables['global_stage'] = tf.Variable(0, trainable=False, dtype=tf.int32, name="global_stage")
    variables['global_stage_start'] = tf.Variable(0, trainable=False, dtype=tf.int32, name="global_stage_start")
    if False:  # Adaptive
        lr = lr0 * 10 ** tf.cast(-variables['global_stage'], tf.float32)
        max_iter = None
    else:
        steps = [22500, 30000, 40000]
        print('steps', steps)
        max_iter = steps[-1]
        lrs = list(lr0 * 10**-np.arange(len(steps) + 1, dtype=np.float32))
        lr = tf.train.piecewise_constant(variables['global_step'], steps, lrs)

    variables['learning_rate'] = lr
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('global_stage', variables['global_stage'])
    tf.summary.scalar('global_stage_start', variables['global_stage_start'])
    train_step = tf.train.MomentumOptimizer(lr, 0.9).minimize(variables['loss'],
            global_step=variables['global_step'])

    variables['window_sum'] = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='window_sum')
    variables['window_last_average'] = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='window_last_average')
    variables['window_size'] = tf.Variable(0, trainable=False, dtype=tf.int32, name='window_pointer')

    #variables['last_loss'] = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='last_loss')
    #variables['lossdiff0'] = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='lossdiff0')
    #variables['lossdiff1'] = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='lossdiff1')
    #tf.summary.scalar('lossdiff1', variables['lossdiff1'])

    #ema = tf.train.ExponentialMovingAverage(decay=0.9)

    done = False
    with tf.Session(config=selfsup.config()) as sess:
        with tf.device(DEV):
            saver = tf.train.Saver()

            merged = tf.summary.merge_all()
            tr_sw = tf.summary.FileWriter(os.path.join('tf-log', 'train_train'), sess.graph)
            vl_sw = tf.summary.FileWriter(os.path.join('tf-log', 'train_val'), sess.graph)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            snapshot_fn = selfsup.snapshot.find()
            if snapshot_fn is not None:
                selfsup.snapshot.restore(saver, sess, None, snapshot_fn)
                start_iter = variables['global_step'].eval()
            else:
                sess.run(tf.initialize_all_variables())
                start_iter = 0


            time_up = False
            while True:
                iteration = variables['global_step'].eval()

                if TIMELIMIT is not None:
                    time_up = time.time() - start > 3600 * TIMELIMIT

                if iteration % TEST_EVERY == 0:
                    # Train loss
                    val = False
                    inputs_val = {variables['pick_val']: val,
                                  variables['phase_test']: val}

                    summary, kl_div, mean_hypercolumn, loss_wd, lri, flat_z = sess.run(
                            [merged, variables['kl_div'],
                             variables['mean_hypercolumn'],
                             variables['weight_decay'],
                             variables['learning_rate'],
                             variables['flat_z']], feed_dict=inputs_val)

                    tr_sw.add_summary(summary, iteration)

                    # Does it make non-background predictions?
                    #flat_z = variables['flat_z'].eval(feed_dict=inputs)
                    print('Max label:', flat_z.argmax(-1).max())
                    print(now(), 'Train Iteration', iteration, 'Primary Loss', kl_div, 'lr =', lri)

                if False and iteration % TEST_EVERY == 0:
                    val = True
                    inputs_val = {variables['pick_val']: val,
                                  variables['phase_test']: val}

                    summary, kl_div, accuracy, lri, flat_z, wsize, stage_start = sess.run([
                            merged,
                            variables['kl_div'],
                            variables['accuracy'],
                            variables['learning_rate'],
                            variables['flat_z'],
                            variables['window_size'],
                            variables['global_stage_start'],
                        ], feed_dict=inputs_val)

                    vl_sw.add_summary(summary, iteration)

                    sess.run(variables['window_sum'].assign_add(accuracy))

                    if wsize == WINDOW:
                        win_sum, last_avg = sess.run([variables['window_sum'], variables['window_last_average']])
                        avg = win_sum / WINDOW
                        sess.run(variables['window_last_average'].assign(avg))
                        print('%8d AVG %8.4f LAST %8.4f [%s] lr=%7.5f' % (iteration, avg, last_avg, 'x' if avg < last_avg else ' ', lri))
                        sess.run(variables['window_sum'].assign(0.0))
                        sess.run(variables['window_size'].assign(0))

                        if (last_avg != 0.0 and avg <= last_avg and 
                                iteration - stage_start > STAGE_MIN_ITER and
                                iteration >= INIT_MIN_ITER):
                            print("NEW STAGE")
                            sess.run(variables['global_stage'].assign_add(1))
                            sess.run(variables['global_stage_start'].assign(iteration))
                            if variables['global_stage'].eval() == STAGES:
                                snap_fn = selfsup.snapshot.save(saver, sess, variables, None)
                                print('Saved snapshot to', snap_fn)
                                done = True
                                break

                    vl_sw.add_summary(summary, iteration)
                    sess.run(variables['window_size'].assign_add(1))

                    # Does it make non-background predictions?
                    #flat_z = variables['flat_z'].eval(feed_dict=inputs)
                    print('Max label:', flat_z.argmax(-1).max())
                    print(now(), 'Val   Iteration', iteration, 'Primary Loss', kl_div, 'lr =', lri)

                # Snapshot
                if (iteration % SNAPSHOT_EVERY == 0 and iteration > start_iter or
                        iteration == max_iter or time_up):
                    # Save checkpoint
                    snap_fn = selfsup.snapshot.save(saver, sess, variables, None)
                    print('Saved snapshot to', snap_fn)

                # Train step
                with dd.timed():
                    inputs = {variables['pick_val']: False,
                              variables['phase_test']: False}
                    train_step.run(feed_dict=inputs)

                if iteration == max_iter or time_up:
                    if iteration == max_iter:
                        done = True
                    snap_fn = selfsup.snapshot.save(saver, sess, variables, None)
                    print('Saved snapshot to', snap_fn)
                    break
            if done:
                with open('0-done', 'w') as f:
                    print('done', file=f)

            coord.request_stop()
            coord.join(threads)
    del sess


def train_and_test(*args, **kwargs):
    train(*args, **kwargs)
    tf.reset_default_graph()
    test(*args, **kwargs)


if __name__ == '__main__':
    if os.path.exists('0-done'):
        print('Done already')
        sys.exit(0)
    main()
