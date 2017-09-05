import sys
import os
import tensorflow as tf
import deepdish as dd
import itertools as itr
import numpy as np
import datetime
import glob
import re
import ipdb
import datetime
from selfsup.util import DummyDict
import gl
import time
import functools
from sklearn import metrics
import selfsup
import selfsup.info
import selfsup.model.alex

BATCH_SIZE = 16  # TODO: NOTE!
TEST_BATCH_SIZE = 10

CLASSES = 20
SEED = 1234

MIN_SCALE = 0.5
MAX_SCALE = 2.0

CROPS = 10

IMAGENET_TRAIN = os.path.expandvars('$LARSSON/ImageNet/imagenet_train.txt')
IMAGENET_A_TRAIN = os.path.expandvars('$LARSSON/ImageNet/100k_100k/imagenet_tr.txt')
IMAGENET_A_VAL = os.path.expandvars('$LARSSON/ImageNet/100k_100k/imagenet_vl.txt')

TEST_EVERY = 100
SNAPSHOT_EVERY = 10000
#WINDOW = 2500
WINDOW = 1000
#WINDOW = 100 // TEST_EVERY
STAGE_MIN_ITER = 0

TIMELIMIT = 3.7
if 'forever' in sys.argv:
    TIMELIMIT = 300.0


def special_loss(logits, labels):
    """
    This loss (and the rest of the training procedure) was taken from Philipp
    Kraehenbuehl's code.
    """
    mask = labels != 255
    labels1 = tf.clip_by_value(labels, 0, 1)
    lz = tf.nn.softplus(-tf.abs(logits)) * mask
    return tf.reduce_sum(lz + (tf.to_float(logits > 0) - labels1) * logits * mask, 1) 


def build_network(raw_x, y, model_filename=None, network_type='alex-lrn'):
    outputs = []
    activations = {}

    phase_test = tf.placeholder(tf.bool, name='phase_test')

    info = selfsup.info.create(scale_summary=True)

    x = raw_x - 114.451/255

    # Scale and subtract mean
    if model_filename is not None:
        data = dd.io.load(model_filename, '/data')
    else:
        data = {}

    if network_type in ['alex', 'alex-lrn']:
        use_lrn = network_type == 'alex-lrn'
        print('USE_LRN', use_lrn)
        z = selfsup.model.alex.build_network(x, info=info, parameters=data,
                                 final_layer=False,
                                 phase_test=phase_test,
                                 pre_adjust_batch_norm=True,
                                 use_lrn=use_lrn,
                                 use_dropout=True,
                                 well_behaved_size=False)
    elif network_type == 'vgg16':
        z = selfsup.model.vgg16.build_network(x, info=info, parameters=data,
                                 final_layer=False,
                                 phase_test=phase_test,
                                 pre_adjust_batch_norm=True,
                                 use_dropout=True)
    else:
        raise ValueError('Unsupported network type')

    #z = info['activations']['conv5']
    z = selfsup.model.vgg16.vgg_inner(z, CLASSES, info=info, activation=None, name='task')

    # Create hypercolumn
    activations = info['activations']

    with tf.name_scope('cross_entropy'):
        #cross_entropy_each = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)
        cross_entropy_each = special_loss(logits=z, labels=y)
        cross_entropy = tf.reduce_mean(cross_entropy_each)

    with tf.name_scope('weight_decay'):
        wd = 1e-6
        l2_loss = tf.add_n([
            tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name# or 'biases' in v.name)
        ])

        weight_decay = wd * l2_loss

    primary_loss = cross_entropy

    with tf.name_scope('loss'):
        loss = primary_loss + weight_decay

    global_step = tf.Variable(0, trainable=False, name="global_step")

    activations['z'] = z
    activations['raw_x'] = raw_x
    activations['x'] = x
    activations['y'] = y
    activations['global_step'] = global_step
    activations['loss'] = loss
    activations['primary_loss'] = primary_loss
    activations['each'] = cross_entropy_each
    activations['cross_entropy'] = cross_entropy
    activations['phase_test'] = phase_test
    activations['weight_decay'] = weight_decay

    with tf.name_scope('summaries'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('primary_loss', primary_loss)
        tf.summary.scalar('weight_decay', weight_decay)

    return activations, info


def crop_size(network_type):
    if network_type.startswith('alex'):
        return 227
    else:
        return 224


def train(model_filename, output_dir, device='/gpu:0', time_limit=None,
          iterations=80000, network_type='alex-lrn'):
    if os.path.exists(os.path.join(output_dir, '0-done')):
        return

    rs = np.random.RandomState(0)

    for path in ['', 'snapshots']:
        os.makedirs(os.path.join(output_dir, path), exist_ok=True)

    imgs = []
    segs = []
    with tf.device('/cpu'):
        x, y, sh, imgname, scale = selfsup.datasets.voc2007_classification_batching('trainval',
                                                              batch_size=BATCH_SIZE,
                                                              input_size=crop_size(network_type),
                                                              min_scale=MIN_SCALE,
                                                              max_scale=MAX_SCALE,
                                                              #seed=SEED,
                                                              shuffle=True,
                                                              ignore_label=255,
                                                              num_threads=10,
                                                              random_mirror=True)

        """
        tr_x, tr_y = dd.io.load('/share/data/vision-greg/larsson/train_patches_mini.hf5', ['/data', '/cls'])

        keys = list(map(str, sorted(map(int, tr_x.keys()))))
        tr_x = np.asarray([tr_x[x].transpose(1, 2, 0)[..., ::-1].astype(np.float32)/255 for x in keys])
        tr_y = np.asarray([tr_y[y].astype(np.float32)%255 for y in keys])

        min_after_dequeue = 10
        capacity = min_after_dequeue + 3 * BATCH_SIZE
        num_threads = 1

        x1, y1 = tf.train.slice_input_producer([tr_x, tr_y], shuffle=False)
        x, y = tf.train.batch(
                [x1, y1], batch_size=BATCH_SIZE, capacity=capacity,
                num_threads=num_threads)
        """

    with tf.device(device):
        variables, info = build_network(x, y, model_filename=model_filename,
                                        network_type=network_type)

        selfsup.printing.print_init(info)

        lr0 = 0.001
        variables['global_stage'] = tf.Variable(0, trainable=False, dtype=tf.int32, name="global_stage")
        variables['global_stage_start'] = tf.Variable(0, trainable=False, dtype=tf.int32, name="global_stage_start")
        if False:  # Adaptive
            lr = lr0 * 10 ** tf.cast(-variables['global_stage'], tf.float32)
            max_iter = None
        else:
            steps = [int(x) for x in np.arange(1, 9) * (iterations / 8)]
            print('steps', steps)
            max_iter = steps[-1]
            lrs = list(lr0 * 2**-np.arange(len(steps) + 1, dtype=np.float32))
            lr = tf.train.piecewise_constant(variables['global_step'], steps, lrs)

        variables['learning_rate'] = lr
        tf.summary.scalar('learning_rate', lr)
        tf.summary.scalar('global_stage', variables['global_stage'])
        tf.summary.scalar('global_stage_start', variables['global_stage_start'])
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        #opt = tf.train.GradientDescentOptimizer(lr)
        grads_and_vars = opt.compute_gradients(variables['loss'])

        if 0:
            subset = [
                'task',
            ]
            grads_and_vars_subset = [
                kk
                for kk in grads_and_vars
                if kk[1].name.split('/')[0] in subset
            ]
        else:
            grads_and_vars_subset = []
            for g, v in grads_and_vars:
                if g is not None:
                    g = tf.clip_by_norm(g, 10)

                grads_and_vars_subset.append((g, v))

        """
        grads_and_vars_subset = []
        for g, v in grads_and_vars:
            if 'biases' in v.name and g is not None:
                g *= 2
            grads_and_vars_subset.append((g, v))
        """
        grads = {}
        for g, v in grads_and_vars:
            if g is not None:
                grads[v.name] = g

        selfsup.tprint('Updating variables:')
        for a, b in grads_and_vars_subset:
            selfsup.tprint('-', b.name)

        train_step = opt.apply_gradients(grads_and_vars_subset, global_step=variables['global_step'])

        variables['window_sum'] = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='window_sum')
        variables['window_last_average'] = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='window_last_average')
        variables['window_size'] = tf.Variable(0, trainable=False, dtype=tf.int32, name='window_pointer')

    with tf.Session(config=selfsup.config()) as sess:
        saver = tf.train.Saver()

        merged = tf.summary.merge_all()
        tr_sw = tf.summary.FileWriter(os.path.join('tf-log', 'train_train'), sess.graph)
        vl_sw = tf.summary.FileWriter(os.path.join('tf-log', 'train_val'), sess.graph)

        snapshot_fn = selfsup.snapshot.find(os.path.join(output_dir, 'snapshots'))
        if snapshot_fn is not None:
            selfsup.snapshot.restore(saver, sess, None, snapshot_fn)
            start_iter = variables['global_step'].eval()
        else:
            sess.run(tf.global_variables_initializer())
            start_iter = 0

        if variables['global_step'].eval() == 0:
            # Randomize fc6, fc7, fc8
            if 0:
                selfsup.tprint('Randomized')
                assigns = []
                for l, sh in [('fc6', (4096, 9216)), ('fc7', (4096, 4096)), ('task', (CLASSES, 4096))]:
                    a = info['weights'][l + ':weights']
                    #sh = a.get_shape().as_list()
                    rs = np.random.RandomState(0)
                    cfW = rs.normal(0, 0.01, size=sh).astype(np.float32)
                    print(cfW.ravel()[:5])
                    tfW = selfsup.caffe.from_caffe(cfW, name=l, conv_fc_transitionals={'fc6': (4096, 256, 6, 6)})
                    assigns.append(a.assign(tfW))

                    b = info['weights'][l + ':biases']
                    sh = b.get_shape().as_list()
                    assigns.append(b.assign(np.full(sh, 0.1, dtype=np.float32)))

                    selfsup.tprint('-', l)

                sess.run(assigns)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        time_up = False
        times = []
        train_losses = []

        #x0 = dd.image.load('/share/data/vision-greg/larsson/data/places/Images/o/office_building/gsun_58084ea7d836c63d473fc73847f1da42.jpg')
        #x0 = x0[:225, :225][np.newaxis]
        #I = {x: x0}

        #sess.run(info['activations'], {x: x0})

        while True:
            iteration = variables['global_step'].eval()

            time_up = selfsup.timesup(time_limit)

            if iteration % TEST_EVERY == 0 and iteration > 0:
                try:
                    itr_per_s = 1 / np.mean(times)
                    train_loss = np.mean(train_losses)
                except:
                    itr_per_s = np.nan
                    train_loss = np.nan

                times = []
                train_losses = []

                # Train loss
                val = False
                inputs_val = {variables['phase_test']: val}

                summary, cross_entropy, loss_wd, lri = sess.run(
                    [merged, variables['cross_entropy'],
                     variables['weight_decay'],
                     variables['learning_rate'],
                     ], feed_dict=inputs_val)

                tr_sw.add_summary(summary, iteration)

                selfsup.tprint('[{itr_per_s:3.0f} it/s] Train {iteration:8d} Loss {train_loss:7.3f} lr = {lri:7.5g} '.format(
                    itr_per_s=itr_per_s, iteration=iteration, train_loss=train_loss, lri=lri))

            # Snapshot
            if (iteration % SNAPSHOT_EVERY == 0 and iteration > start_iter or
                iteration == max_iter or time_up):
                # Save checkpoint
                snap_fn = selfsup.snapshot.new_save(saver, sess, iteration,
                        path=os.path.join(output_dir, 'snapshots'))
                print('Saved snapshot to', snap_fn)

            # Train step
            with dd.timed(callback=times.append):
                inputs = {variables['phase_test']: False}
                #p_loss, x0, y0, fc6, _, gg = sess.run([variables['primary_loss'], info['activations']['x'], info['activations']['y'], info['activations']['fc6'], train_step, grads], feed_dict=inputs)
                p_loss, _ = sess.run([variables['primary_loss'], train_step], feed_dict=inputs)

                #conv1 = info['weights']['conv1:weights'].eval().transpose(3, 2, 0, 1)[:, ::-1]
                #conv1g = gg['conv1/weights:0'].transpose(3, 2, 0, 1)[:, ::-1]
                #print(conv1g * 0.001)

                train_losses.append(p_loss)

            if iteration == max_iter or time_up:
                if iteration == max_iter:
                    with open(os.path.join(output_dir, '0-done'), 'w') as f:
                        print('done', file=f)
                break

        coord.request_stop()
        coord.join(threads)
    del sess


def test(model_filename, output_dir, device='/gpu:0', time_limit=None,
         iterations=80000, network_type='alex-lrn'):
    if not os.path.exists(os.path.join(output_dir, '0-done')) or os.path.exists(os.path.join(output_dir, '1-tested')):
        return

    start = time.time()

    imgs = []
    segs = []
    with tf.device('/cpu'):
        x, y, sh, name, scale = selfsup.datasets.voc2007_classification_batching('test',
                                                              batch_size=TEST_BATCH_SIZE,
                                                              input_size=crop_size(network_type),
                                                              min_scale=MIN_SCALE,
                                                              max_scale=MAX_SCALE,
                                                              num_threads=10,
                                                              shuffle=False,
                                                              ignore_label=255,
                                                              random_mirror=True,
                                                              )

    with tf.device(device):
        variables, info = build_network(x, y, network_type=network_type)

    # TODO: Enforce time limit here
    with tf.Session(config=selfsup.config()) as sess:
        with tf.device(device):
            saver = tf.train.Saver()
            merged = tf.summary.merge_all()

            snapshot_fn = selfsup.snapshot.find(os.path.join(output_dir, 'snapshots'))
            if len(sys.argv) == 2:
                snapshot_fn = sys.argv[1]

            if snapshot_fn is not None:
                selfsup.snapshot.restore(saver, sess, None, snapshot_fn)
                start_iter = variables['global_step'].eval()
            else:
                sess.run(tf.global_variables_initializer())
                start_iter = 0

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            yhats = []
            ytrue = []
            N = 4952

            d_yhats = {}
            d_ytrue = {}
            completed = set()

            from progressbar import ProgressBar, Percentage, Bar, ETA
            progress = ProgressBar(widgets=["Testing", Percentage(), Bar(), ETA()])
            for loop in progress(range(N+1)):
                val = True
                inputs_val = {variables['phase_test']: val}

                yhat, yy, nn = sess.run([variables['z'], variables['y'], name], feed_dict=inputs_val)

                for yhat0, y0, n0 in zip(yhat, yy, nn):
                    d_yhats[n0] = d_yhats.get(n0, [])
                    if n0 not in completed:
                        d_yhats[n0].append(yhat0)

                        if len(d_yhats[n0]) == CROPS:
                            completed.add(n0)

                    d_ytrue[n0] = y0

                if len(completed) == N:
                    break

            for key in d_ytrue:
                assert len(d_yhats[key]) >= CROPS
                yhats.append(np.mean(d_yhats[key][:CROPS], 0))
                ytrue.append(d_ytrue[key])

            yhats = np.array(yhats)
            ytrue = np.array(ytrue)

            aps = np.zeros(CLASSES)
            for c in range(CLASSES):
                yhat_c = yhats[:, c]
                ytru_c = ytrue[:, c]

                keep = ytru_c <= 1
                yhat_c = yhat_c[keep]
                ytru_c = ytru_c[keep].astype(int)

                #noise = np.random.normal(0, 1e-4, size=yhat_c.shape)
                #aps[c] = metrics.average_precision_score(ytru_c, yhat_c + noise)
                aps[c] = metrics.average_precision_score(ytru_c, yhat_c-1e-5*ytru_c)


            print(aps)
            mean_ap = np.mean(aps)
            print('mAP %.2f' % (100 * mean_ap))
            with open(os.path.join(output_dir, '1-tested'), 'w') as f:
                print('done', file=f)
            with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
                print('Model file:', os.path.abspath(model_filename), file=f)
                print('Args:', sys.argv)
                print('mAP: {mean_ap:.2%}'.format(mean_ap=mean_ap), file=f)

    del sess


def train_and_test(*args, **kwargs):
    train(*args, **kwargs)
    tf.reset_default_graph()
    test(*args, **kwargs)


if __name__ == '__main__':
    main()
