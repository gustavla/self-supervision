from __future__ import division, print_function, absolute_import
import re
import os
import glob
import deepdish as dd
import tensorflow as tf


def save(saver, sess, variables, info, path='snapshots'):
    iteration = variables['global_step'].eval()
    snap_fn = os.path.join(path, 'snapshot_iter_%d.ckpt' % iteration)
    saver.save(sess, snap_fn)
    if info is not None:
        info_fn = os.path.join(path, 'info.h5')
        info0 = {k: v for k, v in info.items() if v.size > 0}
        dd.io.save(info_fn, info0)
    return snap_fn


def new_save(saver, sess, iteration, path='snapshots'):
    snap_fn = os.path.join(path, 'snapshot_iter_%d.ckpt' % iteration)
    saver.save(sess, snap_fn)
    return snap_fn


def restore(saver, sess, info, snap_fn, path='snapshots'):
    saver.restore(sess, snap_fn)
    if info is not None:
        new_info = dd.io.load(os.path.join(path, 'info.h5'))
        info.update(new_info)


def find_old(path='snapshots'):
    snap_fns = glob.glob(os.path.join(path, 'snapshot_iter_*.ckpt'))
    snapshots = sorted([
        (int(re.sub('[^0-9]', '', os.path.splitext(fn)[0])), fn)
        for fn in snap_fns
    ])
    if snapshots:
        fn = snapshots[-1][1]
        return fn
    else:
        return None

def find(path='snapshots'):
    return tf.train.latest_checkpoint(path)
