from .loader import Loader
import numpy as np
import deepdish as dd
import tensorflow as tf


class CifarLoader(Loader):
    def __init__(self, path, num_classes=100, partition=0, num_threads=5, batch_size=32,
            count=None, count_from='head', class_method='random', engine='hdf5', offset=0):
        """
        count is taken before subsetting for labels
        """

        assert engine == 'hdf5'
        self._path = path
        self._num_classes = num_classes
        self._num_threads = num_threads
        self._batch_size = batch_size

        if num_classes != 100:
            if class_method == 'random':
                rs = np.random.RandomState(1443 + partition)
                labels = np.arange(100)
                rs.shuffle(labels)
                labels = labels[:num_classes]
            elif class_method == 'disjoint':
                rs = np.random.RandomState(1443)
                labels = np.arange(100)
                rs.shuffle(labels)
                labels = labels[partition*num_classes:(partition+1)*num_classes]
        else:
            labels = np.arange(100)

        self._labels = labels

        if count is not None:
            if count_from == 'head':
                ss = np.s_[offset:offset+count]
            else:
                assert offset == 0
                ss = np.s_[-count:]
        else:
            ss = np.s_[:]

        # Load data and subset it appropriately
        tr_xraw, tr_yraw = dd.io.load(path, ['/data', '/label'], sel=ss)
        tr_x = tr_xraw.transpose(0, 2, 3, 1)
        tr_y = tr_yraw.astype(np.int32)

        if num_classes != 100:
            # TODO: Doesn't work on cifar10
            Y = np.full(100, -1, dtype=np.int32)
            Y[labels] = np.arange(len(labels))

            tr_y = Y[tr_y]

        # Now sort out all -1's
        ok = tr_y != -1

        self._data = tr_x[ok]
        self._labels = tr_y[ok]

    def batch(self):
        shuffle = True
        seed = None
        min_after_dequeue = 250
        capacity = min_after_dequeue + 3 * self._batch_size

        x1, y1 = tf.train.slice_input_producer([self._data, self._labels], shuffle=shuffle, seed=seed)
        x, y = tf.train.shuffle_batch(
                [x1, y1], batch_size=self._batch_size, capacity=capacity,
                min_after_dequeue=min_after_dequeue, num_threads=self._num_threads, seed=seed)
        return x, {'y': y, 'num_classes': self._num_classes}

    @property
    def batch_size(self):
        return self._batch_size

    def start(self, session):
        pass
