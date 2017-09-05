from __future__ import division, print_function, absolute_import
from .loader import Loader
import selfsup.datasets


class UnlabeledImagesLoader(Loader):
    def __init__(self, path, batch_size, num_threads=5, input_size=224, resize_if_small=False,
            root_path=None):
        self._path = path
        self._batch_size = batch_size
        self._num_threads = num_threads
        self._input_size = input_size
        self._resize_if_small = resize_if_small
        self._root_path = root_path

    def batch(self):
        x, imgshape, imgname = selfsup.datasets.unlabeled_batching(
                self._path,
                batch_size=self._batch_size,
                num_threads=self._num_threads,
                input_size=self._input_size,
                resize_if_small=self._resize_if_small,
                root_path=self._root_path,
                random_mirror=True)
        self.x = x
        return x, {}
        #x, y = _batch_random_rotate(x_upright)
        #return x, y, imgshape, imgname

    @property
    def batch_size(self):
        return self._batch_size

    def start(self, session):
        pass
