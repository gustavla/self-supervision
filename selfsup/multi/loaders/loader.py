from __future__ import division, print_function, absolute_import

class Loader:
    def __init__(self, batch_size=16, num_threads=5):
        self._batch_size = batch_size
        self._num_threads = num_threads

    def batch(self):
        raise NotImplemented('Cannot use base class')

    def start(self, sess):
        pass
