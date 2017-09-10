from __future__ import division, print_function, absolute_import
import time
import os
import sys

from . import snapshot
from . import caffe
from . import util
from . import ops

_start_time = time.time()

def timesup(limit):
    global _start_time
    if limit is None:
        return False
    else:
        return time.time() - _start_time > 3600 * limit

def res(path):
    return os.path.join(os.path.dirname(__file__), 'res', path)

def print_git(fatal=False):
    import subprocess
    curdir = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    files = subprocess.check_output(["git", "ls-files", "-m"]).decode('utf-8').split('\n')
    files = [f for f in files if f]
    if len(files) > 0:
        print("HAS {len(files)} MODIFIED FILES")
        if fatal:
            sys.exit(1)
    else:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode('utf-8').strip()
        print('sha {sha}')
    os.chdir(curdir)


from .util import config, tlog, tprint, mkdirs

from copy import copy
from collections import OrderedDict
import numpy as np

class MeanCalculator(object):
    def __init__(self):
        self._sum = None
        self._count = 0

    def add(self, values):
        self._count += 1
        if self._sum is None:
            self._sum = copy(values)
        elif isinstance(values, (int, float, np.float32, np.int32, np.ndarray)):
            self._sum += values
        elif isinstance(values, (dict, OrderedDict)):
            for k, v in values.items():
                self._sum[k] += v
        elif isinstance(values, list):
            for i, v in enumerate(values):
                self._sum[i] += v
        else:
            raise ValueError('Unsupported type')

    def pop(self, default=None):
        """
        Return mean and reset mean
        """
        if self._count == 0:
            return default

        if isinstance(self._sum, np.ndarray):
            m = None
        else:
            m = self._sum.__class__()
        if isinstance(self._sum, (int, float, np.float32, np.int32, np.ndarray)):
            m = self._sum / self._count
        elif isinstance(self._sum, (dict, OrderedDict)):
            for k, v in self._sum.items():
                m[k] = v / self._count
        elif isinstance(self._sum, list):
            for v in self._sum:
                m.append(v / self._count)
        else:
            raise ValueError('Unsupported type')

        self._sum = None
        self._count = 0
        return m

    def empty(self):
        return self._count == 0
