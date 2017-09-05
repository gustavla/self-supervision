from .loader import Loader
import tensorflow as tf
import threading
import numpy as np
import time
import glob
import os
import imageio
import cv2
import deepdish as dd

SAMPLES_PER_VIDEO = 1
SAMPLES_PER_FRAME = 1
FRAMES = 6

def pad(x, min_side):
    if np.min(x.shape[:2]) >= min_side:
        return x
    else:
        sh = (max(min_side, x.shape[0]), max(min_side, x.shape[1])) + x.shape[2:]
        new_x = np.zeros(sh, dtype=x.dtype)
        new_x[:x.shape[0], :x.shape[1]] = x
        return new_x


def extract_optical_flow(fn, n_frames=34):
    img = dd.image.load(fn)
    if img.shape != (128*34, 128, 3):
        return []
    frames = np.array_split(img, 34, axis=0)
    grayscale_frames = [fr.mean(-1) for fr in frames]
    mags = []
    skip_frames = np.random.randint(34 - n_frames + 1)
    middle_frame = frames[np.random.randint(skip_frames, skip_frames+n_frames)]
    im0 = grayscale_frames[skip_frames]
    for f in range(1+skip_frames, 1+skip_frames+n_frames-1):
        im1 = grayscale_frames[f]
        flow = cv2.calcOpticalFlowFarneback(im0, im1,
                    None, # flow
                    0.5, # pyr_scale
                    3, # levels
                    np.random.randint(3, 20), # winsize
                    3, #iterations
                    5, #poly_n 
                    1.2, #poly_sigma
                    0 # flags
        )
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        mags.append(mag)
        im0 = im1
    mag = np.sum(mags, 0)
    mag = mag.clip(min=0)
    #norm_mag = np.tanh(mag * 10000)
    norm_mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-5)
    outputs = []
    outputs.append((middle_frame, norm_mag))
    return outputs


class VideoJPEGRollsFlowSaliency(Loader):
    def __init__(self, path, root_path='', batch_size=16, input_size=227, num_threads=10):
        self._path = path
        self._root_path = root_path
        with open(path) as f:
            self._list_files = [x.rstrip('\n') for x in f.readlines()]
        print('list_files', len(self._list_files))

        self._batch_size = batch_size
        self._input_size = input_size
        self._num_threads = num_threads
        self._coord = tf.train.Coordinator()
        self._base_shape = [batch_size, input_size, input_size]
        self._image_shape = self._base_shape + [3]
        self._label_shape = self._base_shape + [1]
        p_x = tf.placeholder(tf.float32, self._image_shape, name='x')
        p_y = tf.placeholder(tf.float32, self._label_shape, name='y')
        inputs = [p_x, p_y]
        self._queue = tf.FIFOQueue(400,
                [i.dtype for i in inputs], [i.get_shape() for i in inputs])
        self._inputs = inputs
        self._enqueue_op = self._queue.enqueue(inputs)
        self._queue_close_op = self._queue.close(cancel_pending_enqueues=True)
        self._threads = []

    def __feed(self, rank):
        time.sleep(np.random.uniform(0, 3))
        batch_x = np.zeros(self._image_shape, dtype=np.float32)
        batch_y = np.zeros(self._label_shape, dtype=np.float32)
        pool = []
        N = len(self._list_files)
        input_size = self._input_size
        while True:
            while len(pool) < self._batch_size * 30:
                i = np.random.randint(N)

                fn = os.path.join(self._root_path, self._list_files[i])
                #print(fn)
                outputs = extract_optical_flow(fn, n_frames=FRAMES)
                for img, mag in outputs:
                    img0 = dd.image.resize(img, min_side=input_size)
                    mag0 = dd.image.resize(mag, min_side=input_size)

                    # Now find a random window
                    h = np.random.randint(img0.shape[0] - input_size + 1)
                    w = np.random.randint(img0.shape[1] - input_size + 1)
                    if np.random.randint(2) == 0:
                        ss = np.s_[:]
                    else:
                        # flipped
                        ss = np.s_[:, ::-1]

                    pool.append((img0[ss], mag0[ss]))

                if len(pool) >= self._batch_size:
                    break

            for b in range(self._batch_size):
                i = np.random.randint(len(pool))
                img, mag = pool.pop(i)
                batch_x[b] = img
                batch_y[b, ..., 0] = mag

            yield {self._inputs[0]: batch_x, self._inputs[1]: batch_y}

    def __thread(self, session, rank):
        with self._coord.stop_on_exception():
            for feed_dict in self.__feed(rank):
                session.run(self._enqueue_op, feed_dict)

    def batch(self):
        x, y = self._queue.dequeue()
        return x, {'saliency': y}

    @property
    def batch_size(self):
        return self._batch_size

    def start(self, session):
        for i in range(self._num_threads):
            t = threading.Thread(target=VideoJPEGRollsFlowSaliency.__thread,
                                 args=(self, session, i))
            t.daemon = True
            t.start()
            self._threads.append(t)

    def check_status(self):
        ret = False
        for i, t in enumerate(self._threads):
            if not t.is_alive():
                print(f'Thread #{i} has died')
                ret = True
        return ret

    def stop(self, session):
        self._coord.request_stop()
        session.run(self._queue_close_op)
        self._coord.join(self._threads)
