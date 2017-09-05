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

SAMPLES_PER_VIDEO = 5
SAMPLES_PER_FRAME = 10
FRAMES = 4

MIN_QUEUE_SIZE = 250

def pad(x, min_side):
    if np.min(x.shape[:2]) >= min_side:
        return x
    else:
        sh = (max(min_side, x.shape[0]), max(min_side, x.shape[1])) + x.shape[2:]
        new_x = np.zeros(sh, dtype=x.dtype)
        new_x[:x.shape[0], :x.shape[1]] = x
        return new_x


def extract_optical_flow(fn, times, frames=8, scale_factor=1.0):
    cap = cv2.VideoCapture(fn)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    outputs = []
    if n_frames < frames * 2:
        return outputs

    def resize(im):
        if scale_factor != 1.0:
            new_size = (int(im.shape[1] * scale_factor), int(im.shape[0] * scale_factor))
            return cv2.resize(im, new_size, interpolation=cv2.INTER_LINEAR)
        else:
            return im

    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(t * n_frames, n_frames - 1 - frames))
        ret, frame0 = cap.read()
        im0 = resize(cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY))
        mags = []
        middle_frame = frame0
        for f in range(frames - 1):
            ret, frame1 = cap.read()
            if f == frames // 2:
                middle_frame = frame1
            im1 = resize(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY))
            flow = cv2.calcOpticalFlowFarneback(im0, im1,
                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            mags.append(mag)
            im0 = im1
        mag = np.sum(mags, 0)
        mag = mag.clip(min=0)
        norm_mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-5)
        x = middle_frame[..., ::-1].astype(np.float32) / 255
        outputs.append((x, norm_mag))
        return outputs


class VideoAVIFlowSaliency(Loader):
    def __init__(self, path, batch_size=16, input_size=227,
                 scale_factor=1.0, num_threads=10):
        self._path = path

        self._list_files = glob.glob(os.path.join(path, "**/*.avi"))

        self._batch_size = batch_size
        self._scale_factor = scale_factor
        self._image_size = input_size
        self._label_size = int(input_size * self._scale_factor)
        self._num_threads = num_threads
        self._coord = tf.train.Coordinator()
        self._image_shape = [batch_size, self._image_size, self._image_size, 3]
        self._label_shape = [batch_size, self._label_size, self._label_size, 1]
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
        while True:
            while len(pool) < MIN_QUEUE_SIZE:# self._batch_size * 30:
                i = np.random.randint(N)#, size=self._batch_size)

                fn = self._list_files[i]
                times = np.random.uniform(size=SAMPLES_PER_VIDEO)
                outputs = extract_optical_flow(fn, times=times, frames=FRAMES,
                                               scale_factor=self._scale_factor)
                for img, mag in outputs:
                    img = pad(img, min_side=self._image_size)
                    mag = pad(mag, min_side=self._label_size)

                    max_side = np.min(img.shape[:2])
                    label_max_side = np.min(mag.shape[:2])

                    for j in range(SAMPLES_PER_FRAME):
                        r, rh, rw, flip = np.random.uniform(size=4)
                        if self._image_size != max_side:
                            side = self._image_size + int(r * (max_side + 1 - self._image_size))
                            img0 = dd.image.resize(img, min_side=side)
                        else:
                            img0 = img

                        if self._label_size != label_max_side:
                            side = self._label_size + int(r * (label_max_side + 1 - self._label_size))
                            mag0 = dd.image.resize(mag, min_side=side)
                        else:
                            mag0 = mag

                        # Now find a random window
                        if flip > 0.5:
                            img0 = img0[:, ::-1]
                            mag0 = mag0[:, ::-1]

                        image_h = int(rh * (img0.shape[0] - self._image_size + 1))
                        image_w = int(rw * (img0.shape[1] - self._image_size + 1))
                        image_ss = np.s_[image_h:image_h+self._image_size, image_w:image_w+self._image_size]

                        label_h = int(rh * (mag0.shape[0] - self._label_size + 1))
                        label_w = int(rw * (mag0.shape[1] - self._label_size + 1))
                        label_ss = np.s_[label_h:label_h+self._label_size, label_w:label_w+self._label_size]

                        pool.append((img0[image_ss], mag0[label_ss]))

                #if len(pool) >= self._batch_size:
                    #break

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

    @property
    def scale_factor(self):
        return self._scale_factor

    def start(self, session):
        for i in range(self._num_threads):
            t = threading.Thread(target=VideoAVIFlowSaliency.__thread,
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
