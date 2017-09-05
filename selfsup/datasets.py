import numpy as np
import tensorflow as tf
import os
import deepdish as dd
import struct
from array import array

from tensorflow.python.framework import ops
from tensorflow.contrib.framework.python.framework import is_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import check_ops


def decode_image(contents, channels=None, name=None):
  """Convenience function for `decode_gif`, `decode_jpeg`, and `decode_png`.
  Detects whether an image is a GIF, JPEG, or PNG, and performs the appropriate
  operation to convert the input bytes `string` into a `Tensor` of type `uint8`.

  Note: `decode_gif` returns a 4-D array `[num_frames, height, width, 3]`, as
  opposed to `decode_jpeg` and `decode_png`, which return 3-D arrays
  `[height, width, num_channels]`. Make sure to take this into account when
  constructing your graph if you are intermixing GIF files with JPEG and/or PNG
  files.

  Args:
    contents: 0-D `string`. The encoded image bytes.
    channels: An optional `int`. Defaults to `0`. Number of color channels for
      the decoded image.
    name: A name for the operation (optional)

  Returns:
    `Tensor` with type `uint8` with shape `[height, width, num_channels]` for
      JPEG and PNG images and shape `[num_frames, height, width, 3]` for GIF
      images.
  """
  with ops.name_scope(name, 'decode_image') as scope:
    if channels not in (None, 0, 1, 3):
      raise ValueError('channels must be in (None, 0, 1, 3)')
    substr = tf.substr(contents, 0, 4)

    def _gif():
      # Create assert op to check that bytes are GIF decodable
      is_gif = tf.equal(substr, b'\x47\x49\x46\x38', name='is_gif')
      decode_msg = 'Unable to decode bytes as JPEG, PNG, or GIF'
      assert_decode = control_flow_ops.Assert(is_gif, [decode_msg])
      # Create assert to make sure that channels is not set to 1
      # Already checked above that channels is in (None, 0, 1, 3)
      gif_channels = 0 if channels is None else channels
      good_channels = tf.not_equal(gif_channels, 1, name='check_channels')
      channels_msg = 'Channels must be in (None, 0, 3) when decoding GIF images'
      assert_channels = control_flow_ops.Assert(good_channels, [channels_msg])
      with ops.control_dependencies([assert_decode, assert_channels]):
        return gen_image_ops.decode_gif(contents)

    def _png():
      return gen_image_ops.decode_png(contents, channels)

    def check_png():
      is_png = tf.equal(substr, b'\211PNG', name='is_png')
      return control_flow_ops.cond(is_png, _png, _gif, name='cond_png')

    def _jpeg():
      return gen_image_ops.decode_jpeg(contents, channels)

    is_jpeg = tf.logical_or(tf.equal(substr, b'\xff\xd8\xff\xe0', name='is_jpeg0'),
                           tf.equal(substr, b'\xff\xd8\xff\xe1', name='is_jpeg0'))

    return control_flow_ops.cond(is_jpeg, _jpeg, check_png, name='cond_jpeg')


VOC_CLASSES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]

def _assert(cond, ex_type, msg):
  """A polymorphic assert, works with tensors and boolean expressions.

  If `cond` is not a tensor, behave like an ordinary assert statement, except
  that a empty list is returned. If `cond` is a tensor, return a list
  containing a single TensorFlow assert op.

  Args:
    cond: Something evaluates to a boolean value. May be a tensor.
    ex_type: The exception class to use.
    msg: The error message.

  Returns:
    A list, containing at most one assert op.
  """
  if is_tensor(cond):
    return [logging_ops.Assert(cond, [msg])]
  else:
    if not cond:
      raise ex_type(msg)
    else:
      return []


def _ImageDimensions(images, static_only=True):
    """Returns the dimensions of an image tensor.

    Args:
        images: 4-D Tensor of shape `[batch, height, width, channels]`
        static_only: Boolean, whether to return only static shape.

    Returns:
        list of integers `[batch, height, width, channels]`, when static shape is
        fully defined or `static_only` is `True`.
        list of integer scalar tensors `[batch, height, width, channels]`, when
        static shape is not fully defined.
    """
    # A simple abstraction to provide names for each dimension. This abstraction
    # should make it simpler to switch dimensions in the future (e.g. if we ever
    # want to switch height and width.)
    if static_only or images.get_shape().is_fully_defined():
        return images.get_shape().as_list()
    else:
        return tf.unstack(tf.shape(images))


def _Check3DImage(image, require_static=True):
  """Assert that we are working with properly shaped image.

  Args:
    image: 3-D Tensor of shape [height, width, channels]
    require_static: If `True`, requires that all dimensions of `image` are
      known and non-zero.

  Raises:
    ValueError: if `image.shape` is not a 3-vector.

  Returns:
    An empty list, if `image` has fully defined dimensions. Otherwise, a list
    containing an assert op is returned.
  """
  try:
    image_shape = image.get_shape().with_rank(3)
  except ValueError:
    raise ValueError("'image' must be three-dimensional.")
  if require_static and not image_shape.is_fully_defined():
    raise ValueError("'image' must be fully defined.")
  if any(x == 0 for x in image_shape):
    raise ValueError("all dims of 'image.shape' must be > 0: %s" %
                     image_shape)
  if not image_shape.is_fully_defined():
    return [check_ops.assert_positive(array_ops.shape(image),
                                      ["all dims of 'image.shape' "
                                       "must be > 0."])]
  else:
    return []


def pad_to_ensure_size(image, target_height, target_width):
    """Pads if below target size, but does nothing if above.

    If `width` or `height` is smaller than the specified `target_width` or
    `target_height` respectively, this op centrally pads with 0 along that
    dimension.

    Args:
        image: 3-D tensor of shape `[height, width, channels]`
        target_height: Target height.
        target_width: Target width.

    Raises:
        ValueError: if `target_height` or `target_width` are zero or negative.

    Returns:
        Padded image of shape
        `[max(target_height, height), max(target_width, height), channels]`
    """
    image = ops.convert_to_tensor(image, name='image')

    assert_ops = []
    assert_ops += _Check3DImage(image, require_static=False)

    image = control_flow_ops.with_dependencies(assert_ops, image)
    # `crop_to_bounding_box` and `pad_to_bounding_box` have their own checks.
    # Make sure our checks come first, so that error messages are clearer.
    if is_tensor(target_height):
        target_height = control_flow_ops.with_dependencies(assert_ops, target_height)
    if is_tensor(target_width):
        target_width = control_flow_ops.with_dependencies(assert_ops, target_width)

    def max_(x, y):
        if is_tensor(x) or is_tensor(y):
            return math_ops.maximum(x, y)
        else:
            return max(x, y)

    height, width, _ = _ImageDimensions(image, static_only=False)
    width_diff = target_width - width
    offset_crop_width = max_(-width_diff // 2, 0)
    offset_pad_width = max_(width_diff // 2, 0)

    height_diff = target_height - height
    offset_crop_height = max_(-height_diff // 2, 0)
    offset_pad_height = max_(height_diff // 2, 0)

    # Maybe pad if needed.
    resized = tf.image.pad_to_bounding_box(image, offset_pad_height, offset_pad_width,
                                           max_(target_height, height), max_(target_width, width))

    # In theory all the checks below are redundant.
    if resized.get_shape().ndims is None:
        raise ValueError('resized contains no shape.')

    resized_height, resized_width, _ = \
        _ImageDimensions(resized, static_only=False)

    #assert_ops = []
    #assert_ops += _assert(equal_(resized_height, target_height), ValueError,
                          #'resized height is not correct.')
    #assert_ops += _assert(equal_(resized_width, target_width), ValueError,
                          #'resized width is not correct.')

    #resized = control_flow_ops.with_dependencies(assert_ops, resized)
    return resized


def resize_to_ensure_size(image, target_height, target_width):
    height, width, _ = _ImageDimensions(image, static_only=False)

    #if height < target_height or width < target_width:
        # Do not preserve aspect ratio
    image4 = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image4, [tf.maximum(height, target_height), tf.maximum(width, target_width)])
    return image[0]


def _voc_seg_load_file(path, epochs=None, shuffle=True, seed=0):

    PASCAL_ROOT = os.environ['VOC_DIR']
    filename_queue = tf.train.string_input_producer([path],
            num_epochs=epochs, shuffle=shuffle, seed=seed)

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    image_path, seg_path = tf.decode_csv(value, record_defaults=[[''], ['']], field_delim=' ')

    image_abspath = PASCAL_ROOT + image_path
    seg_abspath = PASCAL_ROOT + seg_path

    image_content = tf.read_file(image_abspath)
    image = decode_image(image_content, channels=3)
    image.set_shape([None, None, 3])

    imgshape = tf.shape(image)[:2]
    imgname = image_path

    seg_content = tf.read_file(seg_abspath)
    seg = tf.cast(tf.image.decode_png(seg_content, channels=1), tf.int32)
    return image, seg, imgshape, imgname


def voc_seg_batching(path, batch_size, input_size, epochs=None,
                     shuffle=True, min_after_dequeue=250, seed=0,
                     pad255=False, num_threads=1, color_transform=None, random_mirror=False):
    assert seed is not None, "Seed must be specified, to synchronize images and segmentation maps"

    image, seg, imgshape, imgname = _voc_seg_load_file(path, epochs=epochs, shuffle=shuffle, seed=seed)

    if pad255:
        seg += 1

    image = tf.cast(image, tf.float32) / 255.0
    if color_transform is not None:
        image = color_transform(image)

    pad_image = pad_to_ensure_size(image, input_size, input_size)
    pad_seg = pad_to_ensure_size(seg, input_size, input_size)

    fixed_image = tf.random_crop(pad_image, (input_size, input_size, 3), seed=seed)
    #fixed_image = tf.cast(fixed_image, tf.float32) / 255.0

    if pad255:
        fixed_seg = tf.random_crop(tf.cast(pad_seg, tf.int32), (input_size, input_size, 1), seed=seed)
        fixed_seg = ((fixed_seg + 255) % 256)
    else:
        fixed_seg = tf.random_crop(pad_seg, (input_size, input_size, 1), seed=seed)
        fixed_seg = tf.cast(fixed_seg, tf.int32)

    if random_mirror:
        assert seed is not None
        fixed_image = tf.image.random_flip_left_right(fixed_image, seed=seed)
        fixed_seg = tf.image.random_flip_left_right(fixed_seg, seed=seed)

    capacity = min_after_dequeue + 3 * batch_size
    if shuffle:
        batch_image, batch_seg, batch_imgshape, batch_imgname = tf.train.shuffle_batch(
                [fixed_image, fixed_seg, imgshape, imgname], batch_size=batch_size, capacity=capacity,
                min_after_dequeue=min_after_dequeue, num_threads=num_threads)
    else:
        batch_image, batch_seg, batch_imgshape, batch_imgname  = tf.train.batch(
                [fixed_image, fixed_seg, imgshape, imgname], batch_size=batch_size, capacity=capacity,
                num_threads=num_threads)

    batch_seg = tf.squeeze(batch_seg, [3])

    return batch_image, batch_seg, batch_imgshape, batch_imgname


def _imagenet_load_file(path, epochs=None, shuffle=True, seed=0, subset='train', prepare_path=True):
    IMAGENET_ROOT = os.environ.get('IMAGENET_DIR', '')
    if not isinstance(path, list):
        path = [path]
    filename_queue = tf.train.string_input_producer(path,
            num_epochs=epochs, shuffle=shuffle, seed=seed)

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    image_path, label_str = tf.decode_csv(value, record_defaults=[[''], ['']], field_delim=' ')

    if prepare_path:
        image_abspath = IMAGENET_ROOT + '/images/' + subset + image_path
    else:
        image_abspath = image_path

    image_content = tf.read_file(image_abspath)
    image = decode_image(image_content, channels=3)
    image.set_shape([None, None, 3])

    imgshape = tf.shape(image)[:2]
    label = tf.string_to_number(label_str, out_type=tf.int32)

    return image, label, imgshape, image_path


def _relpath_no_label_load_file(path, root_path, epochs=None, shuffle=True, seed=0):
    filename_queue = tf.train.string_input_producer([path],
            num_epochs=epochs, shuffle=shuffle, seed=seed)

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    #image_path, = tf.decode_csv(value, record_defaults=[['']], field_delim=' ')
    image_path = value

    image_abspath = root_path + '/' + image_path

    image_content = tf.read_file(image_abspath)
    image = decode_image(image_content, channels=3)
    image.set_shape([None, None, 3])

    imgshape = tf.shape(image)[:2]

    return image, imgshape, image_path


def _abspath_no_label_load_file(path, epochs=None, shuffle=True, seed=0):
    filename_queue = tf.train.string_input_producer([path],
            num_epochs=epochs, shuffle=shuffle, seed=seed)

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    #image_path, = tf.decode_csv(value, record_defaults=[['']], field_delim=' ')
    image_path = value

    image_abspath = image_path

    image_content = tf.read_file(image_abspath)
    image = decode_image(image_content, channels=3)
    image.set_shape([None, None, 3])

    imgshape = tf.shape(image)[:2]

    return image, imgshape, image_path


def do_center_crop(value, size, name=None):
    """Randomly crops a tensor to a given size.
    Slices a shape `size` portion out of `value` at a uniformly chosen offset.
    Requires `value.shape >= size`.
    If a dimension should not be cropped, pass the full size of that dimension.
    For example, RGB images can be cropped with
    `size = [crop_height, crop_width, 3]`.
    Args:
        value: Input tensor to crop.
        size: 1-D tensor with size the rank of `value`.
        seed: Python integer. Used to create a random seed. See
            [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
            for behavior.
        name: A name for this operation (optional).
    Returns:
        A cropped tensor of the same rank as `value` and shape `size`.
    """
    # TODO(shlens): Implement edge case to guarantee output size dimensions.
    # If size > value.shape, zero pad the result so that it always has shape
    # exactly size.
    from tensorflow.python.framework import dtypes
    with ops.op_scope([value, size], name, "center_crop") as name:
        value = ops.convert_to_tensor(value, name="value")
        size = ops.convert_to_tensor(size, dtype=dtypes.int32, name="size")
        shape = array_ops.shape(value)
        check = logging_ops.Assert(
                math_ops.reduce_all(shape >= size),
                ["Need value.shape >= size, got ", shape, size])
        shape = control_flow_ops.with_dependencies([check], shape)
        limit = shape - size + 1
        offset = tf.random_uniform(
                array_ops.shape(shape),
                dtype=size.dtype,
                maxval=size.dtype.max,
                seed=0) % limit
        offset2 = shape // 2 - size // 2
        #import ipdb; ipdb.set_trace()
        return array_ops.slice(value, offset, size, name=name)


def classification_resizing_batching(path, batch_size, input_size, epochs=None,
                            shuffle=True, center_crop=False, min_after_dequeue=250, seed=0,
                            num_threads=1, color_transform=None, random_mirror=False,
                            min_size=None, max_size=None,
                            subset='train', max_min_side=None):
    assert seed is not None, "Seed must be specified, to synchronize images and segmentation maps"
    image, label, imgshape, imgname = _imagenet_load_file(path, epochs=epochs, shuffle=shuffle, seed=seed, subset=subset,
                  prepare_path=subset is not None)

    image = tf.cast(image, tf.float32) / 255.0

    if color_transform is not None:
        image = color_transform(image)

    imgshape = tf.shape(image)
    height, width, _ = tf.unstack(imgshape)
    smaller_side = tf.to_float(tf.minimum(height, width))

    if min_size is not None and max_size is not None:
        scale = tf.random_uniform([], minval=tf.to_float(min_size) / smaller_side, maxval=tf.to_float(max_size) / smaller_side, seed=seed)
        new_height = tf.to_int32(tf.to_float(height) * scale)
        new_width = tf.to_int32(tf.to_float(width) * scale)
        image = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(image, 0), [new_height, new_width]), [0])
    else:
        scale = tf.constant(1.0)
        new_height = height
        new_width = width

    pad_image = pad_to_ensure_size(image, input_size, input_size)

    if not center_crop:
        fixed_image = tf.random_crop(pad_image, (input_size, input_size, 3), seed=seed)
    else:
        if max_min_side is not None:
            min_side = tf.random_uniform([1], minval=input_size, maxval=max_min_side, dtype=tf.int32, seed=seed)
            pad_image = dd.image.resize(pad_image, min_side=min_side)

        #height, width, _ = _ImageDimensions(pad_image, static_only=False)
        fixed_image = array_ops.slice(pad_image, [new_height//2 - input_size//2,
                                              new_width//2 - input_size//2, 0],
                                              [input_size, input_size, 3])
        #fixed_image = array_ops.slice(pad_image, [0, 0, 0],
                                              #[input_size, input_size, 3])
        #fixed_image = tf.image.crop_to_bounding_box(pad_image,
                #0, 0, input_size, input_size)


    if random_mirror:
        assert seed is not None
        fixed_image = tf.image.random_flip_left_right(fixed_image, seed=seed)

    capacity = min_after_dequeue + 3 * batch_size
    if shuffle:
        batch_image, batch_label, batch_imgshape, batch_imgname = tf.train.shuffle_batch(
                [fixed_image, label, imgshape, imgname], batch_size=batch_size, capacity=capacity,
                min_after_dequeue=min_after_dequeue, num_threads=num_threads)
    else:
        batch_image, batch_label, batch_imgshape, batch_imgname  = tf.train.batch(
                [fixed_image, label, imgshape, imgname], batch_size=batch_size, capacity=capacity,
                num_threads=num_threads)

    return batch_image, batch_label, batch_imgshape, batch_imgname


def classification_batching(path, batch_size, input_size, epochs=None,
                            shuffle=True, center_crop=False, min_after_dequeue=250, seed=0,
                            num_threads=10, color_transform=None, random_mirror=False,
                            subset='train', max_min_side=None):
    assert seed is not None, "Seed must be specified, to synchronize images and segmentation maps"
    image, label, imgshape, imgname = _imagenet_load_file(path, epochs=epochs, shuffle=shuffle, seed=seed, subset=subset,
                  prepare_path=subset is not None)

    image = tf.cast(image, tf.float32) / 255.0
    if color_transform is not None:
        image = color_transform(image)

    pad_image = pad_to_ensure_size(image, input_size, input_size)

    if not center_crop:
        fixed_image = tf.random_crop(pad_image, (input_size, input_size, 3), seed=seed)
    else:
        if max_min_side is not None:
            min_side = tf.random_uniform([1], minval=input_size, maxval=max_min_side, dtype=tf.int32, seed=seed)
            #pad_image = dd.image.resize(pad_image, min_side=min_side)
            pad_image = tf.image.resize_bilinear(pad_image, min_side=min_side)

        height, width, _ = _ImageDimensions(pad_image, static_only=False)
        fixed_image = array_ops.slice(pad_image, [height//2 - input_size//2,
                                              width//2 - input_size//2, 0],
                                              [input_size, input_size, 3])
        #fixed_image = array_ops.slice(pad_image, [0, 0, 0],
                                              #[input_size, input_size, 3])
        #fixed_image = tf.image.crop_to_bounding_box(pad_image,
                #0, 0, input_size, input_size)


    if random_mirror:
        assert seed is not None
        fixed_image = tf.image.random_flip_left_right(fixed_image, seed=seed)

    capacity = min_after_dequeue + 3 * batch_size
    if shuffle:
        batch_image, batch_label, batch_imgshape, batch_imgname = tf.train.shuffle_batch(
                [fixed_image, label, imgshape, imgname], batch_size=batch_size, capacity=capacity,
                min_after_dequeue=min_after_dequeue, num_threads=num_threads)
    else:
        batch_image, batch_label, batch_imgshape, batch_imgname  = tf.train.batch(
                [fixed_image, label, imgshape, imgname], batch_size=batch_size, capacity=capacity,
                num_threads=num_threads)

    return batch_image, batch_label, batch_imgshape, batch_imgname


def unlabeled_batching(path, batch_size, input_size, epochs=None,
                            shuffle=True, center_crop=False, min_after_dequeue=250, seed=0,
                            num_threads=1, color_transform=None, random_mirror=False,
                            resize_if_small=False, root_path=None):
    assert seed is not None, "Seed must be specified, to synchronize images and segmentation maps"
    if root_path is not None:
        image, imgshape, imgname = _relpath_no_label_load_file(path, root_path, epochs=epochs, shuffle=shuffle, seed=seed)
    else:
        image, imgshape, imgname = _abspath_no_label_load_file(path, epochs=epochs, shuffle=shuffle, seed=seed)

    image = tf.cast(image, tf.float32) / 255.0
    if color_transform is not None:
        image = color_transform(image)

    if resize_if_small:
        ensure_size = resize_to_ensure_size
    else:
        ensure_size = pad_to_ensure_size

    pad_image = ensure_size(image, input_size, input_size)

    if not center_crop:
        fixed_image = tf.random_crop(pad_image, (input_size, input_size, 3), seed=seed)
    else:
        height, width, _ = _ImageDimensions(pad_image, static_only=False)
        fixed_image = array_ops.slice(pad_image, [height//2 - input_size//2,
                                              width//2 - input_size//2, 0],
                                              [input_size, input_size, 3])
        #fixed_image = array_ops.slice(pad_image, [0, 0, 0],
                                              #[input_size, input_size, 3])
        #fixed_image = tf.image.crop_to_bounding_box(pad_image,
                #0, 0, input_size, input_size)


    if random_mirror:
        assert seed is not None
        fixed_image = tf.image.random_flip_left_right(fixed_image, seed=seed)

    capacity = min_after_dequeue + 3 * batch_size
    if shuffle:
        batch_image, batch_imgshape, batch_imgname = tf.train.shuffle_batch(
                [fixed_image, imgshape, imgname], batch_size=batch_size, capacity=capacity,
                min_after_dequeue=min_after_dequeue, num_threads=num_threads)
    else:
        batch_image, batch_imgshape, batch_imgname  = tf.train.batch(
                [fixed_image, imgshape, imgname], batch_size=batch_size, capacity=capacity,
                num_threads=num_threads)

    return batch_image, batch_imgshape, batch_imgname


def voc2007_classification_generator_save(which, batch_size, input_size, outer_input_size,
                            shuffle=True,# seed=0,
                            color_transform=None, random_mirror=False):
    path = os.path.expandvars('$VOC2007_DIR/ImageSets/Main')

    # First load image list

    fn = os.path.join(path, '{}.txt'.format(which))
    imgids = np.genfromtxt(fn, dtype=np.int32)#[:100]
    C = np.zeros((imgids.size, len(VOC_CLASSES)), dtype=np.float32)

    for c, cls_name in enumerate(VOC_CLASSES):
        fn = os.path.join(path, '{}_{}.txt'.format(cls_name, which))
        data = np.genfromtxt(fn)#[:100]
        C[:, c] = data[:, 1]

    # Convert to 0.0, 0.5, 1.0
    C = (C + 1) / 2

    filenames = [os.path.expandvars('$VOC2007_DIR/JPEGImages/{:06d}.jpg').format(imgid)
            for imgid in imgids]

    rs = np.random.RandomState(0)
    imgs = np.zeros((len(imgids), input_size, input_size, 3), dtype=np.float32)
    for i, fn in enumerate(filenames):
        if i % 200 == 0:
            print(i)
        img = dd.image.load(fn)
        if color_transform is not None:
            img = color_transform(img)

        # Resize to smaller side
        img = dd.image.resize(img, min_side=input_size)
        img = dd.image.crop(img, (input_size, input_size))

        imgs[i] = img

    dd.io.save('{}.h5'.format(which), dict(data=imgs, labels=C))

    if shuffle:
        rs = np.random.RandomState()
        while True:
            II = rs.randint(len(imgs), size=batch_size)

            ii, cc = imgs[II], C[II]
            if random_mirror and rs.randint(2) == 1:
                ii = ii[:, :, ::-1]
            yield ii, cc
    else:
        for i in range(len(imgs)//batch_size):
            ss = np.s_[i*batch_size:(i+1)*batch_size]
            yield imgs[ss], C[ss]


def voc2007_classification_batching(which, batch_size, input_size,# outer_input_size=None,
                            shuffle=True, seed=None,
                            random_mirror=False,
                            min_scale=None, max_scale=None,
                            center_crop=False,
                            ignore_label=None,
                            min_after_dequeue=250, num_threads=10):
    path = os.path.expandvars('$VOC2007_DIR/ImageSets/Main')

    # First load image list
    fn = os.path.join(path, '{}.txt'.format(which))
    imgids = np.genfromtxt(fn, dtype=str)
    C = np.zeros((imgids.size, len(VOC_CLASSES)), dtype=np.float32)

    for c, cls_name in enumerate(VOC_CLASSES):
        fn = os.path.join(path, '{}_{}.txt'.format(cls_name, which))
        data = np.genfromtxt(fn)
        C[:, c] = data[:, 1]

    # Convert to 0.0, 0.5, 1.0
    if ignore_label is not None:
        # -1 -> 0
        # 0 -> ignore_label
        # 1 -> 1
        C = tf.to_float(C == 1) * 1 + tf.to_float(C == 0) * ignore_label
    else:
        C = (C + 1) / 2

    filenames = [os.path.expandvars('$VOC2007_DIR/JPEGImages/{}.jpg').format(imgid)
            for imgid in imgids]

    imgname, label = tf.train.slice_input_producer([filenames, C], shuffle=shuffle, seed=seed)

    image_content = tf.read_file(imgname)
    image = decode_image(image_content, channels=3)
    image.set_shape([None, None, 3])
    imgshape = tf.shape(image)
    height, width, _ = tf.unstack(imgshape)

    if min_scale is not None and max_scale is not None:
        scale = tf.random_uniform([], minval=min_scale, maxval=max_scale, seed=seed)
        new_height = tf.to_int32(tf.to_float(height) * scale)
        new_width = tf.to_int32(tf.to_float(width) * scale)
        image = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(image, 0), [new_height, new_width]), [0])
    else:
        scale = tf.constant(1)
        new_height = height
        new_width = width

    new_imgshape = tf.stack([new_height, new_width, 3])

    image = pad_to_ensure_size(image, input_size, input_size)
    #new_height = tf.maximum(new_height, input_size)
    new_height, new_width, _ = tf.unstack(tf.shape(image))

    if not center_crop:
        fixed_image = tf.random_crop(image, (input_size, input_size, 3), seed=seed)
    else:
        fixed_image = array_ops.slice(image, [new_height//2 - input_size//2,
                                              new_width//2 - input_size//2,
                                              0],
                                              [input_size, input_size, 3])

    if random_mirror:
        fixed_image = tf.image.random_flip_left_right(fixed_image, seed=seed)

    fixed_image = tf.to_float(fixed_image) / 255.0

    capacity = min_after_dequeue + 3 * batch_size
    if shuffle:
        batch_image, batch_label, batch_imgshape, batch_imgname, batch_scale = tf.train.shuffle_batch(
                [fixed_image, label, new_imgshape, imgname, scale], batch_size=batch_size, capacity=capacity,
                min_after_dequeue=min_after_dequeue, num_threads=num_threads, seed=seed)
    else:
        batch_image, batch_label, batch_imgshape, batch_imgname, batch_scale = tf.train.batch(
                [fixed_image, label, new_imgshape, imgname, scale], batch_size=batch_size, capacity=capacity,
                num_threads=num_threads)

    return batch_image, batch_label, batch_imgshape, batch_imgname, batch_scale


def voc2007_classification_generator(which, batch_size, input_size, outer_input_size=None,
                            shuffle=True, seed=None,
                            color_transform=None, random_mirror=False,
                            min_scale=None, max_scale=None,
                            return_filenames=False):
    path = os.path.expandvars('$VOC2007_DIR/ImageSets/Main')

    # First load image list

    fn = os.path.join(path, '{}.txt'.format(which))
    imgids = np.genfromtxt(fn, dtype=np.int32)#[:100]
    C = np.zeros((imgids.size, len(VOC_CLASSES)), dtype=np.float32)

    for c, cls_name in enumerate(VOC_CLASSES):
        fn = os.path.join(path, '{}_{}.txt'.format(cls_name, which))
        data = np.genfromtxt(fn)#[:100]
        C[:, c] = data[:, 1]

    # Convert to 0.0, 0.5, 1.0
    C = (C + 1) / 2

    filenames = [os.path.expandvars('$VOC2007_DIR/JPEGImages/{:06d}.jpg').format(imgid)
            for imgid in imgids]

    rs = np.random.RandomState(seed)
    def load_image(imgid):
        fn = os.path.expandvars('$VOC2007_DIR/JPEGImages/{:06d}.jpg').format(imgid)
        img = dd.image.load(fn)
        if color_transform is not None:
            img = color_transform(img)

        if min_scale is not None and max_scale is not None:
            s = rs.uniform(min_scale, max_scale)
            img = dd.image.resize_by_factor(img, factor=s)

        # Resize to smaller side
        img = pad_if_too_small(img, (input_size, input_size))
        if outer_input_size is not None:
            img = dd.image.resize(img, min_side=outer_input_size)
        #img = dd.image.crop(img, (input_size, outer_input_size))
        h, w = img.shape[:2]
        if shuffle:
            dh = rs.randint(h - input_size + 1)
            dw = rs.randint(w - input_size + 1)
        else:
            dh = (h - input_size + 1) // 2
            dw = (w - input_size + 1) // 2

        img = img[dh:dh+input_size, dw:dw+input_size]
        return img

    if shuffle:
        #rs = np.random.RandomState()
        while True:
            II = rs.randint(len(imgids), size=batch_size)
            ii = np.array([
                load_image(imgid) for imgid in imgids[II]
            ])

            #ii, cc = imgs[II], C[II]
            cc = C[II]
            if random_mirror and rs.randint(2) == 1:
                ii = ii[:, :, ::-1]
            yield ii, cc
    else:
        all_II = np.arange(len(imgids))
        for i in range(int(np.ceil(len(imgids) / batch_size))):
            ss = np.s_[i*batch_size:(i+1)*batch_size]
            II = all_II[ss]
            xx = np.zeros((batch_size, input_size, input_size, 3), dtype=np.float32)
            yy = np.zeros((batch_size, 20), dtype=np.float32)
            xx[:len(imgids[II])] = np.array([
                load_image(imgid) for imgid in imgids[II]
            ])
            yy[:len(imgids[II])] = C[ss]
            if return_filenames:
                nn = [j for j in imgids[ss]]
                nn += [0] * (batch_size - len(nn))
                yield xx, yy, nn
            else:
                yield xx, yy


def pad_if_too_small(img, shape, value=0.0):
    if img.shape[0] >= shape[0] and img.shape[1] >= shape[1]:
        return img
    else:
        img0 = np.full((max(img.shape[0], shape[0]), max(img.shape[1], shape[1])) + img.shape[2:], value, dtype=np.float32)
        img0[:img.shape[0], :img.shape[1]] = img
        return img0

def voc2007_classification_generator_10crop(which, input_size, outer_input_size=None,
                            shuffle=True,# seed=0,
                            color_transform=None, random_mirror=False):
    path = os.path.expandvars('$VOC2007_DIR/ImageSets/Main')

    # First load image list

    fn = os.path.join(path, '{}.txt'.format(which))
    imgids = np.genfromtxt(fn, dtype=np.int32)#[:100]
    C = np.zeros((imgids.size, len(VOC_CLASSES)), dtype=np.float32)

    for c, cls_name in enumerate(VOC_CLASSES):
        fn = os.path.join(path, '{}_{}.txt'.format(cls_name, which))
        data = np.genfromtxt(fn)#[:100]
        C[:, c] = data[:, 1]

    # Convert to 0.0, 0.5, 1.0
    C = (C + 1) / 2

    filenames = [os.path.expandvars('$VOC2007_DIR/JPEGImages/{:06d}.jpg').format(imgid)
            for imgid in imgids]

    rs = np.random.RandomState(1234)

    def load_image(imgid):
        fn = os.path.expandvars('$VOC2007_DIR/JPEGImages/{:06d}.jpg').format(imgid)
        img = dd.image.load(fn)
        if color_transform is not None:
            img = color_transform(img)

        # Resize to smaller side
        img = pad_if_too_small(img, (input_size, input_size))
        if outer_input_size is not None:
            img = dd.image.resize(img, min_side=outer_input_size)
        return img

    if shuffle:
        assert 0
    else:
        all_II = np.arange(len(imgids))
        for i in range(len(imgids)):
            img = load_image(imgids[i])
            h0, w0 = img.shape[0]//2 - input_size//2, img.shape[1]//2 - input_size//2
            ii = []
            ii.append(img[:input_size, :input_size])
            ii.append(img[:input_size, -input_size:])
            ii.append(img[-input_size:, -input_size:])
            ii.append(img[-input_size:, :input_size])
            ii.append(img[h0:h0+input_size, w0:w0+input_size])
            img = img[:, ::-1]
            ii.append(img[:input_size, :input_size])
            ii.append(img[:input_size, -input_size:])
            ii.append(img[-input_size:, -input_size:])
            ii.append(img[-input_size:, :input_size])
            ii.append(img[h0:h0+input_size, w0:w0+input_size])
            yield np.array(ii), C[[i]]



def voc2007_classification_generator2(which, batch_size, input_size,
                            outer_input_size,
                            shuffle=True,# seed=0,
                            color_transform=None, random_mirror=False):
    path = os.path.expandvars('$VOC2007_DIR/ImageSets/Main')

    assert which in ['test', 'val']
    imgs, C = dd.io.load('{}.h5'.format(which), ['/data', '/labels'])

    if shuffle:
        rs = np.random.RandomState()
        while True:
            II = rs.randint(len(imgs), size=batch_size)

            ii, cc = imgs[II], C[II]
            if random_mirror and rs.randint(2) == 1:
                ii = ii[:, :, ::-1]
            yield ii, cc
    else:
        for i in range(len(imgs)//batch_size):
            ss = np.s_[i*batch_size:(i+1)*batch_size]
            yield imgs[ss], C[ss]
"""
def voc_seg_trainval_batching(tr_path, vl_path, pick_val, batch_size,
                              input_size, epochs=None, shuffle=True, min_after_dequeue=100,
                              seed=0, pad255=False, num_threads=1):
    assert seed is not None, ("Seed must be specified, to synchronize "
                              "images and segmentation maps")


    tr_image, tr_seg, tr_imgshape, tr_imgname = _voc_seg_load_file(tr_path, epochs=epochs, shuffle=shuffle, seed=seed)
    vl_image, vl_seg, vl_imgshape, vl_imgname = _voc_seg_load_file(vl_path, epochs=epochs, shuffle=shuffle, seed=seed)

    if 1:
        image = tf.cond(pick_val, lambda: vl_image, lambda: tr_image)
        imgshape = tf.cond(pick_val, lambda: vl_imgshape, lambda: tr_imgshape)
        imgname = tf.cond(pick_val, lambda: vl_imgname, lambda: tr_imgname)
        seg = tf.cond(pick_val, lambda: vl_seg, lambda: tr_seg)
    else:
        image = vl_image
        imgshape = vl_imgshape
        imgname = vl_imgname
        seg = vl_seg

    #seg = tf.cast(tf.image.decode_png(seg_content, channels=1), tf.int32)
    if pad255:
        seg += 1

    pad_image = pad_to_ensure_size(image, input_size, input_size)
    pad_seg = pad_to_ensure_size(seg, input_size, input_size)

    fixed_image = tf.random_crop(pad_image, (input_size, input_size, 3), seed=seed)
    fixed_image = tf.cast(fixed_image, tf.float32) / 255.0

    if pad255:
        fixed_seg = tf.random_crop(tf.cast(pad_seg, tf.int32), (input_size, input_size, 1), seed=seed)
        fixed_seg = ((fixed_seg + 255) % 256)
    else:
        fixed_seg = tf.random_crop(pad_seg, (input_size, input_size, 1), seed=seed)
        fixed_seg = tf.cast(fixed_seg, tf.int32)

    capacity = min_after_dequeue + 3 * batch_size
    if shuffle:
        batch_image, batch_seg, batch_imgshape, batch_imgname = tf.train.shuffle_batch(
                [fixed_image, fixed_seg, imgshape, imgname], batch_size=batch_size, capacity=capacity,
                min_after_dequeue=min_after_dequeue, num_threads=num_threads)
    else:
        batch_image, batch_seg, batch_imgshape, batch_imgname  = tf.train.batch(
                [fixed_image, fixed_seg, imgshape, imgname], batch_size=batch_size, capacity=capacity,
                num_threads=num_threads)

    batch_seg = tf.squeeze(batch_seg, [3])

    return batch_image, batch_seg, batch_imgshape, batch_imgname
"""

def _load_mnist(section="training", offset=0, count=None, ret='xy',
               x_dtype=np.float64, y_dtype=np.int64, path=None):
    """
    Loads MNIST files into a 3D numpy array.

    You have to download the data separately from [MNIST]_. It is recommended
    to set the environment variable ``MNIST_DIR`` to point to the folder where
    you put the data, so that you don't have to select path. On a Linux+bash
    setup, this is done by adding the following to your ``.bashrc``::

        export MNIST_DIR=/path/to/mnist

    Parameters
    ----------
    section : str
        Either "training" or "testing", depending on which section you want to
        load.
    offset : int
        Skip this many samples.
    count : int or None
        Try to load this many samples. Default is None, which loads until the
        end.
    ret : str
        What information to return. See return values.
    x_dtype : dtype
        Type of samples. If ``np.uint8``, intensities lie in {0, 1, ..., 255}.
        If a float type, then intensities lie in [0.0, 1.0].
    y_dtype : dtype
        Integer type to store labels.
    path : str
        Path to your MNIST datafiles. The default is ``None``, which will try
        to take the path from your environment variable ``MNIST_DIR``. The data
        can be downloaded from http://yann.lecun.com/exdb/mnist/.

    Returns
    -------
    images : ndarray
        Image data of shape ``(N, 28, 28)``, where ``N`` is the number of
        images. Returned if ``ret`` contains ``'x'``.
    labels : ndarray
        Array of size ``N`` describing the labels. Returned if ``ret``
        contains ``'y'``.

    Examples
    --------
    Assuming that you have downloaded the MNIST database and set the
    environment variable ``$MNIST_DIR`` point to the folder, this will load all
    images and labels from the training set:

    >>> images, labels = ag.io.load_mnist('training')  # doctest: +SKIP

    Load 100 samples from the testing set:

    >>> sevens = ag.io.load_mnist('testing', offset=200, count=100,
                                  ret='x') # doctest: +SKIP

    """

    # The files are assumed to have these names and should be found in 'path'
    files = {
        'training': ('train-images-idx3-ubyte',
                     'train-labels-idx1-ubyte',
                     60000),
        'testing': ('t10k-images-idx3-ubyte',
                    't10k-labels-idx1-ubyte',
                    10000),
    }

    if count is None:
        count = files[section][2] - offset

    if path is None:
        try:
            path = os.environ['MNIST_DIR']
        except KeyError:
            raise ValueError("Unspecified path requires the environment"
                             "variable $MNIST_DIR to be set")

    try:
        images_fname = os.path.join(path, files[section][0])
        labels_fname = os.path.join(path, files[section][1])
    except KeyError:
        raise ValueError("Data set must be 'testing' or 'training'")

    returns = ()
    if 'x' in ret:
        with open(images_fname, 'rb') as fimg:
            magic_nr, size, d0, d1 = struct.unpack(">IIII", fimg.read(16))
            fimg.seek(offset * d0 * d1, 1)
            images_raw = array("B", fimg.read(count * d0 * d1))

        images = np.asarray(images_raw, dtype=x_dtype).reshape(-1, d0, d1)
        if x_dtype == np.uint8:
            pass  # already this type
        elif x_dtype in (np.float16, np.float32, np.float64):
            images /= 255.0
        else:
            raise ValueError("Unsupported value for x_dtype")

        returns += (images,)

    if 'y' in ret:
        with open(labels_fname, 'rb') as flbl:
            magic_nr, size = struct.unpack(">II", flbl.read(8))
            flbl.seek(offset, 1)
            labels_raw = array("b", flbl.read(count))

        labels = np.asarray(labels_raw)

        returns += (labels,)

    if len(returns) == 1:
        return returns[0]  # Don't return a tuple of one
    else:
        return returns

def mnist_batching(batch_size, subset='training', input_size=None,
        num_threads=1):
    xraw, y = _load_mnist(subset, x_dtype=np.float32, y_dtype=np.int32)

    if input_size is None or input_size == xraw.shape[1]:
        x = xraw
    else:
        x = np.zeros((xraw.shape[0], input_size, input_size, 1), dtype=np.float32)
        w = (input_size - xraw.shape[1]) // 2
        x[:, w:w+xraw.shape[1], w:w+xraw.shape[2]] = xraw[..., np.newaxis]

    min_after_dequeue = 10
    capacity = min_after_dequeue * 3 + batch_size
    x1, y1 = tf.train.slice_input_producer([x, y], shuffle=True)
    batch_x, batch_y = tf.train.shuffle_batch([x1, y1], batch_size=batch_size,
            capacity=capacity, min_after_dequeue=min_after_dequeue,
            num_threads=num_threads)
    return batch_x, batch_y
