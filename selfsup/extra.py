
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow import nn
import tensorflow as tf

def normalize_moments(counts, mean_ss, variance_ss, shift, name=None):
  """Calculate the mean and variance of based on the sufficient statistics.

  Args:
    counts: A `Tensor` containing a the total count of the data (one value).
    mean_ss: A `Tensor` containing the mean sufficient statistics: the (possibly
      shifted) sum of the elements to average over.
    variance_ss: A `Tensor` containing the variance sufficient statistics: the
      (possibly shifted) squared sum of the data to compute the variance over.
    shift: A `Tensor` containing the value by which the data is shifted for
      numerical stability, or `None` if no shift was performed.
    name: Name used to scope the operations that compute the moments.

  Returns:
    Two `Tensor` objects: `mean` and `variance`.
  """
  with tf.variable_scope(name, "normalize", [counts, mean_ss, variance_ss, shift]):
    divisor = math_ops.reciprocal(counts, name="divisor")
    if shift is not None:
      shifted_mean = math_ops.multiply(mean_ss, divisor, name="shifted_mean")
      mean = math_ops.add(shifted_mean, shift, name="mean")
    else:  # no shift.
      shifted_mean = math_ops.multiply(mean_ss, divisor, name="mean")
      mean = shifted_mean
    variance = math_ops.subtract(math_ops.multiply(variance_ss, divisor),
                                 math_ops.square(shifted_mean),
                                 name="variance")
  return (mean, variance)

def moments(x, axes, shift=None, name=None, keep_dims=False):
  """Calculate the mean and variance of `x`.

  The mean and variance are calculated by aggregating the contents of `x`
  across `axes`.  If `x` is 1-D and `axes = [0]` this is just the mean
  and variance of a vector.

  Note: for numerical stability, when shift=None, the true mean
  would be computed and used as shift.

  When using these moments for batch normalization (see
  `tf.nn.batch_normalization`):

   * for so-called "global normalization", used with convolutional filters with
     shape `[batch, height, width, depth]`, pass `axes=[0, 1, 2]`.
   * for simple batch normalization pass `axes=[0]` (batch only).

  Args:
    x: A `Tensor`.
    axes: Array of ints.  Axes along which to compute mean and
      variance.
    shift: A `Tensor` containing the value by which to shift the data for
      numerical stability, or `None` in which case the true mean of the data is
      used as shift. A shift close to the true mean provides the most
      numerically stable results.
    name: Name used to scope the operations that compute the moments.
    keep_dims: produce moments with the same dimensionality as the input.

  Returns:
    Two `Tensor` objects: `mean` and `variance`.
  """
  #with ops.name_scope(name, "moments", [x, axes, shift]):
  if 1:
    # The dynamic range of fp16 is too limited to support the collection of
    # sufficient statistics. As a workaround we simply perform the operations
    # on 32-bit floats before converting the mean and variance back to fp16
    y = math_ops.cast(x, dtypes.float32) if x.dtype == dtypes.float16 else x
    if shift is None:
      # Compute true mean while keeping the dims for proper broadcasting.
      shift = array_ops.stop_gradient(
          math_ops.reduce_mean(y, axes, keep_dims=True))
    else:
      shift = math_ops.cast(shift, y.dtype)
    counts, m_ss, v_ss, shift = nn.sufficient_statistics(
        y, axes, shift=shift, keep_dims=keep_dims, name=name+'_statistics')
    # Reshape shift as needed.
    shift = array_ops.reshape(shift, array_ops.shape(m_ss))
    shift.set_shape(m_ss.get_shape())
    with ops.control_dependencies([counts, m_ss, v_ss]):
      mean, variance = normalize_moments(counts, m_ss, v_ss, shift, name=name)
      if x.dtype == dtypes.float16:
        return (math_ops.cast(mean, dtypes.float16),
                math_ops.cast(variance, dtypes.float16))
      else:
        return (mean, variance)


