import math
import tensorflow as tf
import twodlearn as tdl
import twodlearn.core


def compute_fans(kernel_shape):
    '''Computes the number of input and output units for a weight shape.
    Args:
        shape: Integer shape tuple or TF tensor shape.
    Returns:
        A tuple of scalars (fan_in, fan_out).
    '''
    shape = kernel_shape
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1.
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out


class SumFanConstant(object):
    '''Initializer that creates a matrix fill with a constant value equal
    to the SumFan rule:
        output = sqrt(scale**2/(fain_in  + fan_out)) eye(shape)
    '''
    @property
    def scale(self):
        return self._scale

    def __init__(self, scale=1.0):
        self._scale = scale

    def __call__(self, shape):
        shape = tf.TensorShape(shape)
        if shape.ndims != 2:
            raise ValueError('GlorotConstant is only defined for two '
                             'dimensional matrices.')
        shape = shape.as_list()
        fan_in = shape[0]
        fan_out = shape[1]
        value = math.sqrt((self.scale*self.scale)/(fan_in + fan_out))
        return tf.keras.initializers.Constant(value=value)(shape=shape)


class KernelInitializer(object):
    def __init__(self, scale=1.0, shape=None):
        self.scale = scale
        self.shape = shape


class FrobeniusNormal(KernelInitializer):
    '''Initializer that creates a matrix sampled from
       Normal(0, scale**2/(fan_in * fan_out))
    '''
    def __call__(self, shape=None):
        if shape is None:
            shape = self.shape
        if isinstance(shape, tf.TensorShape):
            shape = shape.as_list()
        sigma = math.sqrt((self.scale**2)/(shape[-2] * shape[-1]))
        return tf.truncated_normal(shape, stddev=sigma)


class SumNormal(KernelInitializer):
    '''Initializer that creates a matrix sampled from
       Normal(0, scale**2/(fan_in + fan_out))
    '''
    def __call__(self, shape=None):
        if shape is None:
            shape = self.shape
        if isinstance(shape, tf.TensorShape):
            shape = shape.as_list()
        sigma = math.sqrt((self.scale**2)/(shape[-2] + shape[-1]))
        return tf.truncated_normal(shape, stddev=sigma)


class MaxNormal(KernelInitializer):
    '''Initializer that creates a matrix sampled from
       Normal(0, scale**2/(max(fan_in, fan_out)))
    '''
    def __call__(self, shape=None):
        if shape is None:
            shape = self.shape
        if isinstance(shape, tf.TensorShape):
            shape = shape.as_list()
        sigma = math.sqrt((self.scale**2)/max(shape[-2], shape[-1]))
        return tf.truncated_normal(shape, stddev=sigma)


class SingularNormal(KernelInitializer):
    '''Initializer that creates a matrix '''
    def __call__(self, shape=None):
        if shape is None:
            shape = self.shape
        if isinstance(shape, tf.TensorShape):
            shape = shape.as_list()

        N = min(shape[-1], shape[-2])
        M = max(shape[-1], shape[-2])
        t = (N - 1) / (M - 1)
        t = 1 - math.exp(-10 * t)
        alpha = math.sqrt(math.minimum(N, M) *
                          (self.scale * ((1 - t) + 0.3 * t)))
        sigma = math.sqrt((alpha**2) / (N * M))
        return tf.truncated_normal(shape, stddev=sigma)
