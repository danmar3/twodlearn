import numpy as np
import tensorflow as tf


def is_scalar(value):
    if isinstance(value, (int, float)):
        return True
    elif isinstance(value, np.ndarray):
        return value.ndim == 0
    return False


def mesh2array(mesh):
    ''' convert a 2d mesh created with np.meshgrid to an matrix
        representation '''
    x1 = mesh[0]
    x2 = mesh[1]
    return np.concatenate(
            [np.reshape(x1, (x1.shape[0] * x1.shape[1], 1)),
             np.reshape(x2, (x2.shape[0] * x2.shape[1], 1))],
            1)


class Np2dMesh(object):
    ''' 2d mesh created with np.meshgrid'''
    @property
    def asarray(self):
        return mesh2array(self._mesh)

    @property
    def asmesh(self):
        return self._mesh

    def vect2mesh(self, array):
        x1 = self._mesh[0]
        x2 = self._mesh[1]
        return np.reshape(array, [x1.shape[0], x2.shape[1]])

    def __init__(self, x1, x2):
        self._mesh = np.meshgrid(x1, x2)


def reduce_sum_rightmost(x, ndims):
    ''' Return tensor with right-most ndims summed'''
    x = tf.convert_to_tensor(x)
    if x.shape.ndims is not None:
        axis = tf.range(x.shape.ndims - ndims, x.shape.ndims)
    else:
        axis = tf.range(tf.rank(x) - ndims, tf.rank(x))
    return tf.reduce_sum(x, axis=axis)


def add_diagonal_shift(matrix, shift):
    ''' Returns matrix + shift*I '''
    diag_plus_shift = tf.matrix_diag_part(matrix) + shift
    return tf.matrix_set_diag(matrix, diag_plus_shift)


def is_square_matrix(array):
    ''' Returns if array is a square matrix '''
    return (array.shape.ndims == 2 and
            array.shape[0].value == array.shape[1].value)


def transpose_rightmost(x):
    ''' transpose the rightmost dimensions '''
    x = tf.convert_to_tensor(x)
    if x.shape.ndims is not None:
        ndims = x.shape.ndims
        if ndims == 2:
            return tf.transpose(x)
        else:
            left_axis = tf.range(ndims - 2)
            right_axis = tf.convert_to_tensor([ndims-1, ndims-2])
            perm = tf.concat([left_axis, right_axis], axis=0)
    else:
        ndims = tf.rank(x)
        left_axis = tf.range(tf.rank(x) - 2)
        right_axis = tf.convert_to_tensor([ndims-1, ndims-2])
        perm = tf.concat(left_axis, right_axis, axis=0)
    return tf.transpose(x, perm)


def np_dtype(value):
    return tf.convert_to_tensor(value).dtype.as_numpy_dtype


def expand_dims_left(x, ndims):
    '''add ndim dimensions on the left'''
    if ndims == 0:
        return x
    elif ndims < 0:
        raise ValueError('invalid number of dimensions to expand ({})'
                         ''.format(ndims))

    x = tf.convert_to_tensor(x)
    ndims = (ndims.value if hasattr(ndims, 'value')
             else ndims)
    if x.shape.is_fully_defined() and isinstance(ndims, int):
        new_shape = tf.TensorShape([1]*ndims)\
                      .concatenate(x.shape)
    else:
        x_shape = tf.shape(x)
        new_shape = tf.concat([tf.ones(shape=ndims, dtype=x_shape.dtype),
                               x_shape],
                              axis=0)
    return tf.reshape(x, shape=new_shape)


def expand_dims_left_like(x, y):
    '''add dimensions on the left to match the number of dimensions in y'''
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    ndims = y.shape.ndims - x.shape.ndims
    assert ndims >= 0, 'x must have less dimensions than y'
    return expand_dims_left(x, ndims=ndims)
