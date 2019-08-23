from __future__ import division
from __future__ import print_function
import typing
import warnings
import numpy as np
import tensorflow as tf
import twodlearn as tdl
import twodlearn.feedforward as tdlf


@tdl.core.create_init_docstring
class PairwiseL2(tdl.core.TdlOp):
    @property
    def x1(self):
        ''' :math:`x_1` input for the PairwiseL2 operation
        :math:`(||x_1 - x_2||^2)`
        '''
        return self._x1

    @property
    def x2(self):
        ''' :math:`x_2` input for the PairwiseL2 operation
        :math:`(||x_1 - x_2||^2)`
        '''
        return self._x2

    @property
    def value(self):
        return self._value

    def __init__(self, x1, x2, name='PairwiseL2'):
        super(PairwiseL2, self).__init__(name=name)
        self._x1 = x1
        self._x2 = x2
        with tf.name_scope(self.scope):
            x1_norm = tf.reduce_sum(x1**2, axis=1, keepdims=True)
            x2_norm = tf.reduce_sum(x2**2, axis=1, keepdims=True)
            self._value = (x1_norm
                           - 2*tf.matmul(x1, tf.transpose(x2))
                           + tf.transpose(x2_norm))


@tdl.core.create_init_docstring
class ConcatOnes(tdl.core.TdlModel):
    ''' Concat ones at the right of input matrix '''
    class ConcatOnesBasis(tdl.core.OutputModel):
        @tdl.core.LazzyProperty
        def linop(self):
            return tf.linalg.LinearOperatorFullMatrix(self.value)

        def evaluate(self, w):
            ''' evaluate the product with vector w '''
            return self.linop.matvec(w)

        def __call__(self, w):
            return self.evaluate(w)

    @tdl.core.ModelMethod(['value'], ['inputs'], ConcatOnesBasis)
    def evaluate(self, object, inputs):
        x = tf.convert_to_tensor(inputs)
        assert x.shape.ndims == 2, 'ConcatOnes is only defined for matrices'
        n_batch = (x.shape[0] if x.shape[0].value is not None
                   else tf.shape(x)[0])
        return tf.concat([x, tf.ones(shape=[n_batch, 1])], axis=1)

    def __call__(self, x):
        return self.evaluate(x)


@tdl.core.create_init_docstring
class QuadraticFeatures(tdl.core.TdlModel):
    ''' Concat ones at the right of input matrix '''
    @tdl.core.InputArgument
    def include_inputs(self, value):
        return value

    def __init__(self, include_inputs=False, name=None, **kargs):
        super(QuadraticFeatures, self).__init__(
            include_inputs=include_inputs, name=name, **kargs)

    @tdl.core.ModelMethod(['value', 'square_batches'], ['inputs'])
    def evaluate(self, object, inputs):
        x = tf.convert_to_tensor(inputs)
        assert x.shape.ndims == 2, \
            'QuadraticFeatures is only defined for matrices'
        n_batch = (x.shape[0] if x.shape[0].value is not None
                   else tf.shape(x)[0])
        xl = tf.expand_dims(x, axis=-1)
        xr = tf.expand_dims(x, axis=1)
        square_batches = tf.matmul(xl, xr)
        output = tf.reshape(square_batches, [n_batch, -1])
        if self.include_inputs:
            output = tf.concat([output, x], axis=1)
        return output, square_batches

    def __call__(self, x):
        return self.evaluate(x)


@tdl.core.create_init_docstring
class GaussianKernel(tdl.TdlModel):
    '''gaussian kernel calculation between matrices X1 and X2

    X1 is a matrix, whose rows represent samples.
    X2 is a matrix, whose rows represent samples.

    .. math::

        K(i,j) =
        (f_{scale}^2) \exp(-0.5 (x_1(i)-x_2(j))^T (l^{-2}) I (x_1(i)-x_2(j)) )
        + y_{scale}^2 I

    '''
    _default_name = 'GaussianKernel'
    _parameters = ['l_scale', 'f_scale', 'y_scale']
    @tdl.core.InputArgument
    def input_shape(
        self, value: typing.Union[typing.List[int], typing.Tuple[int],
                                  tf.TensorShape]) -> tf.TensorShape:
        if not isinstance(value, tf.TensorShape):
            value = tf.TensorShape(value)
        return value

    @tdl.core.SimpleParameter
    def l_scale(self, value, AutoType=None):
        if AutoType is None:
            AutoType = tdl.constrained.PositiveVariableExp
        if value is None:
            value = AutoType(0.3, name='l_scale')
        elif isinstance(value, (int, float, np.ndarray)):
            value = AutoType(value, name='l_scale')
        return value

    @tdl.core.SimpleParameter
    def f_scale(self, value, AutoType=None):
        if AutoType is None:
            AutoType = tdl.constrained.PositiveVariableExp
        if value is None:
            value = AutoType(1.0, name='f_scale')
        elif isinstance(value, (int, float, np.ndarray)):
            value = AutoType(value, name='f_scale')
        return value

    @tdl.core.SimpleParameter
    def y_scale(self, value):
        return value

    @tdl.InputArgument
    def feature_ndims(self, value):
        return value

    @tdl.ModelMethod(['value', 'diff'], ['x1', 'x2'])
    def evaluate(self, object, x1, x2):
        x1 = tf.convert_to_tensor(x1)/self.l_scale
        x2 = tf.convert_to_tensor(x2)/self.l_scale
        x1 = tf.expand_dims(x1, -(self.feature_ndims + 1))
        x2 = tf.expand_dims(x2, -(self.feature_ndims + 2))
        g = tdl.core.array.reduce_sum_rightmost((x1-x2)**2, self.feature_ndims)
        k_value = self.f_scale**2 * tf.exp(-g/2)
        if self.y_scale is not None:
            warnings.warn("y_scale for caussian kernel is deprecated",
                          DeprecationWarning)
            k_dim = tf.shape(k_value)
            k_value = k_value + self.y_scale**2 * tf.eye(k_dim[0])
        return k_value, g

    @tdl.ModelMethod(['value'], ['x1', 'x2'])
    def batch_eval(self, object, x1, x2):
        x1 = tf.convert_to_tensor(x1)/self.l_scale
        x2 = tf.convert_to_tensor(x2)/self.l_scale
        g = tdl.core.array.reduce_sum_rightmost((x1-x2)**2, self.feature_ndims)
        k = self.f_scale**2 * tf.exp(-g/2)
        if self.y_scale is not None:
            warnings.warn("y_scale for caussian kernel is deprecated",
                          DeprecationWarning)
            k_dim = tf.shape(k)
            k = k + self.y_scale**2 * tf.eye(k_dim[0])
        return k

    def __init__(self, l_scale=None, f_scale=None, y_scale=None,
                 feature_ndims=1, **kargs):
        super(GaussianKernel, self).__init__(l_scale=l_scale,
                                             f_scale=f_scale,
                                             y_scale=y_scale,
                                             feature_ndims=feature_ndims,
                                             **kargs)


def polynomial_kernel():
    pass
