from __future__ import division
from __future__ import print_function

import typing
import collections
import tensorflow as tf
import twodlearn as tdl
import twodlearn.core.exceptions


def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
    """Determines output length of a convolution given input length.
    Arguments:
        input_length: integer.
        filter_size: integer.
        padding: one of "same", "valid", "full", "causal"
        stride: integer.
        dilation: dilation rate, integer.
    Returns:
        The output length (integer).
    """
    if input_length is None:
        return None
    assert padding in {'same', 'valid', 'full', 'causal'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding in ['same', 'causal']:
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride


@tdl.core.create_init_docstring
class Conv2DLayer(tdl.core.Layer):
    @tdl.core.InputArgument
    def kernel_size(self, value):
        '''Size of the convolution kernels. Must be a tuple/list of two
        elements (height, width)
        '''
        if value is None:
            raise tdl.core.exceptions.ArgumentNotProvided(self, 'kernel_size')
        if isinstance(value, collections.Iterable):
            assert len(value) == 2, 'kernel_size must have a length of 2'
        if isinstance(value, int):
            value = [value, value]
        return value

    @tdl.core.InputArgument
    def strides(self, value):
        '''Convolution strides. Default is (1, 1).'''
        if value is None:
            value = (1, 1)
        if isinstance(value, collections.Iterable):
            assert len(value) == 2, 'strides must have a length of 2'
        return value

    @tdl.core.InputArgument
    def input_shape(self, value):
        '''Input tensor shape: (n_samples, n_rows, n_cols, n_channels).'''
        if value is None:
            raise tdl.core.exceptions.ArgumentNotProvided(self)
        if len(value) != 4:
            raise ValueError('input_shape must specify four values: '
                             '(n_samples, n_rows, n_cols, n_channels)')
        if not isinstance(value, tf.TensorShape):
            value = tf.TensorShape(value)
        return value

    @tdl.core.InputArgument
    def filters(self, value):
        '''Number of filters (int), equal to the number of output maps.'''
        if value is None:
            raise tdl.core.exceptions.ArgumentNotProvided(self)
        if not isinstance(value, int):
            raise TypeError('filters must be an integer')
        return value

    @tdl.core.ParameterInit
    def kernel(self, initializer=None, trainable=True, **kargs):
        tdl.core.assert_initialized(
            self, 'kernel', ['kernel_size', 'input_shape'])
        if initializer is None:
            initializer = tf.keras.initializers.glorot_uniform()
        return self.add_weight(
            name='kernel',
            initializer=initializer,
            shape=[self.kernel_size[0], self.kernel_size[1],
                   self.input_shape[-1].value, self.filters],
            trainable=trainable,
            **kargs)

    @tdl.core.ParameterInit
    def bias(self, initializer=None, trainable=True, use_bias=True, **kargs):
        tdl.core.assert_initialized(self, 'bias', ['filters', 'use_bias'])
        if (use_bias and self.use_bias) is False:
            return None
        if initializer is None:
            initializer = tf.keras.initializers.zeros()
        return self.add_weight(
            name='bias',
            initializer=initializer,
            shape=[self.filters],
            trainable=trainable,
            **kargs)

    @tdl.core.InputArgument
    def use_bias(self, value: typing.Union[bool, None]):
        tdl.core.assert_initialized_if_available(
            self, 'use_bias', ['bias', 'filters'])
        if value is None:
            if tdl.core.is_property_initialized(self, 'bias'):
                value = self.bias is not None
            else:
                value = True
        assert isinstance(value, bool), 'use_bias should be bool'
        if value is True:
            if tdl.core.is_property_initialized(self, 'bias'):
                assert self.bias is not None, \
                    'use_bias is True, but bias was set to None'
        if value is False:
            if tdl.core.is_property_initialized(self, 'bias'):
                assert self.bias is None, \
                    'use_bias is False, but bias was not set to None'
        return value

    @tdl.core.InputArgument
    def padding(self, value):
        """Padding for the convolution. It could be either 'valid' or 'same'.
        Default is 'valid'.
        """
        if value is None:
            value = 'valid'
        assert value in ('valid', 'same'),\
            'padding should be either same or valid'
        return value

    @tdl.core.InputArgument
    def dilation_rate(self, value):
        '''Defaults to (1, 1).'''
        if value is None:
            value = (1, 1)
        if isinstance(value, int):
            value = (value, value)
        if not (isinstance(value, collections.Iterable) and len(value) == 2):
            raise ValueError('dilation_rate must be an iterable of length 2')
        value = tuple((v if isinstance(v, int) else int(v))
                      for v in value)
        return value

    def compute_output_shape(self, input_shape=None):
        if input_shape is None:
            tdl.core.assert_initialized(
                self, 'copute_output_shape',
                ['input_shape', 'kernel_size', 'padding', 'strides',
                 'dilation_rate'])
            input_shape = self.input_shape
        input_shape = tf.TensorShape(input_shape).as_list()
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        return tf.TensorShape([input_shape[0]] + new_space + [self.filters])

    def call(self, inputs, *args, **kargs):
        inputs = tf.convert_to_tensor(inputs)
        if not tdl.core.is_property_initialized(self, 'input_shape'):
            self.input_shape = inputs.shape
        tdl.core.assert_initialized(
            self, 'call', ['kernel', 'bias', 'strides', 'padding'])
        conv = tf.nn.conv2d(
            inputs, self.kernel,
            strides=[1, self.strides[0], self.strides[1], 1],
            padding=self.padding.upper(),
            dilations=[1, self.dilation_rate[0], self.dilation_rate[1], 1])
        if self.bias is not None:
            conv = conv + self.bias
        return conv


@tdl.core.create_init_docstring
class Conv2DTranspose(Conv2DLayer):
    @tdl.core.ParameterInit
    def kernel(self, initializer=None, trainable=True, **kargs):
        tdl.core.assert_initialized(
            self, 'kernel', ['kernel_size', 'input_shape'])
        if initializer is None:
            initializer = tf.keras.initializers.glorot_uniform()
        return self.add_weight(
            name='kernel',
            initializer=initializer,
            shape=[self.kernel_size[0], self.kernel_size[1],
                   self.filters, self.input_shape[-1].value],
            trainable=trainable,
            **kargs)

    @tdl.core.InputArgument
    def output_padding(self, value):
        if isinstance(value, (list, tuple)):
            assert len(value) == 2, 'kernel_size must have a length of 2'
        else:
            value = (value, value)
        return value

    @staticmethod
    def transpose_output_lenght(
            input_length, filter_size, padding,
            output_padding=None, stride=0, dilation=1):
        assert padding in {'same', 'valid', 'full'}
        if input_length is None:
            return None
        # Get the dilated kernel size
        filter_size = filter_size + (filter_size - 1) * (dilation - 1)
        # Infer length if output padding is None, else compute the exact length
        if output_padding is None:
            if padding == 'valid':
                length = input_length * stride + max(filter_size - stride, 0)
            elif padding == 'full':
                length = input_length * stride - (stride + filter_size - 2)
            elif padding == 'same':
                length = input_length * stride
        else:
            if padding == 'same':
                pad = filter_size // 2
            elif padding == 'valid':
                pad = 0
            elif padding == 'full':
                pad = filter_size - 1
            length = ((input_length - 1) * stride + filter_size - 2 * pad +
                      output_padding)
        return length

    def _compute_output_shape(self, input_shape):
        # bypass eager iterable error
        batch, height, width, depth = [input_shape[i] for i in range(4)]

        if self.output_padding is None:
            output_padding = (None, None)
        else:
            output_padding = self.output_padding

        new_h = self.transpose_output_lenght(
            height, self.kernel_size[0], padding=self.padding,
            output_padding=output_padding[0], stride=self.strides[0],
            dilation=self.dilation_rate[0])
        new_w = self.transpose_output_lenght(
            width, self.kernel_size[1], padding=self.padding,
            output_padding=output_padding[1], stride=self.strides[1],
            dilation=self.dilation_rate[1])
        return (batch, new_h, new_w, self.filters)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        assert input_shape.ndims == 4, 'provided shape is not four dimensional'
        input_shape = input_shape.as_list()
        return tf.TensorShape(self._compute_output_shape(input_shape))

    def call(self, inputs, *args, **kargs):
        inputs = tf.convert_to_tensor(inputs)
        if not tdl.core.is_property_initialized(self, 'input_shape'):
            self.input_shape = inputs.shape
        tdl.core.assert_initialized(
            self, 'call', ['kernel', 'bias', 'strides', 'padding'])
        output_shape = self._compute_output_shape(tf.shape(inputs))
        output_shape = tf.stack(output_shape)
        conv = tf.keras.backend.conv2d_transpose(
            inputs,
            self.kernel,   # tf.transpose(self.kernel, perm=[0, 1, 3, 2]),
            output_shape,
            strides=tuple(self.strides),
            padding=self.padding,
            dilation_rate=self.dilation_rate
            )
        if self.bias is not None:
            conv = conv + self.bias
        # output fix shape
        if inputs.shape[1:].is_fully_defined():
            conv.set_shape(self._compute_output_shape(inputs.shape))
        return conv


class Conv1x1Proj(tdl.core.Layer):
    @tdl.core.InputArgument
    def units(self, value: int):
        '''Number of output units (int).'''
        if value is None:
            raise tdl.core.exceptions.ArgumentNotProvided(self)
        if not isinstance(value, int):
            raise TypeError('units must be an integer')
        return value

    @tdl.core.ParameterInit
    def kernel(self, initializer=None, trainable=True, **kargs):
        tdl.core.assert_initialized(
            self, 'kernel', ['units', 'input_shape'])
        if initializer is None:
            initializer = tf.keras.initializers.glorot_uniform()
        return self.add_weight(
            name='kernel',
            initializer=initializer,
            shape=[self.input_shape[-1].value, self.units],
            trainable=trainable,
            **kargs)

    @tdl.core.ParameterInit
    def bias(self, initializer=None, trainable=True, use_bias=True, **kargs):
        tdl.core.assert_initialized(self, 'bias', ['units', 'use_bias'])
        if (use_bias and self.use_bias) is False:
            return None
        if initializer is None:
            initializer = tf.keras.initializers.zeros()
        return self.add_weight(
            name='bias',
            initializer=initializer,
            shape=[self.units],
            trainable=trainable,
            **kargs)

    @tdl.core.InputArgument
    def use_bias(self, value: typing.Union[bool, None]):
        tdl.core.assert_initialized_if_available(self, 'use_bias', ['bias'])
        if value is None:
            if tdl.core.is_property_initialized(self, 'bias'):
                value = self.bias is not None
            else:
                value = True
        assert isinstance(value, bool), 'use_bias should be bool'
        if value is True:
            if tdl.core.is_property_initialized(self, 'bias'):
                assert self.bias is not None, \
                    'use_bias is True, but bias was set to None'
        if value is False:
            if tdl.core.is_property_initialized(self, 'bias'):
                assert self.bias is None, \
                    'use_bias is False, but bias was not set to None'
        return value

    @tdl.core.InputArgument
    def activation(self, value):
        return value

    @tdl.core.Submodel
    def _linop(self, _):
        tdl.core.assert_initialized(self, '_linop', ['kernel'])
        return tf.linalg.LinearOperatorFullMatrix(self.kernel)

    def compute_output_shape(self, input_shape=None):
        input_shape = tf.TensorShape(input_shape)
        output_shape = input_shape[:-1]
        return output_shape.concatenate(self.units)

    def call(self, inputs):
        output = self._linop.matvec(inputs, adjoint=True)
        if self.bias is not None:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_transpose(self, use_bias=None, activation=None, trainable=True):
        tdl.core.assert_initialized(
            self, 'get_transpose', ['kernel', 'bias', 'activation'])
        kargs = dict()
        if use_bias is False or self.bias is None:
            kargs['bias'] = None
        transpose = type(self)(
            units=self.kernel.shape[0].value,
            kernel=tf.transpose(self.kernel),
            activation=activation,
            **kargs)
        if trainable:
            transpose.add_weight(self.kernel)
        return transpose
