from . import core
from . import convnet
from .image import ImageResize
import tensorflow as tf
import tensorflow.keras.layers as tf_layers


def tf_compat_dim(value):
    if hasattr(value, 'value'):
        return value.value


class ResLayer(core.Layer):
    residual = core.Submodel.required(
        'residual', doc='residual model')
    projection = core.Submodel.required(
         'projection', doc='projection model')


class ResConv2D(ResLayer):
    def _resize_type(self, input_shape, output_shape):
        input_size = input_shape[1:-1].as_list()
        output_size = output_shape[1:-1].as_list()
        if output_size == input_size:
            return None, None
        elif all(oi % ii == 0 for ii, oi in zip(input_size, output_size)):
            size = list(oi // ii for ii, oi in zip(input_size, output_size))
            return 'upsample', size
        elif all(ii % oi == 0 for ii, oi in zip(input_size, output_size)):
            size = list(ii // oi for ii, oi in zip(input_size, output_size))
            return 'downsample', size
        else:
            return 'resize', output_size

    resize_method = core.InputArgument.optional(
        'resize_method',
        doc='Method used for resizing when residual has different shape than '
            'input.',
        default=None)
    data_format = core.InputArgument.optional(
        'data_format',
        doc='a string, either "channels_last" (default) or "channels_first".',
        default=None)

    DEFAULT_RESIZE = {'upsample': 'nearest',
                      'downsample': 'max_pool',
                      'resize': 'nearest'}

    def _get_resize_method(self, resize_type):
        if self.resize_method is None:
            return self.DEFAULT_RESIZE[resize_type]
        else:
            return self.resize_method

    @core.SubmodelInit(lazzy=True)
    def upsample(self, size=None, **kargs):
        core.assert_initialized(
            self, 'upsample', ['resize_method', 'input_shape', 'residual'])
        # check if more than one of upsample, downsample or resize was provided
        if sum(core.is_property_provided(self, prop)
               for prop in ('upsample', 'downsample', 'resize')) > 1:
            raise ValueError(
                'can only specify one of: upsample, downsample, resize')
        if core.any_provided(self, ['downsample', 'resize']):
            return None

        output_shape = self.residual.compute_output_shape(
            input_shape=self.input_shape)
        if size is None:
            resize_type, output_size = \
                self._resize_type(self.input_shape, output_shape)
            if resize_type != 'upsample':
                if kargs:
                    raise ValueError('kargs provided when upsample is not '
                                     'executed.')
                return None
            size = output_size
        method = self._get_resize_method('upsample')
        if method in ('nearest, bilinear'):
            return tf_layers.UpSampling2D(
                size=size, data_format=self.data_format, interpolation=method,
                **kargs)
        if method in ('lanczos3', 'lanczos5', 'bicubic', 'gaussian', 'area',
                      'mitchellcubic'):
            if isinstance(size, int):
                size = [size, size]
            size = tf.TensorShape(size)
            size = [i*si for i, si in
                    zip(self.input_shape[1:-1].as_list(), size.as_list())]
            return ImageResize(size=size, method=method, **kargs)
        raise ValueError(
            f'Upsample method {method} not available in {type(self)}')

    @core.SubmodelInit(lazzy=True)
    def downsample(self, size=None, **kargs):
        core.assert_initialized(
            self, 'downsample', ['resize_method', 'input_shape', 'residual'])
        if core.any_provided(self, ['upsample', 'resize']):
            return None

        output_shape = self.residual.compute_output_shape(
            input_shape=self.input_shape)
        if size is None:
            resize_type, output_size = \
                self._resize_type(self.input_shape, output_shape)
            if resize_type != 'downsample':
                if kargs:
                    raise ValueError('kargs provided when downsample is not '
                                     'executed.')
                return None
            size = output_size

        method = self._get_resize_method('downsample')
        if method == 'max_pool':
            return tf_layers.MaxPool2D(
                pool_size=size, strides=size, data_format=self.data_format,
                **kargs)
        elif method == 'avg_pool':
            return tf_layers.AvgPool2D(
                pool_size=size, strides=size, data_format=self.data_format,
                **kargs)
        elif method in ('bilinear', 'nearest', 'lanczos3', 'lanczos5',
                        'bicubic', 'gaussian', 'area', 'mitchellcubic'):
            if isinstance(size, int):
                size = [size, size]
            size = tf.TensorShape(size)
            size = [i//si for i, si in
                    zip(self.input_shape[1:-1].as_list(), size.as_list())]
            return ImageResize(size=output_shape[1:-1], method=method, **kargs)
        raise ValueError(
            f'Downsample method {method} not available in {type(self)}')

    @core.SubmodelInit(lazzy=True)
    def resize(self, size=None, **kargs):
        core.assert_initialized(
            self, 'resize', ['resize_method', 'input_shape', 'residual'])
        if core.any_provided(self, ['upsample', 'downsample']):
            return None

        output_shape = self.residual.compute_output_shape(
            input_shape=self.input_shape)
        if size is None:
            resize_type, output_size = \
                self._resize_type(self.input_shape, output_shape)
            if resize_type != 'resize':
                if kargs:
                    raise ValueError('kargs provided when resize is not '
                                     'executed.')
                return None
            size = output_size

        method = self._get_resize_method('resize')
        return ImageResize(size=size, method=method, **kargs)

    @core.SubmodelInit(lazzy=True)
    def projection(self, units=None, use_bias=False, channel_dim=-1):
        core.assert_initialized(
            self, 'projection', ['input_shape', 'residual'])
        input_shape = self.input_shape
        input_units = tf_compat_dim(input_shape[channel_dim])
        if units is None:
            output_shape = self.residual.compute_output_shape(input_shape)
            output_units = tf_compat_dim(output_shape[channel_dim])
        else:
            output_units = units
        if input_units == output_units:
            return tf.keras.layers.Activation(activation=tf.identity)
        else:
            return convnet.Conv1x1Proj(units=output_units)

    def compute_output_shape(self, input_shape=None):
        input_shape = tf.TensorShape(input_shape)
        output_shape = self.residual.compute_output_shape(
            input_shape=input_shape)
        return output_shape

    def call(self, inputs):
        residual = self.residual(inputs)
        if self.upsample is not None:
            resized = self.upsample(inputs)
        elif self.downsample is not None:
            resized = self.downsample(inputs)
        elif self.resize is not None:
            resized = self.resize(inputs)
        else:
            resized = inputs
        projection = self.projection(resized)
        return projection + residual
