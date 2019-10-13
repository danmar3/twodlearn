from . import core
from . import convnet
import tensorflow as tf


def tf_compat_dim(value):
    if hasattr(value, 'value'):
        return value.value


class ResLayer(core.Layer):
    residual = core.Submodel.required(
        'residual', doc='residual model')
    projection = core.Submodel.required(
         'projection', doc='projection model')


class ResConv2D(ResLayer):
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
        projection = self.projection(inputs)
        return projection + residual
