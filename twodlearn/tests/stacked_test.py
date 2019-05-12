import unittest
import twodlearn as tdl
import tensorflow as tf
import tensorflow.keras.layers as tf_layers
import functools
import operator


@tdl.core.create_init_docstring
class TransposeLayer(tdl.core.Layer):
    @tdl.core.SubmodelInit
    def conv(self, units, kernels, strides, padding='same'):
        return tf_layers.Conv2DTranspose(
                units, kernels, strides=strides,
                padding=padding,
                use_bias=False)

    @tdl.core.Submodel
    def activation(self, value):
        return value

    def call(self, inputs):
        output = self.conv(inputs)
        if self.activation is not None:
            output = self.activation(output)
        return output


class StackedTest(unittest.TestCase):
    def test_get_trainable1(self):
        init_shape = (4, 4, 128)
        model = tdl.stacked.StackedLayers()
        model.add(tdl.dense.LinearLayer(
            units=functools.reduce(operator.mul, init_shape, 1)))
        model.add(tf_layers.Reshape(init_shape))
        model.add(TransposeLayer(
            conv={'units': 64, 'kernels': (5, 5), 'strides': (2, 2)}))
        model.add(TransposeLayer(
            conv={'units': 32, 'kernels': (5, 5), 'strides': (2, 2)}))

        input = tf.random.normal([32, 100])
        output = model(input)
        assert len(tdl.core.get_trainable(model)) == 3


if __name__ == "__main__":
    unittest.main()
