from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import unittest
import twodlearn as tdl
import tensorflow as tf


class LayersTest(unittest.TestCase):
    def test_get_trainable1(self):
        layer = tf.keras.layers.Conv2D(
            kernel_size=[5, 5], filters=5, input_shape=(None, 28, 28, 1))
        input = tf.keras.layers.Input(shape=(28, 28, 1), name='the_input')
        # layer.build(input_shape=(None,28,28,1))
        output = layer(input)
        assert len(tdl.core.get_trainable(layer)) == 2

    def test_get_trainable2(self):
        layer = tdl.convnet.Conv2DLayer(kernel_size=[5, 5], filters=5)
        input = tf.keras.layers.Input(shape=(28, 28, 1), name='the_input')
        output = layer(input)
        assert len(tdl.core.get_trainable(layer)) == 2

    def test_get_trainable3(self):
        layer = tdl.convnet.Conv2DLayer(
            kernel_size=[5, 5], filters=5)
        layer.build(input_shape=(None, 28, 28, 1))
        assert len(tdl.core.get_trainable(layer)) == 2

    def test_get_trainable4(self):
        layer = tdl.convnet.Conv2DLayer(
            kernel_size=[5, 5], filters=5, input_shape=(None, 28, 28, 1))
        layer.build()
        assert len(tdl.core.get_trainable(layer)) == 2
        assert layer.built is True


if __name__ == "__main__":
    unittest.main()
