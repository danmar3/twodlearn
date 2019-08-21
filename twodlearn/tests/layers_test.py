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

    def test_nested_vars(self):
        @tdl.core.create_init_docstring
        class GMM(tdl.core.layers.Layer):
            n_dims = tdl.core.InputArgument.required(
                'n_dims', doc='dimensions of the GMM model')
            n_components = tdl.core.InputArgument.required(
                 'n_components', doc='number of mixture components')

            @tdl.core.SubmodelInit(lazzy=True)
            def components(self, trainable=True, tolerance=1e-5):
                tdl.core.assert_initialized(
                    self, 'components', ['n_components', 'n_dims'])
                components = [
                    tdl.core.SimpleNamespace(
                      loc=tf.Variable(tf.zeros(self.n_dims),
                                      trainable=trainable),
                      scale=tdl.constrained.PositiveVariable(
                          tf.ones(self.n_dims),
                          tolerance=tolerance,
                          trainable=trainable))
                    for k in range(self.n_components)]
                return components

            @tdl.core.SubmodelInit(lazzy=True)
            def logits(self, trainable=True, tolerance=1e-5):
                tdl.core.assert_initialized(self, 'weights', ['n_components'])
                return tf.Variable(tf.zeros(self.n_components),
                                   trainable=trainable)

        test1 = GMM(n_dims=10, n_components=20)
        test1.build()
        assert len(tdl.core.get_variables(test1)) == test1.n_components*2 + 1
        assert len(tdl.core.get_trainable(test1)) == test1.n_components*2 + 1
        test2 = GMM(n_dims=10, n_components=20,
                    logits={'trainable': False}).build()
        assert len(tdl.core.get_variables(test2)) == test2.n_components*2 + 1
        assert len(tdl.core.get_trainable(test2)) == test2.n_components*2


if __name__ == "__main__":
    unittest.main()
