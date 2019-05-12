from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import unittest
import twodlearn as tdl
import tensorflow as tf


class ParallelTest(unittest.TestCase):
    def test_nest1(self):
        model = tdl.parallel.ParallelLayers(
            layers={'loc': tf.keras.layers.Dense(units=15),
                    'scale': tdl.dense.AffineLayer(units=5)}
        )
        inputs0 = tf.keras.layers.Input((10,))
        inputs1 = tf.keras.layers.Input((9,))
        output = model({'loc': inputs0, 'scale': inputs1})
        assert (output['loc'].shape.as_list() ==
                tf.TensorShape([None, 15]).as_list())
        assert (output['scale'].shape.as_list() ==
                tf.TensorShape([None, 5]).as_list())
        input_shape = model.input_shape
        assert input_shape['loc'].as_list() == inputs0.shape.as_list()
        assert input_shape['scale'].as_list() == inputs1.shape.as_list()

    def test_nest2(self):
        model = tdl.parallel.ParallelLayers(
            layers=[tf.keras.layers.Dense(units=15),
                    tdl.dense.AffineLayer(units=5)]
        )
        inputs0 = tf.keras.layers.Input((10,))
        inputs1 = tf.keras.layers.Input((9,))
        output = model([inputs0, inputs1])
        assert (output[0].shape.as_list() ==
                tf.TensorShape([None, 15]).as_list())
        assert output[1].shape.as_list() == tf.TensorShape([None, 5]).as_list()
        input_shape = model.input_shape
        assert input_shape[0].as_list() == inputs0.shape.as_list()
        assert input_shape[1].as_list() == inputs1.shape.as_list()

    def test_compute_shape(self):
        model = tdl.parallel.ParallelLayers(
            layers=[tf.keras.layers.Dense(units=15),
                    tdl.dense.AffineLayer(units=5)]
        )
        inputs0 = tf.keras.layers.Input((10,))
        inputs1 = tf.keras.layers.Input((9,))
        output_shape = model.compute_output_shape(
            input_shape=[inputs0.shape, inputs1.shape])
        assert (output_shape[0].as_list() ==
                tf.TensorShape([None, 15]).as_list())
        assert (output_shape[1].as_list() ==
                tf.TensorShape([None, 5]).as_list())


if __name__ == "__main__":
    unittest.main()
