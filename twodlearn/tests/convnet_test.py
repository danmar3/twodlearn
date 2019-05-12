import unittest
import numpy as np
import tensorflow as tf
import twodlearn as tdl
import twodlearn.convnet
import twodlearn.bayesnet.bayesnet
import twodlearn.bayesnet.gaussian_process
import twodlearn.templates.bayesnet


class ConvnetTest(unittest.TestCase):
    def test_error(self):
        layer1 = tdl.convnet.Conv2DLayer(kernel_size=[5, 5])
        with self.assertRaises(tdl.core.exceptions.ArgumentNotProvided):
            layer1.kernel.init()

    def test_conv1x1(self):
        layer = tdl.convnet.Conv1x1Proj(
            units=3, activation=tf.keras.layers.ReLU())
        input = np.random.normal(size=(32, 28, 28, 10)).astype(np.float32)
        proj = layer(input)
        assert proj.shape.as_list() == [32, 28, 28, 3]
        assert (proj.shape.as_list() ==
                layer.compute_output_shape(input.shape).as_list())
        layer_t = layer.get_transpose()
        tran = layer_t(proj)
        assert input.shape == tuple(tran.shape.as_list())
        assert tran.shape.as_list() == [32, 28, 28, 10]
        assert (tran.shape.as_list() ==
                layer_t.compute_output_shape(proj.shape).as_list())

        assert ((set(tdl.core.get_trainable(layer)) &
                 set(tdl.core.get_trainable(layer_t))) ==
                set([layer.kernel]))

    def test_conv1x1_bias(self):
        layer = tdl.convnet.Conv1x1Proj(
            units=3, activation=tf.keras.layers.ReLU(),
            use_bias=False)
        assert not tdl.core.is_property_initialized(layer, 'bias')
        input = np.random.normal(size=(32, 28, 28, 10)).astype(np.float32)
        proj = layer(input)
        assert layer.bias is None
        layer2 = tdl.convnet.Conv1x1Proj(
            units=3, activation=tf.keras.layers.ReLU(),
            bias=None)
        assert layer2.use_bias is False


    def test_conv(self):
        with tf.Session().as_default():
            input = tf.convert_to_tensor(
                np.random.normal(size=(32, 28, 28, 10)).astype(np.float32))
            layer_tf = tf.keras.layers.Conv2D(
                filters=15, kernel_size=[5, 5],
                strides=[2, 3], padding='valid', dilation_rate=[1, 1])
            output_tf = layer_tf(input)
            layer_tdl = tdl.convnet.Conv2DLayer(
                filters=15, kernel_size=[5, 5],
                strides=[2, 3], padding='valid', dilation_rate=[1, 1],
                kernel=layer_tf.kernel
            )
            output_tdl = layer_tdl(input)
            tdl.core.initialize_variables(layer_tf)
            tdl.core.initialize_variables(layer_tdl)
            max_error = tf.reduce_max(tf.abs(output_tdl - output_tf)).eval()
            assert max_error < 1e-10

    def test_conv2(self):
        with tf.Session().as_default():
            input = tf.convert_to_tensor(
                np.random.normal(size=(32, 28, 28, 10)).astype(np.float32))
            layer_tf = tf.keras.layers.Conv2D(
                filters=15, kernel_size=[5, 5],
                strides=[2, 3], padding='valid', dilation_rate=[1, 1],
                use_bias=False)
            _ = layer_tf(input)
            layer_tdl = tdl.convnet.Conv2DLayer(
                filters=15, kernel_size=[5, 5],
                strides=[2, 3], padding='valid', dilation_rate=[1, 1],
                use_bias=False
            )
            assert not tdl.core.is_property_initialized(layer_tdl, 'bias')
            _ = layer_tdl(input)
            assert layer_tdl.bias is None
            assert (layer_tf.kernel.shape.as_list() ==
                    layer_tdl.kernel.shape.as_list())

            layer2 = tdl.convnet.Conv2DLayer(
                filters=3, kernel_size=[5, 5],
                strides=[2, 3], padding='valid', dilation_rate=[1, 1],
                bias=None)
            assert layer2.use_bias is False

    def test_convtrans1(self):
        with tf.Session().as_default():
            input = tf.convert_to_tensor(
                np.random.normal(size=(32, 8, 8, 10)).astype(np.float32))
            layer_tf = tf.keras.layers.Conv2DTranspose(
                filters=5,
                kernel_size=[5, 5],
                strides=(2, 2),
                use_bias=True)
            output_tf = layer_tf(input)
            layer_tdl = tdl.convnet.Conv2DTranspose(
                filters=5,
                kernel_size=[5, 5],
                strides=[2, 2],
                use_bias=True)
            output_tdl = layer_tdl(input)
            assert (layer_tf.kernel.shape.as_list() ==
                    layer_tdl.kernel.shape.as_list())
            assert (layer_tf.bias.shape.as_list() ==
                    layer_tdl.bias.shape.as_list())
            assert (output_tf.shape.as_list() == output_tdl.shape.as_list())

    def test_convtrans2(self):
        with tf.Session().as_default():
            input = tf.convert_to_tensor(
                np.random.normal(size=(32, 8, 8, 10)).astype(np.float32))
            layer_tf = tf.keras.layers.Conv2DTranspose(
                filters=5,
                kernel_size=[5, 5],
                strides=(2, 2),
                use_bias=True)
            output_tf = layer_tf(input)
            layer_tdl = tdl.convnet.Conv2DTranspose(
                filters=5,
                kernel_size=[5, 5],
                kernel=layer_tf.kernel,
                strides=[2, 2],
                use_bias=True)
            output_tdl = layer_tdl(input)
            tdl.core.initialize_variables(layer_tf)
            tdl.core.initialize_variables(layer_tdl)
            max_error = tf.reduce_max(tf.abs(output_tdl - output_tf)).eval()
            assert max_error < 1e-10

    def test_convtrans3(self):
        with tf.Session().as_default():
            input = tf.placeholder(tf.float32, shape=[None, 8, 8, 10])
            layer_tdl = tdl.convnet.Conv2DTranspose(
                filters=5,
                kernel_size=[5, 5],
                strides=[2, 2],
                use_bias=True)
            output_tdl = layer_tdl(input)
            tdl.core.initialize_variables(layer_tdl)
            dynamic_shape = tf.shape(output_tdl).eval(
                {input: np.random.normal(size=[32, 8, 8, 10])})
            assert all(dynamic_shape[1:] == output_tdl.shape[1:].as_list())
            assert output_tdl.shape.as_list() == [None, 19, 19, 5]


if __name__ == "__main__":
    unittest.main()
