import unittest
import twodlearn as tdl
import tensorflow as tf


class ResnetTest(unittest.TestCase):
    def test_get_trainable1(self):
        residual = tdl.stacked.StackedLayers(
           layers=[tf.keras.layers.Conv2D(kernel_size=[5, 5], filters=5,
                                          padding='same'),
                   tf.keras.layers.Conv2D(kernel_size=[10, 10], filters=5,
                                          padding='same'),
                   ])
        reslayer = tdl.resnet.ResConv2D(residual=residual)
        input = tf.keras.layers.Input(shape=(28, 28, 1), name='the_input')
        # output_shape
        output_shape = reslayer.compute_output_shape(input.shape)
        assert output_shape.as_list() == [None, 28, 28, 5]

        # output
        output = reslayer(input)
        assert output.shape.as_list() == [None, 28, 28, 5]

        # trainable
        assert len(tdl.core.get_trainable(reslayer)) == 6

    def test_get_trainable2(self):
        residual = tdl.stacked.StackedLayers(
           layers=[tf.keras.layers.Conv2D(kernel_size=[5, 5], filters=5,
                                          padding='same'),
                   tf.keras.layers.Conv2D(kernel_size=[10, 10], filters=1,
                                          padding='same'),
                   ])
        reslayer = tdl.resnet.ResConv2D(residual=residual)
        input = tf.keras.layers.Input(shape=(28, 28, 1), name='the_input')
        # output_shape
        output_shape = reslayer.compute_output_shape(input.shape)
        assert output_shape.as_list() == [None, 28, 28, 1]

        # output
        output = reslayer(input)
        assert output.shape.as_list() == [None, 28, 28, 1]

        # trainable
        assert len(tdl.core.get_trainable(reslayer)) == 4


if __name__ == "__main__":
    unittest.main()
