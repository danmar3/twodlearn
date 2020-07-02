import unittest
import twodlearn as tdl
import tensorflow as tf
from twodlearn import image


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

        # no resize
        assert reslayer.upsample is None
        assert reslayer.downsample is None
        assert reslayer.resize is None

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

    def test_downsample(self):
        residual = tdl.stacked.StackedLayers(
           layers=[
               tf.keras.layers.Conv2D(
                   kernel_size=[5, 5], filters=5, padding='same'),
               tf.keras.layers.Conv2D(
                   kernel_size=[10, 10], filters=5, strides=2, padding='same'),
               ])
        reslayer = tdl.resnet.ResConv2D(residual=residual)
        input = tf.keras.layers.Input(shape=(28, 28, 1), name='the_input')
        # output_shape
        output_shape = reslayer.compute_output_shape(input.shape)
        assert output_shape.as_list() == [None, 14, 14, 5]

        output = reslayer(input)
        assert not tdl.core.is_property_provided(reslayer, 'downsample')
        assert reslayer.upsample is None
        assert isinstance(reslayer.downsample, tf.keras.layers.MaxPool2D)
        assert reslayer.resize is None

        # nearest
        reslayer2 = tdl.resnet.ResConv2D(
            residual=residual, downsample={'size': 2}, resize_method='nearest')
        output_shape = reslayer2.compute_output_shape(input.shape)
        assert output_shape.as_list() == [None, 14, 14, 5]

        output = reslayer2(input)
        assert tdl.core.is_property_provided(reslayer2, 'downsample')
        assert reslayer2.upsample is None
        assert isinstance(reslayer2.downsample, image.ImageResize)
        assert reslayer2.resize is None

        # resize
        reslayer3 = tdl.resnet.ResConv2D(
            residual=residual, resize={'size': [14, 14]},
            resize_method='bicubic')
        assert not tdl.core.is_property_provided(reslayer3, 'upsample')
        assert not tdl.core.is_property_provided(reslayer3, 'downsample')
        assert tdl.core.is_property_provided(reslayer3, 'resize')
        output_shape = reslayer3.compute_output_shape(input.shape)
        assert output_shape.as_list() == [None, 14, 14, 5]

        output = reslayer3(input)
        assert reslayer3.upsample is None
        assert reslayer3.downsample is None
        assert isinstance(reslayer3.resize, image.ImageResize)

        # resize2
        residual = tdl.stacked.StackedLayers(
           layers=[
               tf.keras.layers.Conv2D(
                   kernel_size=[5, 5], filters=5, padding='valid'),
               tf.keras.layers.Conv2D(
                   kernel_size=[10, 10], filters=5, strides=2, padding='same'),
               ])
        reslayer3 = tdl.resnet.ResConv2D(
            residual=residual, resize={'size': [12, 12]},
            resize_method='bicubic')
        assert not tdl.core.is_property_provided(reslayer3, 'upsample')
        assert not tdl.core.is_property_provided(reslayer3, 'downsample')
        assert tdl.core.is_property_provided(reslayer3, 'resize')
        output_shape = reslayer3.compute_output_shape(input.shape)
        assert output_shape.as_list() == [None, 12, 12, 5]

        output = reslayer3(input)
        assert reslayer3.upsample is None
        assert reslayer3.downsample is None
        assert isinstance(reslayer3.resize, image.ImageResize)

        # test for error
        reslayer4 = tdl.resnet.ResConv2D(
            residual=residual,
            downsample={'size': 2}, resize={'size': [12, 12]},
            resize_method='bicubic')
        with self.assertRaises(ValueError):
            output = reslayer4(input)


if __name__ == "__main__":
    unittest.main()
