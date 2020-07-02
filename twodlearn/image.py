from . import core
import tensorflow as tf


class ImageResize(core.Layer):
    # scale = core.InputArgument.required(
    #     'scale', doc='output image is resized by this scale.')
    size = core.InputArgument.required(
        'size', doc='size of the output image.')
    # data_format = core.SimpleParameter()
    method = core.InputArgument.optional(
        'method',
        doc='Method used for resizing. Options are the same as in '
            'tf.image.resize',
        default=tf.image.ResizeMethod.BILINEAR)
    preserve_aspect_ratio = core.InputArgument.optional(
        'preserve_aspect_ratio',
        doc='Whether to preserve the aspect ratio.',
        default=False)
    antialias = core.InputArgument.optional(
        'antialias',
        doc='Whether to use anti-aliasing filter when downsampling an image.',
        default=False)

    @staticmethod
    def _img_size(shape):
        return shape[1:-1].as_list()

    def compute_output_shape(self, input_shape=None):
        input_shape = tf.TensorShape(input_shape)
        output_shape = input_shape[:1].concatenate(self.size)\
                                      .concatenate(input_shape[-1])
        return output_shape

    def call(self, inputs):
        return tf.compat.v2.image.resize(
            inputs, self.size, method=self.method,
            preserve_aspect_ratio=self.preserve_aspect_ratio,
            antialias=self.antialias)
