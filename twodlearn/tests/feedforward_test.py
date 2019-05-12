import unittest
import numpy as np
import tensorflow as tf
import twodlearn as tdl
import twodlearn.feedforward as tdlf


class FeedforwardTest(unittest.TestCase):

    def test_mlp(self):
        mlp = tdl.feedforward.MlpNet(n_inputs=10, n_outputs=5,
                                     n_hidden=[20, 20])
        inputs = tf.keras.Input((10,))
        y = mlp(inputs)
        params1 = tdl.core.get_parameters(mlp)
        params2 = set([mlp.layers[0].kernel,
                       mlp.layers[0].bias,
                       mlp.layers[1].kernel,
                       mlp.layers[1].bias,
                       mlp.layers[2].kernel,
                       mlp.layers[2].bias])

        assert params1 == params2,\
            'found parameters do not coiside with expected parameters'

    def test_stacked_name(self):
        mlp = tdl.feedforward.StackedModel()
        l0 = mlp.add(tdlf.DenseLayer(input_shape=10, units=50),
                     name='hidden_0')
        l1 = mlp.add(tdlf.DenseLayer(input_shape=50, units=50),
                     name='hidden_1')
        x = tf.placeholder(shape=[None, 10], dtype=tf.float32)
        y = mlp(x)
        assert y.hidden_0 == y.hidden[0]
        assert y.hidden_0.model == l0
        assert tdl.core.get_context(mlp).initialized

    def test_transpose(self):
        x = np.array([[[1, 2, 3], [4, 5, 6]]])
        y1 = tdlf.Transpose(inputs=x)
        assert y1.value.shape == y1.shape
        assert y1.shape.as_list() == [3, 2, 1]
        y2 = tdlf.Transpose(inputs=x, rightmost=True)
        assert y2.shape.as_list() == [1, 3, 2]

        y3 = tdlf.TransposeLayer(rightmost=True)(x)
        assert y3.shape.as_list() == [1, 3, 2]


if __name__ == "__main__":
    unittest.main()
