from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf
import twodlearn as tdl
import twodlearn.recurrent


class RecurrentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.InteractiveSession()

    def test_inputs(self):
        model = tdl.recurrent.Lstm(n_inputs=10, n_outputs=2, n_hidden=[50])
        predict1 = model.evaluate(
            x0={'batch_size': 100, 'AutoType': tf.zeros},
            inputs={'batch_size': 100,
                    'AutoType': lambda shape, **kargs:
                    tdl.variable(tf.zeros(shape=shape), **kargs)},
            n_unrollings=9)

        predict2 = model.evaluate(n_unrollings=9)
        predict2.x0.init(batch_size=100, AutoType=tf.zeros)
        predict2.inputs.init(batch_size=100, AutoType=tf.zeros)

        predict3 = model.evaluate(n_unrollings=predict1.n_unrollings,
                                  inputs=predict1.inputs,
                                  x0=predict1.x0)

        assert len(predict3.outputs) == len(predict3.inputs)
        assert len(predict3.outputs) == len(predict3.states)


if __name__ == "__main__":
    unittest.main()
