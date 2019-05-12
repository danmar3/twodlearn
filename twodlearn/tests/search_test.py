from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import unittest
import twodlearn as tdl
import twodlearn.bayesnet
import tensorflow as tf
import twodlearn.recurrent


class LayersTest(unittest.TestCase):
    def test_findinstance(self):
        n_states = 15
        model = tdl.bayesnet.NormalModel(
            batch_shape=[None, n_states],
            loc=tdl.stacked.StackedLayers(
                layers=[tdl.bayesnet.AffineNormalLayer(
                            units=20, tolerance=1e-5),
                        tf.keras.layers.Activation(tf.nn.softplus),
                        # tf.keras.layers.Dropout(rate=0.1),
                        tdl.bayesnet.AffineNormalLayer(
                            units=20, tolerance=1e-5),
                        tf.keras.layers.Activation(tf.nn.softplus),
                        # tf.keras.layers.Dropout(rate=0.1),
                        tdl.bayesnet.AffineNormalLayer(
                            units=n_states, tolerance=1e-5)]))
        instances = tdl.core.find_instances(
            model, tdl.bayesnet.LinearNormalLayer)
        assert len(instances) == 3
        assert instances == set([model.loc.layers[0], model.loc.layers[2],
                                 model.loc.layers[4]])

    def test_findinstance2(self):
        units = [20, 20]
        n_states = 10
        n_inputs = 3
        loc_model = tdl.stacked.StackedLayers()
        for n_units in units:
            loc_model.add(tdl.bayesnet.AffineNormalLayer(
                units=n_units, tolerance=1e-5))
            loc_model.add(tf.keras.layers.Activation(tf.nn.softplus))
            # tf.keras.layers.Dropout(rate=0.1)
        loc_model.add(tdl.bayesnet.AffineNormalLayer(
            units=n_states, tolerance=1e-5))

        dbnn_model = tdl.bayesnet.NormalModel(
            batch_shape=[None, n_states],
            loc=loc_model)

        state_model = tdl.stacked.StackedLayers()
        state_model.add(tf.keras.layers.Concatenate())
        state_model.add(dbnn_model)
        cell = tdl.recurrent.StateSpaceDense(
            input_shape=[None, n_inputs],
            state_shape=[None, n_states],
            state_model=tdl.bayesnet.recurrent
                           .normal_residual_wrapper(state_model)
            )
        instances = tdl.core.find_instances(
            cell, tdl.bayesnet.LinearNormalLayer)
        assert instances == tdl.core.find_instances(
            loc_model, tdl.bayesnet.LinearNormalLayer)


if __name__ == "__main__":
    unittest.main()
