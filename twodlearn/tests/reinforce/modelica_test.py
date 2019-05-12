import pytest
docutils = pytest.importorskip("pyfmi")
import math
import numpy as np
import twodlearn.debug
from twodlearn.reinforce.modelica.modelica_model import ModelicaModel
from twodlearn.reinforce.modelica.models.vdp import VdpModel
from twodlearn.reinforce.modelica.models.first_order \
    import (FirstOrderModel, NoisyFirstOrderModel)
from twodlearn.reinforce.modelica.models.cstr import CstrModel
from twodlearn.reinforce.modelica.models.combined_cycle import CombinedCycleModel


import unittest
from twodlearn.reinforce import systems


class ModelicaTests(unittest.TestCase):
    def test_vdp_model(self):
        model = VdpModel(dt=1.0)

    def test_firstorder(self):
        dt = 1.0
        u = 10.0
        K = 1.0
        tau = 1.0
        model = FirstOrderModel(dt=dt)
        model.parameters.set([K, tau])

        K = model.fmu.get('K')[0]
        tau = model.fmu.get('tau')[0]

        # test iterative simulation
        n_steps = int(math.ceil(tau/dt))
        for step in range(n_steps):
            model.step([u])
        x_iterative = model.x.data[0]
        np.testing.assert_almost_equal(
            x_iterative/(K*u), 0.6319838440004335, 4)

        # test continuous simulation
        model.fmu.reset()
        model.parameters.set([K, tau])
        model.fmu.simulate(final_time=dt*n_steps, input=('u', lambda t: u))
        x_continuous = model.x.data
        np.testing.assert_almost_equal(
            x_continuous/(K*u), 0.6319838440004335, 4)

        np.testing.assert_almost_equal(x_iterative, x_continuous, 7)

    def test_parameters(self):
        model = NoisyFirstOrderModel(dt=1.5)
        for i in range(10):
            model.step(0.0)
        model.reset()
        # Test if parameters hold between simulations
        K = 1.5
        tau = model.parameters.model.data[1]
        model.parameters.model.set([K, None])

        for i in range(10):
            model.step(0.0)

        params = model.parameters.model.data
        np.testing.assert_almost_equal(K, params[0]),\
            'error setting model parameters'
        np.testing.assert_almost_equal(tau, params[1]),\
            'error setting model parameters'

    def test_parameters2(self):
        model = NoisyFirstOrderModel(dt=1.5)
        for i in range(10):
            model.step(0.0)
        model.reset()
        # Test if parameters hold between resets
        K = 1.5
        tau = model.parameters.model.data[1]
        model.parameters.model.set([K, None])

        for i in range(10):
            model.step(0.0)
        model.reset()

        params = model.parameters.model.data
        np.testing.assert_almost_equal(
            K, params[0], err_msg=' model parameters not held after reset')
        np.testing.assert_almost_equal(
            tau, params[1], err_msg=' model parameters not held after reset')


if __name__ == "__main__":
    unittest.main()
