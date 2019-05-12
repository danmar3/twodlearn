import math
import unittest
import numpy as np
from twodlearn.reinforce.modelica.models.vdp import VdpModel
from twodlearn.reinforce.modelica.modelica_model import ModelicaModel
from twodlearn.reinforce.modelica.models.first_order import FirstOrderModel


class ModelicaTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
