import os
import pytest
docutils = pytest.importorskip("pyfmi")
import unittest
from twodlearn.reinforce import systems

try:
    os.environ['DISPLAY']
    SKIP = False
except KeyError:
    SKIP = True


@pytest.mark.skipif(
    SKIP,
    reason='Cannot connect to X server, if running on a server you can use: '
           'xvfb-run -s "-screen 0 1400x900x24" bash')
class EnvTests(unittest.TestCase):
    def test_cartpole(self):
        plant = systems.Cartpole(render_mode=None, output_dir=None)
        plant.simulate(lambda x, k: [0.5],
                       steps=100)
        plant.env.close()

    def test_cartpole_render(self):
        plant = systems.Cartpole(render_mode='human', output_dir=None)
        plant.simulate(lambda x, k: [0.5],
                       steps=100)
        # plant.env.close()

    def test_cartpole_monitor(self):
        plant = systems.Cartpole(render_mode='human', output_dir='tmp/')
        plant.simulate(lambda x, k: [0.5],
                       steps=100)
        plant.env.close()

    def test_acrobot(self):
        plant = systems.Acrobot(render_mode=None, output_dir=None)
        plant.simulate(lambda x, k: [0.5],
                       steps=100)
        plant.env.close()

    def test_acrobot_render(self):
        plant = systems.Acrobot(render_mode='human', output_dir=None)
        plant.simulate(lambda x, k: [0.5],
                       steps=100)
        plant.env.close()

    def test_acrobot_monitor(self):
        plant = systems.Acrobot(render_mode='human', output_dir='tmp/')
        plant.simulate(lambda x, k: [0.5],
                       steps=100)
        plant.env.close()


if __name__ == "__main__":
    unittest.main()
