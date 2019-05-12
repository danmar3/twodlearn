import unittest
from twodlearn.reinforce import systems


class EnvTests(unittest.TestCase):
    def test_cartpole(self):
        plant = systems.Cartpole(render_mode=None, output_dir=None)
        plant.simulate(lambda x, k: [0.5],
                       steps=100)

    def test_cartpole_render(self):
        plant = systems.Cartpole(render_mode='human', output_dir=None)
        plant.simulate(lambda x, k: [0.5],
                       steps=100)

    def test_cartpole_monitor(self):
        plant = systems.Cartpole(render_mode='human', output_dir='tmp/')
        plant.simulate(lambda x, k: [0.5],
                       steps=100)

    def test_acrobot(self):
        plant = systems.Acrobot(render_mode=None, output_dir=None)
        plant.simulate(lambda x, k: [0.5],
                       steps=100)

    def test_acrobot_render(self):
        plant = systems.Acrobot(render_mode='human', output_dir=None)
        plant.simulate(lambda x, k: [0.5],
                       steps=100)

    def test_acrobot_monitor(self):
        plant = systems.Acrobot(render_mode='human', output_dir='tmp/')
        plant.simulate(lambda x, k: [0.5],
                       steps=100)


if __name__ == "__main__":
    unittest.main()
