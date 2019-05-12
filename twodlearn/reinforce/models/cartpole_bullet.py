import os
import numpy as np
from pybullet_envs.robot_bases import MJCFBasedRobot
from twodlearn.reinforce.models.bullet_env import SingleRobotBulletEnv


class CartpoleRobot(MJCFBasedRobot):
    swingup = True

    @property
    def initial_state(self):
        return (np.array([0.0, 3.1415, 0.0, 0.0]) if self.swingup
                else np.array([0.0, 0.0, 0.0, 0.0]))

    def __init__(self):
        path = os.path.dirname(os.path.realpath(__file__))
        model_xml = os.path.join(path, 'cartpole.xml')
        print('path: {}'.format(path))
        MJCFBasedRobot.__init__(self,
                                model_xml=model_xml,
                                robot_name='cartpole',
                                action_dim=1,
                                obs_dim=4,
                                self_collision=True)

    def robot_specific_reset(self):
        self.pole = self.parts["pole"]
        self.slider = self.jdict["slider"]
        self.j1 = self.jdict["hinge"]
        u = self.np_random.uniform(low=-.1, high=.1)
        self.j1.reset_current_position(
            u if not self.swingup else 3.1415 + u, 0)
        self.j1.set_motor_torque(0)

    def apply_action(self, a):
        assert np.isfinite(a).all(),\
            'provided action is not finite'
        self.slider.set_motor_torque(200 * float(np.clip(a[0], -1, +1)))

    def calc_state(self):
        theta, theta_dot = self.j1.current_position()
        x, x_dot = self.slider.current_position()
        state = np.array([x, theta, x_dot, theta_dot])
        assert np.isfinite(state).all(),\
            'state is not finite'
        return state


class CartpoleEnv(SingleRobotBulletEnv):
    @property
    def initial_state(self):
        return self.robot.initial_state

    def __init__(self, frame_skip=4):
        super(CartpoleEnv, self).__init__(BulletRobot=CartpoleRobot,
                                          frame_skip=frame_skip)
