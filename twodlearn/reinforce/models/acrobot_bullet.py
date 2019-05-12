import os
import numpy as np
from pybullet_envs.robot_bases import MJCFBasedRobot
from twodlearn.reinforce.models.bullet_env import SingleRobotBulletEnv


class AcrobotRobot(MJCFBasedRobot):
    @property
    def gear_ratio(self):
        return self._gear_ratio

    @gear_ratio.setter
    def gear_ratio(self, value):
        self._gear_ratio = value

    @property
    def initial_state(self):
        return np.array([3.1415, 0.0, 0.0, 0.0])

    def __init__(self):
        path = os.path.dirname(os.path.realpath(__file__))
        model_xml = os.path.join(path, 'acrobot.xml')
        MJCFBasedRobot.__init__(self,
                                model_xml=model_xml,
                                robot_name='acrobot',
                                action_dim=1,
                                obs_dim=4,
                                self_collision=True)
        self.gear_ratio = 80

    def robot_specific_reset(self):
        self.pole1 = self.parts["pole1"]
        self.pole2 = self.parts["pole2"]
        self.j1 = self.jdict["hinge1"]
        self.j2 = self.jdict["hinge2"]
        u = self.np_random.uniform(low=-.1, high=.1)
        self.j1.reset_current_position(3.1415 + u, 0)
        self.j1.set_motor_torque(0)

    def apply_action(self, a):
        assert np.isfinite(a).all(),\
            'provided action is not finite'
        self.j2.set_motor_torque(
            self.gear_ratio * float(np.clip(a[0], -1, +1)))

    def calc_state(self):
        q1, dq1 = self.j1.current_position()
        q2, dq2 = self.j2.current_position()
        state = np.array([q1, q2, dq1, dq2])
        assert np.isfinite(state).all(),\
            'state is not finite'
        return state


class AcrobotEnv(SingleRobotBulletEnv):
    @property
    def initial_state(self):
        return self.robot.initial_state

    def __init__(self, frame_skip=2):
        super(AcrobotEnv, self).__init__(BulletRobot=AcrobotRobot,
                                         frame_skip=frame_skip)
