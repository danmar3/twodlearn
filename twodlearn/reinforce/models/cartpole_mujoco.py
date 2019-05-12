import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class CartpoleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, frame_skip=2):
        utils.EzPickle.__init__(self)
        # mujoco_env.MujocoEnv.__init__(self, 'inverted_pendulum.xml', 2)
        current_path = os.path.dirname(
            os.path.abspath(__file__))
        # frame_skip: how many simulation "steps" are performed between
        #             steps
        mujoco_env.MujocoEnv.__init__(
            self, os.path.join(current_path, 'cartpole.xml'),
            frame_skip=frame_skip)

    def _step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all()  # and (np.abs(ob[1]) <= .5)
        done = not notdone
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + \
            self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01) + \
            np.array([0.0, 3.1416])
        qvel = self.init_qvel + \
            self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.model.data.qpos,
                               self.model.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = v.model.stat.extent * 2.5
