import gym
import numpy as np
import pandas as pd
from gym.utils import seeding
from gym import error, spaces, utils
import matplotlib.pyplot as plt


class SignalRender(object):
    @property
    def model(self):
        return self._model

    def do_render(self, data):
        self.ax[0, 0].clear()
        self.ax[1, 0].clear()
        trajectory = self.model.trajectory.as_df()
        trajectory.x.plot(ax=self.ax[0, 0])
        trajectory.u.plot(ax=self.ax[1, 0])
        # plot points
        t = self.model.dt * (np.arange(len(data))+1)
        data = np.array(data)
        self.ax[0, 0].plot(t, data, '+k')
        # self.fig.show()

    def __init__(self, env):
        self._model = env.model
        # n_ax = len(self.model.x)
        n_ax = 2
        self.fig, self.ax = plt.subplots(n_ax, 1,
                                         figsize=(10, 3 * n_ax),
                                         squeeze=False)


class StateRender(object):
    @property
    def model(self):
        return self._model

    def do_render(self, data):
        for i in range(self.ax.shape[0]):
            self.ax[i, 0].clear()
        trajectory = self.model.trajectory.as_df()

        t = self.model.dt * (np.arange(len(data))+1)
        data = np.array(data)
        for i in range(self.ax.shape[0]):
            trajectory.x.iloc[:, i].plot(ax=self.ax[i, 0])
            # plot points
            self.ax[i, 0].plot(t, data[:, i], '+k')
        self.fig.show()

    def __init__(self, env):
        self._model = env.model
        n_ax = len(self.model.x)
        self.fig, self.ax = plt.subplots(n_ax, 1,
                                         figsize=(10, 3 * n_ax),
                                         squeeze=False)


class ModelicaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    @property
    def model(self):
        return self._model

    @property
    def dt(self):
        return self.model.dt

    def __init__(self):
        self.viewer = None
        self.observation_buffer = list()

    def step(self, action):
        self.model.step(action)
        state = self.model.x.data
        self.observation_buffer.append(state)
        reward = 0.0
        done = False
        return np.array(state), reward, done, {}

    def reset(self):
        self.model.reset()
        self.observation_buffer = list()
        return self.model.x.data

    def close(self):
        pass

    def render(self, mode='human'):
        if self.viewer is None:
            #self.viewer = SignalRender(self)
            self.viewer = StateRender(self)
        self.viewer.do_render(self.observation_buffer)
