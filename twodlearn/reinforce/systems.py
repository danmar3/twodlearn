import warnings
import numpy as np
import pandas as pd
from time import sleep
from gym import wrappers
import twodlearn.reinforce

try:
    from twodlearn.reinforce.modelica.models.cstr import CstrEnv
except ImportError:
    pass

from .models.cartpole_bullet import CartpoleEnv
from .models.acrobot_bullet import AcrobotEnv


class System(object):
    @property
    def n_states(self):
        return self.env.observation_space.shape[0]

    @property
    def n_sensors(self):
        return self.env.observation_space.shape[0]

    @property
    def n_actuators(self):
        return self.env.action_space.shape[0]

    @property
    def EnvClass(self):
        return self._EnvClass

    @property
    def dt(self):
        return self.env.dt

    def __init__(self, EnvClass, render_mode=None, output_dir=None):
        self._EnvClass = EnvClass
        self.init_env(render_mode=render_mode, output_dir=output_dir)

        self._sensor_noise_std = 0.0

    def init_env(self, render_mode=None, output_dir=None):
        if hasattr(self, 'env'):
            self.env.close()
        if hasattr(self, 'monitor'):
            if self.monitor is not None:
                self.monitor.close()

        if (output_dir is not None) and (render_mode is None):
            warnings.warn('Render_mode not specified while monitoring '
                          'the simulation. render_mode set to \'human\'')
            render_mode = 'human'

        self._render_mode = render_mode
        self._output_dir = output_dir
        # setup environment
        self.env = self.EnvClass()
        if (twodlearn.reinforce.USING_BULLET and
                render_mode is not None):
            self.env.render(mode=render_mode)
        self.env.reset()

        # setup monitor environment
        self.monitor = None
        if output_dir is not None:
            self.monitor = wrappers.Monitor(self.env,
                                            output_dir,
                                            force=True)
            self.monitor.reset()

    def set_sensor_noise(self, std=0.0):
        self._sensor_noise_std = std

    def sensor_model(self, x_k):
        ''' calculates the sensor output from the system's state '''
        if self._sensor_noise_std > 0.0:
            raise ValueError("not yet implemented")
        else:
            return x_k

    def do_render(self, env, mode='human'):
        if twodlearn.reinforce.USING_BULLET:
            if mode == 'human':
                # rendering not required
                sleep(self.env.dt)
        elif twodlearn.reinforce.USING_MUJOCO:
            env.render()
        else:
            raise NotImplementedError(
                'Only Mujoco and Bullet environments are available')

    def simulate(self, policy, steps, render=False):
        # initialize lists
        x = np.zeros((steps, self.n_states))
        y = np.zeros((steps, self.n_sensors))
        y_diff = np.zeros((steps, self.n_sensors))
        u = np.zeros((steps, self.n_actuators))

        x_k = self.env.reset()
        y_k = self.sensor_model(x_k)
        for k in range(steps):
            # evaluate policy
            u_k = policy(x_k, k)
            # save current state vector
            x[k, :] = x_k
            y[k, :] = y_k
            u[k, :] = u_k
            # apply step
            if render:
                if self.monitor is None:
                    x_k1, reward, done, info = self.env.step(u_k)
                    self.do_render(self.env)
                else:
                    x_k1, reward, done, info = self.monitor.step(u_k)
            else:
                x_k1, reward, done, info = self.env.step(u_k)

            y_k1 = self.sensor_model(x_k1)
            # save sensor output
            y_diff[k, :] = y_k1 - y_k
            # finish for next episode
            x_k = x_k1
            y_k = y_k1
            # check end conditions
            if done:
                #    env.monitor.flush(force=True)
                print("Episode finished after {} timesteps".format(k + 1))
                break
        data = \
            pd.DataFrame(np.concatenate([x, y, u, y_diff], axis=1),
                         index=self.dt * np.arange(steps),
                         columns=(['x{}'.format(i)
                                   for i in range(self.n_states)] +
                                  ['y{}'.format(i)
                                   for i in range(self.n_sensors)] +
                                  ['u{}'.format(i)
                                   for i in range(self.n_actuators)] +
                                  ['y_diff{}'.format(i)
                                   for i in range(self.n_sensors)]))
        return data


class ModelicaSystem(System):
    def init_env(self, render_mode=None, output_dir=None):
        # setup environment
        self.env = self.EnvClass()
        self.env.reset()

    def simulate(self, policy, steps, render=False):
        # initialize lists
        x = np.zeros((steps, self.n_states))
        y = np.zeros((steps, self.n_sensors))
        y_diff = np.zeros((steps, self.n_sensors))
        u = np.zeros((steps, self.n_actuators))

        x_k = self.env.reset()
        y_k = self.sensor_model(x_k)
        for k in range(steps):
            # evaluate policy
            u_k = policy(x_k, k)
            # save current state vector
            x[k, :] = x_k
            y[k, :] = y_k
            u[k, :] = u_k
            # apply step
            x_k1, reward, done, info = self.env.step(u_k)
            y_k1 = self.sensor_model(x_k1)
            if render:
                self.env.render()
            # save sensor output
            y_diff[k, :] = y_k1 - y_k
            # finish for next episode
            x_k = x_k1
            y_k = y_k1
            # check end conditions
            if done:
                print("Episode finished after {} timesteps".format(k + 1))
                break
        data = \
            pd.DataFrame(np.concatenate([x, y, u, y_diff], axis=1),
                         index=self.dt * np.arange(steps),
                         columns=(['x{}'.format(i)
                                   for i in range(self.n_states)] +
                                  ['y{}'.format(i)
                                   for i in range(self.n_sensors)] +
                                  ['u{}'.format(i)
                                   for i in range(self.n_actuators)] +
                                  ['y_diff{}'.format(i)
                                   for i in range(self.n_sensors)]))
        return data


class Cartpole(System):
    @property
    def initial_state(self):
        return self.env.initial_state

    def __init__(self, render_mode=None, output_dir=None):
        def EnvClass():
            return CartpoleEnv(frame_skip=2)
        super(Cartpole, self).__init__(EnvClass, render_mode, output_dir)


class Acrobot(System):
    @property
    def initial_state(self):
        return self.env.initial_state

    def __init__(self, render_mode=None, output_dir=None):
        super(Acrobot, self).__init__(AcrobotEnv, render_mode, output_dir)


class Cstr(ModelicaSystem):
    @property
    def initial_state(self):
        return self._initial_state

    @property
    def n_states(self):
        return len(self.env.model.x)

    @property
    def n_sensors(self):
        return len(self.env.model.x)

    @property
    def n_actuators(self):
        return len(self.env.model.u)

    def __init__(self, dt=1.0, render_mode=None, output_dir=None):
        def Env():
            return CstrEnv(dt)
        super(Cstr, self).__init__(Env, render_mode, output_dir)

        # initial state
        self.env.model.reset()
        self._initial_state = self.env.model.x0.data
