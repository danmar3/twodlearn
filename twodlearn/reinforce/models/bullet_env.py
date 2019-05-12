import pybullet
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.scene_abstract import SingleRobotEmptyScene


class SingleRobotBulletEnv(MJCFBaseBulletEnv):
    @property
    def dt(self):
        return self.robot.scene.dt

    @property
    def frame_skip(self):
        return self._frame_skip

    def __init__(self, BulletRobot, frame_skip=2):
        self._frame_skip = frame_skip
        self.robot = BulletRobot()
        MJCFBaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1

    def create_single_player_scene(self):
        return SingleRobotEmptyScene(gravity=9.8,
                                     timestep=0.0165,
                                     frame_skip=self.frame_skip)

    def reset(self):
        if (self.stateId >= 0):
            # print("InvertedPendulumBulletEnv reset p.restoreState(",self.stateId,")")
            pybullet.restoreState(self.stateId)
        r = MJCFBaseBulletEnv._reset(self)
        if (self.stateId < 0):
            self.stateId = pybullet.saveState()
            # print("InvertedPendulumBulletEnv reset self.stateId=",self.stateId)
        return r

    def calc_reward(self, state, action):
        return 1.0

    def calc_done(self, state, action):
        return False

    def step(self, action):
        self.robot.apply_action(action)
        self.scene.global_step()
        state = self.robot.calc_state()
        done = self.calc_done(state, action)
        reward = self.calc_reward(state, action)
        self.HUD(state, action, done)
        return state, reward, done, {}

    def camera_adjust(self):
        self.camera.move_and_look_at(0, 1.2, 1.0, 0, 0, 0.5)

    def render(self, *args, **kwargs):
        kwargs['close'] = False
        return self._render(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self._close(*args, **kwargs)

    def seed(self, *args, **kwargs):
        return self._seed(*args, **kwargs)
