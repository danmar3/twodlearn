import os.path
from twodlearn.reinforce.modelica.modelica_model import ModelicaModel
from twodlearn.reinforce.modelica.modelica_env import ModelicaEnv

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))


class VdpModel(ModelicaModel):
    def __init__(self, dt):
        filename = os.path.join(CURRENT_FOLDER, 'VDP.mo')
        super(VdpModel, self).__init__("VDP", filename, dt)


class VdpEnv(ModelicaEnv):
    def __init__(self, dt):
        self._model = VdpModel(dt)
