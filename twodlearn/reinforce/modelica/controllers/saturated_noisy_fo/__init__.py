import os
import collections
import twodlearn.reinforce.modelica.models.first_order as first_order
from twodlearn.reinforce.modelica.modelica_model import ModelicaModel
from twodlearn.reinforce.modelica.modelica_env import ModelicaEnv

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))


class SaturatedNoisyFO(ModelicaModel):
    ModelParameters = collections.namedtuple('ModelParameters',
                                             ['model', 'dist', 'sim'])

    def init_parameters(self):
        params = SaturatedNoisyFO.ModelParameters(
            model=ModelicaModel.ModelicaVariables(
                self.fmu, ['y_min', 'y_max', 'K', 'tau']),
            dist=ModelicaModel.ModelicaVariables(
                self.fmu, ['x_stddev', 'y_stddev']),
            sim=ModelicaModel.ModelicaVariables(
                self.fmu, ['ex.samplePeriod', 'ey.samplePeriod']))
        return params

    def set_parameters(self, y_min=-1.0, y_max=1.0, nominal=None,
                       k=None, tau=None,
                       x_stddev=None, y_stddev=None):
        self.parameters.model.set([y_min, y_max, k, tau])
        self.parameters.dist.set([x_stddev, y_stddev])
        if nominal is None:
            self._nominal_x = [(y_min + y_max)/2.0]
        else:
            self._nominal_x = [nominal]
        self.reset(x0=self._nominal_x)

    def __init__(self, dt):
        filename = os.path.join(CURRENT_FOLDER, 'SaturatedFo.mo')
        super(SaturatedNoisyFO, self).__init__(
            'SaturatedNoisyFO', filename, dt)

    def __call__(self, x=None, t=None):
        u = self._nominal_x/self.parameters.model.data[2]
        self.step(u)
        return self.x.data
