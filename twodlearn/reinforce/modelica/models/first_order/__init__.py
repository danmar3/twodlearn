import os.path
import collections
from twodlearn.reinforce.modelica.modelica_model import ModelicaModel
from twodlearn.reinforce.modelica.modelica_env import ModelicaEnv

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))


class FirstOrderModel(ModelicaModel):
    ''' System with a first-order transfer function:
    T(s) = K/(tau*s + 1)
    Y = X
    '''

    def init_parameters(self):
        return ModelicaModel.ModelicaVariables(self.fmu,
                                               ['K', 'tau'])

    def __init__(self, dt):
        filename = os.path.join(CURRENT_FOLDER, 'FirstOrder.mo')
        super(FirstOrderModel, self).__init__('FirstOrder', filename, dt)


class FirstOrderEnv(ModelicaEnv):
    def __init__(self, dt):
        self._model = FirstOrderModel(dt)


class NoisyFirstOrderModel(ModelicaModel):
    ''' System with a first-order transfer function:
    T(s) = K/(tau*s + 1)
    Y = X
    '''
    ModelParameters = collections.namedtuple('ModelParameters',
                                             ['model', 'dist', 'sim'])

    def init_parameters(self):
        params = NoisyFirstOrderModel.ModelParameters(
            model=ModelicaModel.ModelicaVariables(self.fmu,
                                                  ['K', 'tau']),
            dist=ModelicaModel.ModelicaVariables(self.fmu,
                                                 ['x_stddev', 'y_stddev']),
            sim=ModelicaModel.ModelicaVariables(self.fmu,
                                                ['ex.samplePeriod', 'ey.samplePeriod']))
        return params

    def __init__(self, dt):
        filename = os.path.join(CURRENT_FOLDER, 'NoisyFirstOrder.mo')
        super(NoisyFirstOrderModel, self).__init__(
            'NoisyFirstOrder', filename, dt)


class NoisyFirstOrderEnv(ModelicaEnv):
    def __init__(self, dt):
        self._model = NoisyFirstOrderModel(dt)
