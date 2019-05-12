import os.path
import collections
from twodlearn.reinforce.modelica.modelica_model import ModelicaModel
from twodlearn.reinforce.modelica.modelica_env import ModelicaEnv

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))


class CstrModel(ModelicaModel):
    ''' Hicks-Ray Continuously Stirred Tank Reactor (CSTR)
    Doc: https://modelica.org/events/modelica2011/Proceedings/pages/papers/10_2_ID_188_a_fv.pdf
    States = {T: reactor temperature,
              c: reactant concentration}
    Constant inputs =
             {F0: reactant inflow rate,
              c0: reactant concentration,
              T0: reactant temperature}
    Model parameters =
             {r, k0, EdivR, U, rho, Cp, dH, V}
    '''
    ModelParameters = collections.namedtuple('ModelParameters',
                                             ['constants', 'model'])

    def init_x0(self):
        names = [x + '_init' for x in self.x.names]
        return super(CstrModel, self).init_x0(names)

    def init_parameters(self):
        constants = ModelicaModel.ModelicaVariables(self.fmu,
                                                    ['F0', 'c0', 'T0'])

        model = ModelicaModel.ModelicaVariables(self.fmu,
                                                ['r', 'k0', 'EdivR', 'U',
                                                 'rho', 'Cp', 'dH', 'V'])
        return CstrModel.ModelParameters(constants=constants,
                                         model=model)

    def __init__(self, dt):
        filename = os.path.join(CURRENT_FOLDER, 'CSTR.mo')
        super(CstrModel, self).__init__('CSTR.CSTR', filename, dt)


class CstrEnv(ModelicaEnv):
    def __init__(self, dt=1.0):
        self._model = CstrModel(dt)
        super(CstrEnv, self).__init__()
