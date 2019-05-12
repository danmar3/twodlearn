import os.path
import collections
from twodlearn.reinforce.modelica.modelica_model import ModelicaModel
from twodlearn.reinforce.modelica.modelica_env import ModelicaEnv

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))


class CombinedCycleModel(ModelicaModel):
    ''' combined cycle power plant
    Doc: https://modelica.org/events/modelica2011/Proceedings/pages/papers/10_2_ID_188_a_fv.pdf
    states = {
         MX(superheater_gas_side.gas_out.T),
         MX(evaporator_gas_side.gas_out.T),
         MX(economizer_gas_side.gas_out.T),
         MX(economizer.wat_liq_out.T),
         MX(evaporator.alpha),
         MX(evaporator.p),
         MX(superheater.wat_vap_out.T),
         MX(turbineShaft.T__2),
         MX(PI.x)]}
    -- Initial equations --
    superheater_gas_side.gas_out.T = superheater_gas_side.gas_out.Tstart
    evaporator_gas_side.gas_out.T = evaporator_gas_side.gas_out.Tstart
    economizer_gas_side.gas_out.T = economizer_gas_side.gas_out.Tstart
    economizer.wat_liq_out.T = economizer.wat_liq_out.Tstart
    evaporator.alpha = evaporator.alphastart
    evaporator.p = evaporator.pstart
    superheater.wat_vap_out.T = superheater.wat_vap_out.Tstart
    turbineShaft.T__2 = turbineShaft.Tstartext
    PI.x = PI.x_start

    '''
    # ModelParameters = collections.namedtuple('ModelParameters',
    #                                          ['constants', 'model'])

    def init_x0(self):
        # names = ['superheater_gas_side.gas_out.Tstart',
        #         'evaporator_gas_side.gas_out.Tstart',
        #         'economizer_gas_side.gas_out.Tstart',
        #         'economizer.wat_liq_out.Tstart',
        #         'evaporator.alphastart',
        #         'evaporator.pstart',
        #         'superheater.wat_vap_out.Tstart',
        #         'turbineShaft.Tstartext',
        #         'PI.x_start']
        names = [x.replace('.', '__') + '_init' for x in self.x.names]
        return super(CombinedCycleModel, self).init_x0(names)

    def init_parameters(self):
        # constants = ModelicaModel.ModelicaVariables(self.fmu,
        #                                             ['F0', 'c0', 'T0'])

        # model = ModelicaModel.ModelicaVariables(self.fmu,
        #                                         ['r', 'k0', 'EdivR', 'U',
        #                                          'rho', 'Cp', 'dH', 'V'])
        # CstrModel.ModelParameters(constants=constants,
        #                                  model=model)
        return None

    def __init__(self, dt=1.0):
        filename = os.path.join(CURRENT_FOLDER, 'CombinedCycle.mo')
        super(CombinedCycleModel, self).__init__(
            'Tdl.CC0D_explicit_x0', filename, dt)


class CombinedCycleEnv(ModelicaEnv):
    def __init__(self, dt=1.0):
        self._model = CombinedCycleModel(dt)
