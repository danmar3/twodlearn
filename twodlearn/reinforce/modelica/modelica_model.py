import sys
import random
import casadi
import warnings
import numpy as np
import collections
import pandas as pd
from pyfmi import load_fmu
from pymodelica import compile_fmu
from pyjmi import transfer_optimization_problem


class SimulationTrajectory(object):
    Trajectory = collections.namedtuple('Trajectory', ['x', 'y', 'u'])

    @property
    def model(self):
        return self._model

    @property
    def dt(self):
        return self.model.dt

    @property
    def records(self):
        return self._records

    def as_df(self):
        time = [ri['time'] for ri in self.records]
        for i in range(1, len(time)):
            time[i] = time[i] + time[i-1][-1]

        def extract_data(names):
            data = [np.stack([time[i][:-1]] + [ri[x][:-1]
                                               for x in names],
                             axis=1)
                    for i, ri in enumerate(self.records)]
            data = np.concatenate(data, axis=0)
            data = pd.DataFrame(data, columns=['time'] + names)
            return data.set_index('time')

        data_x = extract_data(self.model.x.names)
        data_y = extract_data(self.model.y.names)
        data_u = extract_data(self.model.u.names)

        return SimulationTrajectory.Trajectory(x=data_x,
                                               y=data_y,
                                               u=data_u)

    def knots(self):
        raise NotImplementedError('knots have not been implemented')

    def add_record(self, record):
        self.records.append(record)

    def reset(self):
        self._records = list()

    def __init__(self, model):
        self._model = model
        self.reset()


class CasadiModel(object):
    class ImplicitDae(object):
        def blt(self):
            f = casadi.vertsplit(self.residuals)
            diff_and_algebraic = [x.getVar()
                                  for x in self._program.getVariables(self._program.DIFFERENTIATED)]
            diff_and_algebraic += [x.getVar()
                                   for x in self._program.getVariables(self._program.REAL_ALGEBRAIC)]
            blt = np.zeros((len(f), len(diff_and_algebraic)), dtype=np.int8)
            for i, fi in enumerate(f):
                for j, xj in enumerate(diff_and_algebraic):
                    if casadi.dependsOn(fi, [xj]):
                        blt[i, j] = 1
            keep = np.nonzero(blt.sum(0))[0]
            diff_and_algebraic = [diff_and_algebraic[i]
                                  for i in keep]
            blt = blt[:, keep]

            # sorted = np.argsort(blt.sum(0))
            # diff_and_algebraic = [diff_and_algebraic[i]
            #                       for i in sorted]
            # blt = blt[:, sorted]

            return blt, diff_and_algebraic

        def __init__(self, dae):
            self._program = dae.program
            self.residuals = dae.program.getDaeResidual()
            self.f = [eq for eq in casadi.vertsplit(self.residuals)
                      if casadi.dependsOn(eq, dae.dx_dt)]
            self.gx = [eq for eq in casadi.vertsplit(self.residuals)
                       if (not casadi.dependsOn(eq, dae.y)
                           and not casadi.dependsOn(eq, dae.dx_dt))]
            self.gy = [eq for eq in casadi.vertsplit(self.residuals)
                       if casadi.dependsOn(eq, dae.y)]

    def j_dx_dt(self):
        df_dx = casadi.vertcat([casadi.transpose(casadi.jacobian(self.dae, dxi_dt))
                                for dxi_dt in self.dx_dt])
        Jdx_dt = casadi.MXFunction(self.x + self.dx_dt + self.u, [df_dx])
        Jdx_dt.init()
        return df_dx

    def init_f_xu(self, model):
        #dx_dt = casadi.vertcat(self.dx_dt)
        #zeros = casadi.MX.zeros(len(self.dx_dt), 1)
        # fxu = casadi.substitute(-self.implicit.residuals,
        #                        dx_dt, zeros)
        fxu = casadi.substitute(-self.implicit.residuals,
                                self.dx_dt[0], casadi.MX(0))
        for dxi_dt in self.dx_dt[1:]:
            fxu = casadi.substitute(fxu, dxi_dt, casadi.MX(0))

        # substitute parameters
        if self.parameters is not None:
            for i, data_i in enumerate(model._all_parameters.data):
                fxu = casadi.substitute(
                    fxu, self.parameters[i], casadi.MX(data_i))

        Fxu = casadi.MXFunction(self.x + self.u, [fxu])
        Fxu.init()
        return Fxu

    def __init__(self, class_name, filename, modelica_model):
        compiler_options = {'equation_sorting': True,
                            'automatic_tearing': True}
        self.program = transfer_optimization_problem(class_name,
                                                     filename,
                                                     compiler_options=compiler_options,
                                                     accept_model=True)
        self.program.eliminateAlgebraics()

        self.x = [self.program.getVariable(name).getVar()
                  for name in modelica_model.x.names]
        self.y = [self.program.getVariable(name).getVar()
                  for name in modelica_model.y.names]
        self.u = [self.program.getVariable(name).getVar()
                  for name in modelica_model.u.names]
        # self.dx_dt = [self.program.getVariable(name).getMyDerivativeVariable()
        #              for name in self.program.getVariables(self.program.DERIVATIVE)]
        self.dx_dt = [self.program.getVariable(name).getMyDerivativeVariable().getVar()
                      for name in modelica_model.x.names]

        self.algebraic = [g.getVar()
                          for g in self.program.getVariables(self.program.REAL_ALGEBRAIC)]

        if modelica_model._all_parameters is not None:
            self.parameters = [self.program.getVariable(name).getVar()
                               for name in modelica_model._all_parameters.names]
        else:
            self.parameters = None

        self.implicit = self.ImplicitDae(self)

        self.f_xu = self.init_f_xu(modelica_model)


class ModelicaModel(object):
    class ModelicaVariables(object):
        @property
        def fmu(self):
            return self._fmu

        @property
        def names(self):
            return self._names

        @property
        def data(self):
            return [x[0] for x in self.fmu.get(self.names)]

        @property
        def nominal(self):
            return [self.fmu.get_variable_nominal(name)
                    for name in self.names]

        @property
        def max(self):
            return [self.fmu.get_variable_max(name)
                    for name in self.names]

        @property
        def min(self):
            return [self.fmu.get_variable_min(name)
                    for name in self.names]

        def set(self, data):
            data = [(x1 if (x1 is not None) else x2)
                    for x1, x2 in zip(data, self.data)]
            self.fmu.set(self.names, data)

        def __len__(self):
            return len(self.names)

        def __init__(self, fmu, names):
            self._fmu = fmu
            self._names = names

    @property
    def fmu(self):
        return self._fmu

    @property
    def dt(self):
        return self._dt

    @property
    def x0(self):
        ''' Initial state '''
        return self._x0

    def init_x0(self, names=None):
        if names is None:
            names = [x + '_0' for x in self.x.names]
        if all([name in self.fmu.get_model_variables()
                for name in names]):
            x0 = ModelicaModel.ModelicaVariables(self.fmu, names)
            return x0
        else:
            warnings.warn("Unable find initial states parameters "
                          "for {}. Incremental simulation will fail"
                          "".format(self.x.names), RuntimeWarning)
            return None

    @property
    def x(self):
        ''' system's state '''
        return self._x

    def init_x(self):
        names = self.fmu.get_states_list().keys()
        return ModelicaModel.ModelicaVariables(self.fmu, names)

    @property
    def u(self):
        ''' system's exogenous inputs '''
        return self._u

    def init_u(self):
        names = self.fmu.get_input_list().keys()
        return ModelicaModel.ModelicaVariables(self.fmu, names)

    @property
    def y(self):
        ''' system's outputs '''
        return self._y

    def init_y(self):
        names = self.fmu.get_output_list().keys()
        return ModelicaModel.ModelicaVariables(self.fmu, names)

    @property
    def parameters(self):
        ''' system's outputs '''
        return self._parameters

    def init_parameters(self):
        return None

    @property
    def global_seed(self):
        return self._global_seed

    def init_global_seed(self):
        if 'globalSeed.fixedSeed' in self.fmu.get_model_variables():
            return ModelicaModel.ModelicaVariables(self.fmu, ['globalSeed.fixedSeed'])
        else:
            return None

    @property
    def trajectory(self):
        ''' Simulation trajectory '''
        return self._trajectory

    def reset(self, x0=None, preserve_params=True):
        if preserve_params:
            param_data = self._all_parameters.data
        self.fmu.reset()
        self.trajectory.reset()
        if x0 is not None:
            self.x.set(x0)
        if preserve_params:
            self._all_parameters.set(param_data)

    def step(self, u=None):
        # save current state for iterative simulation
        x_data = self.x.data
        if self.parameters is not None:
            all_parameters_data = self._all_parameters.data
        # reset the simulation
        self.fmu.reset()
        # set the initial state
        self.x0.set(x_data)
        if self._all_parameters is not None:
            self._all_parameters.set(all_parameters_data)
        # set global seed for random numbers
        if self.global_seed is not None:
            self.global_seed.set([random.randrange(sys.maxsize)])
        # simulate
        if u is None:
            results = self.fmu.simulate(final_time=self.dt,
                                        options=self.opt)
        else:
            # assert len(u) == len(self.u),\
            #    'number of provided control inputs is different from the model'
            # self.u.set(u)
            results = self.fmu.simulate(final_time=self.dt,
                                        input=(self.u.names, lambda t: u),
                                        options=self.opt)
        self.trajectory.add_record(results)
        return results

    def simulate(self, n_steps, x0=None):
        if x0 is not None:
            self.reset(x0)
        for step in range(n_steps):
            self.step()

        return self.trajectory.as_df()

    def compile_sym(self):
        # Model
        try:
            self.dae = CasadiModel(self._mo_class_name,
                                   self._mo_filename, self)
        except Exception as err:
            print(err)
            self.dae = None
            warnings.warn("Unable to create Casadi DAE model", RuntimeWarning)

    def __init__(self, class_name, filename, dt):
        self._dt = dt
        self._mo_class_name = class_name
        self._mo_filename = filename
        # Compile model
        fmu_name = compile_fmu(class_name,
                               filename,
                               compiler_log_level='error')
        # Load model
        self._fmu = load_fmu(fmu_name)

        self._x = self.init_x()
        self._x0 = self.init_x0()
        self._u = self.init_u()
        self._y = self.init_y()
        self._parameters = self.init_parameters()
        self._global_seed = self.init_global_seed()

        # Get a reference for all parameters
        if self.parameters is None:
            self._all_parameters = None
        else:
            if isinstance(self.parameters, collections.Iterable):
                all_params = [params for params in self.parameters]
            else:
                all_params = [self.parameters]
            params_names = [name
                            for params in all_params if params is not None
                            for name in params.names]

            self._all_parameters = ModelicaModel.ModelicaVariables(self.fmu,
                                                                   params_names)

        self._trajectory = SimulationTrajectory(self)

        # options
        self.opt = self.fmu.simulate_options()
        self.opt['CVode_options']['verbosity'] = 40  # No output
