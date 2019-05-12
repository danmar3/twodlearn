#  ***********************************************************************
#   This file defines recurrent bayesian neural-networks
#
#   Wrote by: Daniel L. Marino (marinodl@vcu.edu)
#    Modern Heuristics Research Group (MHRG)
#    Virginia Commonwealth University (VCU), Richmond, VA
#    http://www.people.vcu.edu/~mmanic/
#
#   ***********************************************************************
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import types
import warnings
import numpy as np
import collections
import tensorflow as tf

import twodlearn as tdl
import twodlearn.feedforward as tdlf
from twodlearn import (common, recurrent, bayesnet)
import tensorflow_probability as tfp


def normal_residual_wrapper(model):
    model_call = six.get_method_function(model.call)

    def call_wrapper(object, inputs):
        def add_normal(dist_a, dist_b):
            if isinstance(dist_a, tf.Tensor):
                return tfp.distributions.Normal(
                    loc=dist_a + dist_b.loc,
                    scale=dist_b.scale)
            else:
                variance = tf.square(dist_a.scale) + tf.square(dist_b.scale)
                return tfp.distributions.Normal(
                    loc=dist_a.loc + dist_b.loc,
                    scale=tf.sqrt(variance))
        inputs, state = inputs
        outputs = model_call(object, [inputs, state])
        if isinstance(outputs, tuple):
            outputs, next_state = outputs
            return outputs, add_normal(state, next_state)
        else:
            next_state = outputs
            return add_normal(state, next_state)

    if isinstance(model, tdl.core.Layer):
        model.call = types.MethodType(call_wrapper, model)
        return model
    else:
        raise ValueError('concat wrapper does not work for type {}.'
                         ''.format(type(model)))


class NormalNarxCell(common.TdlModel):
    ''' (state, input) -> normalizer -> finput -> fz -> fy'''
    class NarxConcat(tdlf.Concat):
        def __call__(self, state, inputs):
            if isinstance(state, collections.Iterable):
                x = list(state) + [inputs]
            else:
                x = [state, inputs]
            return self.evaluate(x)

    @common.Submodel
    def fz(self, value):
        return value

    @common.Submodel
    def fy(self, value):
        if value is None:
            value = common.Identity()
        return value

    @common.Submodel
    def finput(self, value):
        if value is None:
            value = NormalNarxCell.NarxConcat(axis=1)
        return value

    @common.OptionalProperty
    def normalizer(self, value):
        assert isinstance(value, dict),\
            'normalizer for NormalNarxCell should be a dictionary '\
            'with (key,values): (\'y\', y_normalizer), (\'u\', u_normalizer)'
        for k, v in value.items():
            assert isinstance(v, tdl.normalizer.Normalizer),\
                'Item {} is not a tdl.Normalizer'.format(k)
        return value

    @common.Regularizer
    def regularizer(self, loc_scale=None, scale_scale=None):
        if scale_scale is None:
            scale_scale = loc_scale
        with tf.name_scope(self.scope):
            with tf.name_scope('regularizer'):
                reg = (self.fz.regularizer.value if self.fz.regularizer.is_set
                       else self.fz.regularizer.init(loc_scale=loc_scale,
                                                     scale_scale=scale_scale))
        return reg

    def __init__(self, fz, fy=None,
                 options=None, name='GaussianMlpCell', **kargs):
        super(NormalNarxCell, self).__init__(
            fz=fz, fy=None, name=name, options=options, **kargs)

    class NormalNarxOutput(tdl.core.OutputModel):
        @common.OptionalProperty
        def loss(self):
            return bayesnet.GaussianNegLogLikelihood(self.y)

    @tdl.ModelMethod(['y', 'x', 'z', 'dz'],
                     ['state', 'inputs', 't'],
                     NormalNarxOutput)
    def evaluate(self, object, state, inputs, t):
        # normalize
        if self.normalizer.is_set:
            dz_state = [(xt_i.z.samples if self.normalizer.value['y'] is None
                         else self.normalizer.value['y'].normalize(xt_i.z.samples).y)
                        for xt_i in state]
            dz_inputs = (inputs if self.normalizer.value['u'] is None
                         else self.normalizer.value['u'].normalize(inputs))
        else:
            dz_state = [xt_i.z.samples.value for xt_i in state]
            dz_inputs = inputs

        dz = self.fz.evaluate(self.finput(state=dz_state, inputs=dz_inputs))
        with tf.name_scope('z_{}'.format(t+1)):
            with tf.name_scope('loc'):
                z_mean = (dz.loc + state[-1].z.loc)
            with tf.name_scope('scale'):
                z_var = (dz.scale**2 + state[-1].z.scale**2)
                z_stddev = tf.sqrt(z_var)
            zt = bayesnet.McNormal(loc=z_mean, scale=z_stddev)
        yt = self.fy.evaluate(zt)
        data_t = common.SimpleNamespace(dz=dz, z=zt, y=yt)
        xt = tuple(list(state[1:]) + [data_t])
        return yt, xt, zt, dz


class GaussianMlpCell(tdl.core.TdlModel):
    @property
    def n_inputs(self):
        return self._n_inputs

    @property
    def n_outputs(self):
        return self._n_outputs

    @property
    def n_hidden(self):
        return self._n_hidden

    @property
    def window_size(self):
        return self._window_size

    @common.Submodel
    def mlp(self, value):
        return self._init_cell(
            n_inputs=self.n_inputs, n_outputs=self.n_outputs,
            window_size=self.window_size)

    def _init_cell(self, n_inputs, n_outputs, window_size):
        mlp = bayesnet.NormalMlp(
            loc_args={'n_inputs': n_inputs + n_outputs * window_size,
                      'n_outputs': n_outputs,
                      'n_hidden': self.n_hidden,
                      'afunction': self.options['afunction'],
                      'options': self.options['loc/options']},
            LocClass=bayesnet.BayesianMlp)
        return mlp

    @property
    def normalizer(self):
        return self._normalizer

    @normalizer.setter
    def normalizer(self, value):
        assert isinstance(value, dict),\
            'normalizer for GaussianMlpCell should be a dictionary '\
            'with (key,values): (\'y\', y_normalizer), (\'u\', u_normalizer)'
        for k, v in value.items():
            assert isinstance(v, tdl.normalizer.Normalizer),\
                'Item {} is not a tdl.Normalizer'.format(k)
        self._normalizer = value

    def _init_options(self, options):
        loc_options = {'layers/options': {'w/stddev/init_method': 'sum',
                                          'w/stddev/alpha': 1.0,
                                          'w/stddev/trainable': True,
                                          'w/prior/stddev': None}}
        default = {'afunction': tdlf.selu01,
                   'loc/options': loc_options,
                   'scale/class': None,
                   'scale/n_hidden': None,
                   'scale/y_min': None,
                   'scale/y_max': None,
                   'scale/options': None}
        options = common.check_defaults(options, default)
        return options

    @common.Regularizer
    def regularizer(self, scale=None):
        with tf.name_scope(self.scope):
            with tf.name_scope('regularizer'):
                reg = (self.mlp.regularizer.value
                       if self.mlp.regularizer.is_set
                       else self.mlp.regularizer.init(prior_stddev=scale))
        return reg

    def __init__(self, n_inputs, n_outputs, n_hidden, window_size,
                 options=None, name='GaussianMlpCell'):
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._n_hidden = n_hidden
        self._window_size = window_size
        self._normalizer = collections.defaultdict(lambda: None)

        super(GaussianMlpCell, self).__init__(name=name, options=options)

    class Output(tdl.core.TdlModel):
        @tdl.core.InputModel
        def model(self, value):
            return value

        @property
        def y(self):
            ''' Observable output of the cell '''
            return self._y

        @property
        def x(self):
            ''' State of the cell, in the case of Narx,
            is the list of window_size previous observations y '''
            return self._x

        @property
        def x_inputs(self):
            ''' State inputs to the cell '''
            return self._x_inputs

        @property
        def u_inputs(self):
            ''' Exogenous inputs to the cell '''
            return self._u_inputs

        @property
        def dz(self):
            ''' Output delta increment before applying
            observation noise '''
            return self._dz

        def _do_normalize(self, v, var):
            ''' perform normalization of tensor v using normalizer var '''
            normalizer = self.model.normalizer[var]
            return (v if normalizer is None
                    else normalizer.normalize(v).y)

        def _do_denormalize(self, v, var):
            ''' perform normalization of tensor v using normalizer var '''
            normalizer = self.model.normalizer[var]
            return (v if normalizer is None
                    else normalizer.denormalize(v).y)

        def evaluate(self, xt, ut, t):
            dz = self.model.mlp.evaluate(
                tf.concat([self._do_normalize(z.samples.value, 'y')
                           for z in xt] +
                          [self._do_normalize(ut, 'u')],
                          axis=1))
            with tf.name_scope('y_{}'.format(t+1)):
                with tf.name_scope('loc'):
                    z_mean = (dz.loc + xt[-1].loc)
                with tf.name_scope('scale'):
                    z_var = (dz.scale**2 + xt[-1].scale**2)
                    z_stddev = tf.sqrt(z_var)
                with tf.name_scope('z_{}'.format(t+1)):
                    zt = bayesnet.McNormal(loc=z_mean,
                                           scale=z_stddev)
            xt = tuple(list(xt[1:]) + [zt])
            return zt, xt, dz

        def __init__(self, model, state, inputs, t, options=None, name=None):
            self._x_inputs = state
            self._u_inputs = inputs
            super(GaussianMlpCell.Output, self)\
                .__init__(model=model, options=options, name=name)
            with tf.name_scope(self.scope):
                self._y, self._x, self._dz = self.evaluate(state, inputs, t)
    ModelOutput = Output

    def evaluate(self, state, inputs, t, options=None, name=None):
        return self.setup(state=state, inputs=inputs, t=t,
                          options=options, name=name)

    def setup(self, *args, **kargs):
        warnings.warn('setup is deprecated, will be removed in the future')
        assert len(args) == 0,\
            'arguments for setup must be explicitly specified'
        return self.ModelOutput(self, **kargs)


class DropoutMlpCell(GaussianMlpCell):
    @property
    def keep_prob(self):
        return self._keep_prob

    def _init_options(self, options):
        loc_options = {'layers/options': {'w/prior/stddev': None}}
        default = {'afunction': tdlf.selu01,
                   'y/normalizer': None,
                   'u/normalizer': None,
                   'loc/options': loc_options}
        options = common.check_defaults(options, default)
        return options

    def _init_cell(self, n_inputs, n_outputs, window_size):
        mlp = bayesnet.ConditionalGaussianMlp(
            loc_args={'n_inputs': n_inputs + n_outputs * window_size,
                      'n_outputs': n_outputs,
                      'n_hidden': self.n_hidden,
                      'afunction': self.options['afunction'],
                      'keep_prob': self.keep_prob,
                      'options': self.options['loc/options']},
            LocClass=bayesnet.BernoulliBayesianMlp)
        return mlp

    def __init__(self, n_inputs, n_outputs, n_hidden, window_size,
                 keep_prob, options=None, name='DropoutMlpCell'):
        self._keep_prob = keep_prob
        super(DropoutMlpCell, self).__init__(n_inputs=n_inputs,
                                             n_outputs=n_outputs,
                                             n_hidden=n_hidden,
                                             window_size=window_size,
                                             options=options,
                                             name=name)


class BayesNarx(recurrent.SimpleRnn):
    @property
    def window_size(self):
        return self._window_size

    def __init__(self, cell, window_size, name=None,
                 options=None, **kargs):
        self._window_size = window_size
        super(BayesNarx, self).__init__(cell=cell, options=options,
                                        name=name, **kargs)

    class RnnOutput(recurrent.SimpleRnn.RnnOutput):
        def _init_options(self, options):
            default = {
                'inputs/mean': 0.0,
                'inputs/std': 0.1,
                'inputs/type': 'placeholder',
                'inputs/shape': [None, None],
                'x0/stddev': 1e-4,
                'x0/shape': [None, None]
            }
            options = common.check_defaults(options, default)
            options = super(BayesNarx.RnnOutput, self)._init_options(options)
            return options

        @property
        def window_size(self):
            return self.model.window_size

        def _xt_transfer_func(self, xt, t):
            if t == 0:
                return [common.SimpleNamespace(y=None, z=xi, dz=None)
                        for xi in xt]
            else:
                return xt

        def _ut_transfer_func(self, ut, t):
            return ut

        @common.InputArgument
        def x0(self, value):
            """list of observations of the initial state.
            Args:
                value (None, tf.Tensor, McNormal): value for x0
            Returns:
                list(McNormal): x0
            """
            def xi_to_dist(xi):
                xi = (xi if isinstance(xi, bayesnet.McNormal)
                      else bayesnet.McNormal(loc=xi,
                                             scale=self.options['x0/stddev'],
                                             samples=xi))
                return xi
            base = ([tf.placeholder(tf.float32, shape=self.options['x0/shape'])
                     for i in range(self.window_size)]
                    if value is None
                    else value)
            assert len(base) == self.window_size,\
                'len(x0) is different from model window_size'
            value = [xi_to_dist(xi) for xi in base]
            return value

        def _next_step(self, xt, ut, t):
            ''' Evaluate the cell at time t'''
            with tf.name_scope('cell_{}'.format(t)):
                xt = self._xt_transfer_func(xt, t)
                ut = self._ut_transfer_func(ut, t)
                net_t = self.model.cell.evaluate(state=xt, inputs=ut, t=t)
            return net_t.y, net_t.x, net_t

    class McRnnOutput(RnnOutput):
        @property
        def n_particles(self):
            return self._n_particles

        def _init_options(self, options):
            default = {
                'inputs/mean': 0.0,
                'inputs/std': 0.1,
                'inputs/type': 'placeholder',
                'inputs/shape': [1, None],
                'x0/stddev': 1e-4,
                'x0/shape': [1, None]
            }
            options = common.check_defaults(options, default)
            options = super(BayesNarx.RnnOutput, self)._init_options(options)
            return options

        @common.InputArgument
        def x0(self, value):
            value = ([tf.placeholder(tf.float32,
                                     shape=self.options['x0/shape'])
                      for i in range(self.window_size)]
                     if value is None
                     else value)
            value = [bayesnet.Particles(n_particles=self.n_particles,
                                        base=xi)
                     for xi in value]
            assert len(value) == self.window_size,\
                'len(x0) is different from model window_size'
            value = super(type(self), type(self)).x0.finit(self, value)
            return value

        @common.InputArgument
        def inputs(self, value):
            value = super(type(self), type(self)).inputs.finit(self, value)
            if all([not isinstance(xi, bayesnet.Particles) for xi in value]):
                value = [bayesnet.Particles(n_particles=self.n_particles,
                                            base=xi)
                         for xi in value]
            return value

        def __init__(self, n_particles, x0=None, inputs=None,
                     n_unrollings=None, options=None, name=None, **kargs):
            self._n_particles = n_particles
            super(BayesNarx.McRnnOutput, self).__init__(
                x0=x0, inputs=inputs, n_unrollings=n_unrollings,
                options=options, name=name, **kargs)

    def mc_evaluate(self, n_particles, x0=None, inputs=None,
                    n_unrollings=None, options=None, name=None, **kargs):
        return type(self).McRnnOutput(model=self, n_particles=n_particles,
                                      x0=x0, inputs=inputs,
                                      n_unrollings=n_unrollings,
                                      options=options, name=name)
