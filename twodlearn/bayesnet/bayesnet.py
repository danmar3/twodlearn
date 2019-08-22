"""Definition of several bayesian neural-networks
"""

import numbers
import warnings
import numpy as np
import tensorflow as tf
import twodlearn as tdl
from twodlearn import common
import twodlearn.feedforward as tdlf
import tensorflow_probability as tfp
from collections import namedtuple
# -------------------- Losses -------------------- #


class GaussianKL(tdlf.Loss):
    ''' Evaluate KL(p||q) for p and q Normal '''
    @property
    def p(self):
        return self._p

    @property
    def q(self):
        return self._q

    def get_n_vars(self, p, q):
        def is_fully_defined(x):
            return (x.loc.shape.is_fully_defined() and
                    x.scale.shape.is_fully_defined())

        if is_fully_defined(p) and is_fully_defined(q):
            loc_shape = tf.broadcast_static_shape(p.loc.shape,
                                                  q.loc.shape)
            scale_shape = tf.broadcast_static_shape(p.scale.shape,
                                                    q.scale.shape)
            shape = tf.broadcast_static_shape(loc_shape, scale_shape)
        else:
            loc_shape = tf.broadcast_dynamic_shape(p.loc.shape,
                                                   q.loc.shape)
            scale_shape = tf.broadcast_dynamic_shape(p.scale.shape,
                                                     q.scale.shape)
            shape = tf.broadcast_dynamic_shape(loc_shape, scale_shape)
        return tf.reduce_prod(shape)

    def evaluate(self, p, q):
        ''' Evaluate KL(p||q) '''
        p_var = p.scale**2
        q_var = q.scale**2
        ratio = p_var / q_var
        kl = -tf.log(ratio) + ratio \
            + (tf.square(p.loc - q.loc) / q_var)
        n_vars = self.get_n_vars(p, q)
        return 0.5 * (tf.reduce_sum(kl) - tf.cast(n_vars, tf.float32))

    def fromlist(self, p_list):
        with tf.name_scope('list2vector'):
            p_loc = tf.concat([tf.reshape(p.loc, [-1])
                               for p in p_list], axis=0)
            p_scale = tf.concat([tf.reshape(p.scale, [-1])
                                 for p in p_list], axis=0)
            p = tfp.distributions.Normal(p_loc, p_scale)
        return p

    def __init__(self, p, q, name='GaussianKL'):
        ''' p: a normal distribution or a list of normal distributions,
            q: base normal distribution'''
        super(GaussianKL, self).__init__(name=name)
        with tf.name_scope(self.scope):
            # check type of p
            if isinstance(p, list):
                p = self.fromlist(p)

            # evaluate kl divergence
            assert (isinstance(p, (tfp.distributions.Normal, Normal,
                                   McNormal)) and
                    isinstance(q, (tfp.distributions.Normal, Normal,
                                   McNormal))), \
                'GaussianKL is only defined for p, q being '\
                'tf.distributions.Normal or tdl.bayesnet.Normal'
            self._p = p
            self._q = q
            self._value = self.evaluate(p, q)

    @classmethod
    def fromstats(cls, p_loc, p_scale, q_loc, q_scale):
        p = tfp.distributions.Normal(p_loc, p_scale)
        q = tfp.distributions.Normal(q_loc, q_scale)
        return cls(p, q)


class GaussianNegLogLikelihood(tdlf.EmpiricalLoss):
    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        assert not hasattr(self, '_y'),\
            'property y can only be set during initialization'
        assert (isinstance(value, tf.distributions.Normal) or
                isinstance(value, McNormal)),\
            'y must be a normal (tf.distributions.Normal) distribution'
        self._y = value

    @property
    def n_outputs(self):
        return self.y.shape[1].value

    @property
    def labels(self):
        ''' Labels for computing the loss, if not provided,
        they are created automatically '''
        return self._labels

    @labels.setter
    def labels(self, value):
        assert not hasattr(self, '_labels'), \
            'property labels can only be set during initialization'
        if value is None:
            self._labels = tf.placeholder(tf.float32,
                                          shape=self.y.loc.shape,
                                          name='labels')
        else:
            self._labels = value

    def define_fit_loss(self, y, labels):
        y_mu = y.loc
        y_variance = y.scale**2
        loss_i = tf.reduce_sum(tf.log(y_variance) +
                               (tf.pow(labels - y_mu, 2) / (y_variance)), 1)
        loss = 0.5 * tf.reduce_mean(loss_i, 0)
        return loss

    def __init__(self, y, labels=None, name='NegLogLikelihood'):
        super(GaussianNegLogLikelihood, self).__init__(name=name)
        with tf.name_scope(self.scope):
            self.y = y
            self.labels = labels
            self._value = self.define_fit_loss(self.y, self.labels)


class Entropy(common.TdlModel):
    @property
    def value(self):
        return self._value

    @property
    def prob(self):
        return self._prob

    def _evaluate(self, prob):
        return tf.reduce_mean(- prob * tf.log(prob), axis=1)

    def __init__(self, prob, name=None):
        self._prob = prob
        super(Entropy, self).__init__(name=name)
        with tf.name_scope(self.scope):
            self._value = self._evaluate(self.prob)

# -------------------- Models -------------------- #


class McEstimate(common.TdlModel):
    @common.LazzyProperty
    def mean(self):
        return tf.reduce_mean(self.value, axis=0)

    @common.LazzyProperty
    def stddev(self):
        N = tf.cast(tf.shape(self.value)[0], tf.float32)
        diff = (self.value - self.mean)**2
        sample_variance = (tf.reduce_sum(diff, axis=0) / (N - 1))
        return tf.sqrt(sample_variance)

    def __init__(self, value, name='mc_estimate'):
        self.value = value
        super(McEstimate, self).__init__(name=name)


class McSample(common.TdlModel):
    @tdl.core.InputModel
    def distribution(self, value):
        if not hasattr(value, 'sample'):
            raise TypeError('distribution model should have a sample method')
        return value

    @tdl.core.InputArgument
    def sample_axis(self, value):
        if value is None:
            value = 0
        return value


class McNormal(common.TdlModel):
    @property
    def value(self):
        return self.samples.value

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, value):
        if value is None:
            with tf.name_scope(self.scope):
                self._samples = McEstimate(value=self._distribution.sample())
        elif isinstance(value, McEstimate):
            self._samples = value
        else:
            with tf.name_scope(self.scope):
                self._samples = McEstimate(value=value)

    @common.Submodel
    def _distribution(self, value):
        if value is None:
            value = tfp.distributions.Normal(loc=self.loc, scale=self.scale)
        return value

    @common.Submodel
    def loc(self, value):
        return value

    @common.Submodel
    def scale(self, value):
        return value

    def resample(self, *args, **kargs):
        with tf.name_scope(self.scope):
            samples = self._distribution.sample(*args, **kargs)
        self.samples = samples
        return samples

    def __init__(self, loc, scale, samples=None, name='McNormal', **kargs):
        super(McNormal, self).__init__(loc=loc, scale=scale,
                                       name=name, **kargs)
        self.samples = samples


@tdl.core.create_init_docstring
class SampleLayer(tdl.core.Layer):
    @tdl.core.InputArgument
    def input_shape(self, value):
        return value

    @tdl.core.InputModel
    def distribution(self, value):
        return value

    @tdl.core.InputModel
    def sample_shape(self, value):
        if value is not None:
            value = tf.TensorShape(value)
        return value

    def call(self, inputs, *args, **kargs):
        tdl.core.assert_initialized(
            self, 'call', ['distribution', 'sample_shape'])
        if self.distribution is None:
            distribution = inputs
        else:
            distribution = (self.distribution(inputs)
                            if callable(self.distribution)
                            else self.distribution)
        if self.sample_shape is not None:
            return distribution.sample(sample_shape=self.sample_shape)
        else:
            return distribution.sample()


@tdl.core.create_init_docstring
class NormalModel(tdl.core.Layer):
    @tdl.core.InputArgument
    def input_shape(self, value):
        '''Input tensor shape.'''
        if value is None:
            raise tdl.core.exceptions.ArgumentNotProvided(self)
        return tf.TensorShape(value)

    @tdl.core.InputArgument
    def batch_shape(self, value):
        if value is None:
            raise tdl.core.exceptions.ArgumentNotProvided(self)
        if not isinstance(value, tf.TensorShape):
            value = tf.TensorShape(value)
        return value

    @tdl.core.SubmodelInit
    def loc(self, initializer=None, trainable=True, **kargs):
        tdl.core.assert_initialized(self, 'loc', ['batch_shape'])
        if initializer is None:
            initializer = tf.keras.initializers.zeros()
        shape = tf.TensorShape(
            (1 if dim is None else dim)
            for dim in self.batch_shape.as_list())
        return self.add_weight(
            name='loc',
            initializer=initializer,
            shape=shape,
            trainable=trainable,
            **kargs)

    @tdl.core.SubmodelInit
    def scale(self, initializer=None, trainable=True, tolerance=1e-5,
              **kargs):
        tdl.core.assert_initialized(self, 'shape', ['batch_shape'])
        shape = tf.TensorShape(
            (1 if dim is None else dim)
            for dim in self.batch_shape.as_list())
        if initializer is None:
            initializer = tdl.constrained.PositiveVariableExp.init_wrapper(
                initializer=tf.keras.initializers.ones(),
                trainable=trainable,
                tolerance=tolerance,
                **kargs)
        return initializer(shape=shape)

    def build(self, input_shape=None):
        tdl.core.assert_initialized(self, 'build', ['loc', 'scale'])
        self.built = True

    def call(self, inputs, *args, **kargs):
        if inputs is not None:
            inputs = tf.convert_to_tensor(inputs)
        loc = (self.loc(inputs) if callable(self.loc)
               else self.loc)
        scale = (self.scale(inputs) if callable(self.scale)
                 else self.scale)
        return tfp.distributions.Normal(loc=loc, scale=scale)


class AffineBernoulliLayer(tdl.dense.AffineLayer):
    ''' Implements the layer y=dropout(x) W + b'''
    @tdl.core.InputArgument
    def keep_prob(self, value):
        '''Keep prob for dropout.'''
        return (value if value is not None
                else 0.8)

    @common.Regularizer
    def regularizer(self, prior_stddev=None):
        tdl.core.assert_initialized(
            self, 'regularizer', ['input_shape', 'kernel'])
        if prior_stddev is None:
            prior_stddev = self.options['w/prior/stddev']
        with tf.name_scope(self.scope):
            reg = tdlf.L2Regularizer(
                self.kernel,
                scale=(prior_stddev**2)/np.sqrt(self.input_shape[-1].value))
        return reg

    class Output(tdlf.AffineLayer.Output):
        @property
        def keep_prob(self):
            return self.model.keep_prob

        @tdl.core.Submodel
        def affine(self, _):
            keep_prob = self.model.keep_prob
            inputs = (self.inputs if keep_prob is None
                      else tf.nn.dropout(self.inputs, keep_prob))
            output = tf.linalg.LinearOperatorFullMatrix(self.model.kernel)\
                       .matvec(inputs, adjoint=True)
            if self.model.bias is not None:
                output = output + self.model.bias
            return output

        @tdl.core.OutputValue
        def value(self, _):
            return self.affine


class DenseBernoulliLayer(AffineBernoulliLayer):
    @tdl.core.InputArgument
    def activation(self, value):
        return value

    def __init__(self, activation=tf.nn.relu, name=None, **kargs):
        super(DenseBernoulliLayer, self).__init__(
            activation=activation,
            name=name, **kargs)

    class Output(AffineBernoulliLayer.Output):
        @property
        def keep_prob(self):
            return self.model.keep_prob

        @common.OutputValue
        def value(self, _):
            return self.model.activation(self.affine)


@tdl.core.create_init_docstring
class LinearNormalLayer(tdl.dense.LinearLayer):
    @tdl.core.InputArgument
    def tolerance(self, value):
        return value

    @tdl.core.ParameterInit
    def kernel(self, initializer=None, trainable=True, max_scale=1.0):
        tdl.core.assert_initialized(
            self, 'kernel', ['units', 'input_shape'])
        if initializer is None:
            initializer = tdl.constrained.PositiveVariableExp.init_wrapper(
                initializer=tdl.core.initializers.SumFanConstant(),
                trainable=trainable,
                max=max_scale,
                tolerance=self.tolerance)

        scale = initializer(shape=[self.input_shape[-1].value, self.units])
        for var in tdl.core.get_variables(scale):
            self.add_weight(var)
        loc = self.add_weight(
            name='loc',
            initializer=tf.keras.initializers.zeros(),
            shape=[self.input_shape[-1].value, self.units],
            trainable=trainable)
        return tfp.distributions.Normal(loc=loc, scale=scale, name='kernel')

    @tdl.core.Regularizer
    def regularizer(self, prior_stddev=None):
        ''' Return the KL regularizer for the layer '''
        tdl.core.assert_initialized(self, 'regularizer', ['kernel'])
        if prior_stddev is None:
            assert self.options['w/prior/stddev'] is not None,\
                'prior stddev not specified as agument nor '\
                'found in the model options'
            assert isinstance(self.options['w/prior/stddev'],
                              numbers.Real), \
                'provided prior stddev is not a number'
            assert self.options['w/prior/stddev'] > 0.0, \
                'provided prior stddev is <=0 '
            prior_stddev = self.options['w/prior/stddev']
        with tf.name_scope(self.scope):
            with tf.name_scope('regularizer'):
                prior = tfp.distributions.Normal(
                    loc=0.0, scale=prior_stddev)
                reg = tf.reduce_sum(
                    tfp.distributions.kl_divergence(self.kernel, prior))
        return reg

    def _init_options(self, options):
        default = {'w/stddev/init_method': 'sum',
                   'w/stddev/alpha': 1.0,
                   'w/stddev/trainable': True,
                   'w/prior/stddev': None}
        options = common.check_defaults(options, default)
        options = super(LinearNormalLayer, self)._init_options(options)
        return options

    class Output(tdl.core.TdlModel):
        @property
        def shape(self):
            return self.value.shape

        @tdl.core.InputModel
        def model(self, value):
            return value

        @tdl.core.InputArgument
        def inputs(self, value):
            return value

        def _loc(self):
            kernel = self.model.kernel
            return tf.linalg.LinearOperatorFullMatrix(kernel.loc)\
                     .matvec(self.inputs, adjoint=True)

        def _scale(self):
            kernel_cov = tf.linalg.LinearOperatorFullMatrix(
                tf.square(self.model.kernel.scale))
            output_cov = kernel_cov.matvec(tf.square(self.inputs),
                                           adjoint=True)
            return tf.sqrt(output_cov)

        @tdl.core.Submodel
        def affine(self, _):
            '''Normal distribution for the outputs.'''
            return tfp.distributions.Normal(
                loc=self._loc(), scale=self._scale())

        @tdl.core.OutputValue
        def value(self, _):
            '''Sample from the output distribution.'''
            return self.affine.sample()

    def call(self, inputs, *args, **kargs):
        return type(self).Output(model=self, inputs=inputs, *args, **kargs)


class AffineNormalLayer(LinearNormalLayer):
    @tdl.core.ParameterInit
    def bias(self, initializer=None, trainable=True, **kargs):
        tdl.core.assert_initialized(self, 'bias', ['units'])
        if initializer is None:
            initializer = tf.keras.initializers.zeros()
        return self.add_weight(name='bias', initializer=initializer,
                               shape=[self.units], trainable=trainable,
                               **kargs)

    class Output(LinearNormalLayer.Output):
        def _loc(self):
            kernel = self.model.kernel
            bias = self.model.bias
            loc = tf.linalg.LinearOperatorFullMatrix(kernel.loc)\
                    .matvec(self.inputs, adjoint=True)
            if bias is not None:
                loc = loc + bias
            return loc


class DenseNormalLayer(AffineNormalLayer):
    @tdl.core.InputArgument
    def activation(self, value):
        if value is not None:
            if not callable(value):
                raise ValueError('activation function must be callable')
        return value

    def __init__(self, activation=tf.nn.relu, name=None, **kargs):
        super(DenseNormalLayer, self).__init__(
            activation=activation, name=name, **kargs)

    class Output(AffineNormalLayer.Output):
        @property
        def value(self):
            activation = self.model.activation
            samples = self.affine.sample()
            return (samples if activation is None
                    else activation(samples))


class BayesianMlp(tdl.StackedModel):
    ''' Mlp composed of layers whose weights are sampled from a variational
    posterior distribution
    '''
    @property
    def n_inputs(self):
        ''' size of the input vectors '''
        return self.layers[0].input_shape[-1]

    @property
    def n_outputs(self):
        ''' size of the output vectors '''
        return self.layers[-1].units

    @property
    def kernels(self):
        ''' list the weights distributions from the layers '''
        return [layer.kernel for layer in self.layers
                if hasattr(layer, 'kernel')]

    @property
    def parameters(self):
        return [pi for layer in self.layers for pi in layer.parameters]

    _HiddenClass = DenseNormalLayer
    _OutputClass = AffineNormalLayer

    def _define_layers(self, n_inputs, n_outputs, n_hidden, afunction):
        layers = list()
        Layers = [self._HiddenClass] * len(n_hidden) + [self._OutputClass]
        if not isinstance(afunction, list):
            afunction = [afunction for i in range(len(n_hidden))] + [None]

        _n_inputs = n_inputs
        for l, n_units in enumerate(n_hidden + [n_outputs]):
            if afunction[l] is not None:
                layers.append(Layers[l](input_shape=_n_inputs,
                                        units=n_units,
                                        activation=afunction[l]))
            else:
                layers.append(Layers[l](input_shape=_n_inputs,
                                        units=n_units))
            _n_inputs = n_units
        return layers

    @tdl.Submodel
    def layers(self, value):
        layers = self._define_layers(*value)
        return layers

    @common.Regularizer
    def regularizer(self, prior_stddev=None):
        with tf.name_scope(self.scope):
            with tf.name_scope('regularizer'):
                reg = [(layer.regularizer.value if layer.regularizer.is_set
                        else layer.regularizer.init(prior_stddev))
                       for layer in self.layers
                       if hasattr(layer, 'regularizer')]
                if reg:
                    reg = (reg[0] if len(reg) == 1
                           else tdl.losses.AddNLosses(reg))
                else:
                    raise AttributeError(
                        'None of the Layers has a regularizer defined')
        return reg

    class BayesMlpOutput(tdl.core.OutputModel):
        @property
        def kernels(self):
            return self.model.kernels

        @property
        def shape(self):
            return self.value.shape

    @tdl.ModelMethod(['output', 'hidden', 'value'], ['inputs'], BayesMlpOutput)
    def evaluate(self, object, inputs):
        x = inputs
        hidden = list()
        for layer in self.layers:
            if isinstance(x, (tdl.common.TdlOp)):
                x = layer(x.value)
            else:
                x = layer(x)
            hidden.append(x)
        y = hidden[-1]
        if hasattr(y, 'value'):
            value = y.value
        else:
            value = None
        return y, hidden, value

    def _init_options(self, options):
        layers_default = {'w/stddev/init_method': 'sum',
                          'w/stddev/alpha': 1.0,
                          'w/stddev/trainable': True}
        default = {'layers/options': layers_default}
        options = tdl.common.check_defaults(options, default)
        options = super(BayesianMlp, self)._init_options(options)
        return options

    def __init__(self, n_inputs, n_outputs, n_hidden,
                 afunction=tdlf.selu01, options=None,
                 name='BayesianMlp'):
        super(BayesianMlp, self).__init__(
            layers=(n_inputs, n_outputs, n_hidden, afunction),
            options=options, name=name)


class BernoulliBayesianMlp(BayesianMlp):
    _HiddenClass = DenseBernoulliLayer
    _OutputClass = AffineBernoulliLayer

    def _define_layers(self, n_inputs, n_outputs, n_hidden,
                       keep_prob, afunction):
        layers = list()
        Layers = [self._HiddenClass] * len(n_hidden) + [self._OutputClass]
        if not isinstance(afunction, list):
            afunction = [afunction for i in range(len(n_hidden))] + [None]

        if not isinstance(keep_prob, list):
            if len(n_hidden) == 0:
                keep_prob = [keep_prob]
            else:
                keep_prob = [None] + [keep_prob for i in range(len(n_hidden))]

        _n_inputs = n_inputs
        for l, n_units in enumerate(n_hidden + [n_outputs]):
            if afunction[l] is not None:
                layers.append(Layers[l](input_shape=_n_inputs,
                                        units=n_units,
                                        activation=afunction[l],
                                        keep_prob=keep_prob[l]))
            else:
                layers.append(Layers[l](input_shape=_n_inputs,
                                        units=n_units,
                                        keep_prob=keep_prob[l]))
            _n_inputs = n_units
        return layers

    def __init__(self, n_inputs, n_outputs, n_hidden, keep_prob=0.8,
                 options=None, afunction=tf.nn.relu,
                 name='BernoulliBayesianMlp'):
        tdl.StackedModel.__init__(
            self,
            layers=(n_inputs, n_outputs,
                    n_hidden, keep_prob, afunction),
            options=options, name=name)


class BoundedBayesianMlp(BayesianMlp):
    ''' Multi-layer bayesian neural network, with bounded output '''
    @tdl.Submodel
    def layers(self, value):
        try:
            n_inputs, n_outputs, n_hidden, lower, upper, afunction = value
        except:
            raise AttributeError('Wrong format for initializing layers. '
                                 'Format should be: '
                                 'n_inputs, n_outputs, n_hidden, lower, '
                                 'upper, afunction')
        layers = self._define_layers(n_inputs, n_outputs, n_hidden, afunction)
        layers.append(tdl.feedforward.BoundedOutput(lower=lower, upper=upper))
        return layers

    def __init__(self, n_inputs, n_outputs, n_hidden,
                 lower=1e-7, upper=None,
                 afunction=tdlf.selu01, options=None,
                 name='BayesianMlp'):
        super(BayesianMlp, self).__init__(
            layers=(n_inputs, n_outputs, n_hidden, lower, upper, afunction),
            options=options, name=name)


class BoundedBernoulliBayesianMlp(BernoulliBayesianMlp):
    ''' Multi-layer bayesian neural network, with bounded output '''
    @tdl.Submodel
    def layers(self, value):
        try:
            (n_inputs, n_outputs, n_hidden, keep_prob,
             lower, upper, afunction) = value
        except:
            raise AttributeError('Wrong format for initializing layers. '
                                 'Format should be: '
                                 'n_inputs, n_outputs, n_hidden, lower, '
                                 'upper, afunction')
        layers = self._define_layers(n_inputs, n_outputs, n_hidden,
                                     keep_prob, afunction)
        layers.append(tdl.feedforward.BoundedOutput(lower=lower, upper=upper))
        return layers

    def __init__(self, n_inputs, n_outputs, n_hidden, keep_prob,
                 lower=1e-7, upper=None, afunction=tdlf.selu01,
                 options=None, name='BayesianMlp'):
        super(BayesianMlp, self).__init__(
            layers=(n_inputs, n_outputs, n_hidden, keep_prob,
                    lower, upper, afunction),
            options=options, name=name)


class Normal(tdl.TdlModel):
    _submodels = ['loc', 'scale']

    @common.SubmodelWithArgs
    def loc(self, value, shape):
        if (value is None) and (shape is not None):
            value = tf.Variable(tf.zeros(shape=shape), trainable=True)

        if (value is None) and (shape is None):
            raise ValueError('You must provide either a value for loc '
                             'or a shape to create a variable')

        if isinstance(value, (int, float)) and (shape is not None):
            value = tdl.variable(tf.constant(value, shape=shape),
                                 trainable=True)
        return value

    @common.SubmodelWithArgs
    def scale(self, value, shape):
        def get_value(d):
            return (d if isinstance(d, int)
                    else 1 if d is None
                    else d.value if hasattr(d, 'value')
                    else d)

        def replace_none(shape_in):
            return [get_value(d) for d in shape_in]
        shape = (replace_none(shape) if shape is not None
                 else (replace_none(self.loc.shape)
                       if hasattr(self.loc, 'shape')
                       else None))

        def initializer(initial_value):
            return tf.random_normal(shape=shape,
                                    mean=initial_value,
                                    stddev=0.00001)
        if (value is None) and (shape is not None):
            value = tdl.common.PositiveVariableExp(initializer=initializer,
                                                   initial_value=1.0,
                                                   trainable=True)

        if isinstance(value, (int, float)) and (shape is not None):
            value = tdl.common.PositiveVariableExp(initializer=initializer,
                                                   initial_value=value,
                                                   trainable=True)
        if value is None:
            raise ValueError('Unable to identify the shape for scale. '
                             'Provide shape to create a variable or '
                             'directly specify the scale.')
        return value

    @common.Regularizer
    def regularizer(self, loc_scale=None, scale_scale=None):
        ''' Returns a sum of the loc and scale regularizers '''
        with tf.name_scope(self.scope):
            reg = None
            if hasattr(self.loc, 'regularizer'):
                reg = (self.loc.regularizer.value
                       if self.loc.regularizer.is_set
                       else self.loc.regularizer.init(loc_scale))
            if hasattr(self.scale, 'regularizer'):
                scale_reg = (self.scale.regularizer.value
                             if self.scale.regularizer.is_set
                             else self.scale.regularizer.init(scale_scale))
                reg = (scale_reg if reg is None
                       else reg + scale_reg)
        return reg

    class NormalOutput(McNormal):
        @common.InputArgument
        def inputs(self, value):
            return value

        @common.Submodel
        def loc(self, value):
            if callable(self.model.loc):
                return self.model.loc(self.inputs)
            else:
                return self.model.loc

        @common.Submodel
        def scale(self, value):
            if callable(self.model.scale):
                return self.model.scale(self.inputs)
            else:
                return self.model.scale

        def __init__(self, model, inputs, name=None):
            self.model = model
            super(Normal.NormalOutput, self).__init__(
                loc=None, scale=None, inputs=inputs, name=name)

    def evaluate(self, inputs=None, name=None):
        return type(self).NormalOutput(self, inputs=inputs, name=name)

    def __call__(self, inputs, name=None):
        return self.evaluate(inputs=inputs, name=name)

    def __init__(self, loc=None, scale=None, shape=None,
                 name='Normal', **kargs):
        super(Normal, self).__init__(
            loc={'value': loc, 'shape': shape},
            scale={'value': scale, 'shape': shape},
            name=name, **kargs)


class ConditionalNormal(Normal):
    _submodels = ['loc', 'scale']
    NormalOutput = Normal.NormalOutput

    @common.SubmodelWithArgs
    def loc(self, value, shape):
        if value is None:
            value = tdl.common.Identity
        return value

    def __init__(self, loc=None, scale=None, shape=None,
                 name='ConditionalNormal'):
        super(ConditionalNormal, self).__init__(loc=loc, scale=scale,
                                                shape=shape,
                                                name=name)


class NormalMlp(ConditionalNormal):
    class NormalOutput(Normal.NormalOutput):
        @property
        def n_inputs(self):
            return self.model.n_inputs

        @property
        def n_outputs(self):
            return self.model.n_inputs

        @common.InputArgument
        def inputs(self, value):
            if value is None:
                value = tf.placeholder(tf.float32)
            return value

    class McNormalOutput(NormalOutput):
        @property
        def n_particles(self):
            return self._n_particles

        @common.InputArgument
        def inputs(self, value):
            if value is None:
                value = tf.placeholder(tf.float32, shape=[1, self.n_inputs])
            if not isinstance(value, Particles):
                value = Particles(n_particles=self.n_particles, base=value)
            return value

        def __init__(self, model, n_particles, inputs, name=None):
            self._n_particles = n_particles
            super(NormalMlp.McNormalOutput, self).__init__(
                model=model, inputs=inputs, name=name)

    @property
    def n_inputs(self):
        return self.loc.n_inputs

    @property
    def n_outputs(self):
        return self.loc.n_outputs

    @common.SubmodelWithArgs
    def loc(self, LocClass, loc_args):
        if LocClass is None:
            return super(type(self), type(self)).loc.finit(
                self, value=None, shape=[1, self.n_inputs])
        return LocClass(**loc_args)

    @common.SubmodelWithArgs
    def scale(self, ScaleClass, scale_args):
        if ScaleClass is None:
            return super(type(self), type(self)).scale.finit(
                self, value=None, shape=[1, self.n_inputs])
        scale_args.setdefault('n_inputs', self.loc.n_inputs)
        scale_args.setdefault('n_outputs', self.loc.n_inputs)
        return ScaleClass(**scale_args)

    def __init__(self, loc_args, scale_args={},
                 LocClass=BayesianMlp, ScaleClass=None,
                 options=None, name='GaussianMlp'):
        tdl.TdlModel.__init__(
            self, loc={'LocClass': LocClass, 'loc_args': loc_args},
            scale={'ScaleClass': ScaleClass, 'scale_args': scale_args},
            name=name, options=options)

    def mc_evaluate(self, n_particles, x=None, name=None):
        return NormalMlp.McNormalOutput(self, n_particles, inputs=x, name=name)


class HeteroscedasticNormalMlp(NormalMlp):
    ''' Defines a conditional gaussian
    N(loc=BayesianMlp(x), scale=BoundedOutputBayesianMlp(x)) '''

    def __init__(self, loc_args, scale_args,
                 LocClass=BayesianMlp, ScaleClass=BoundedBayesianMlp,
                 options=None, name='HeteroscedasticGaussianMlp'):
        super(HeteroscedasticNormalMlp, self)\
            .__init__(loc_args, scale_args,
                      LocClass, ScaleClass,
                      options=options, name=name)


class Particles(common.TdlModel):
    @property
    def n_particles(self):
        return self._n_particles

    @property
    def base_shape(self):
        return self._base_shape

    @property
    def value(self):
        return self._value

    @common.SimpleParameter
    def base(self, value):
        if value is None:
            return tf.Variable(tf.random_normal(shape=self.base_shape))
        else:
            return value

    def _evaluate(self):
        value = tf.tile(self.base, multiples=[self.n_particles, 1])
        return value

    def __init__(self, n_particles, base=None, shape=None,
                 name='Particles', **kargs):
        self._n_particles = n_particles
        self._base_shape = (base.shape if base is not None
                            else shape)

        super(Particles, self).__init__(name=name, base=base, **kargs)
        with tf.name_scope(self.scope):
            self._value = self._evaluate()


@tdl.core.create_init_docstring
class ParticlesLayer(tdl.core.Layer):
    @tdl.core.InputArgument
    def input_shape(self, value):
        '''Input tensor shape.'''
        if value is None:
            raise tdl.core.exceptions.ArgumentNotProvided(self)
        if not isinstance(value, tf.TensorShape):
            value = tf.TensorShape(value)
        return value

    @tdl.core.InputArgument
    def particles(self, value):
        '''Number of particles.'''
        if value is None:
            raise tdl.core.exceptions.ArgumentNotProvided(self)
        return value

    def compute_output_shape(self, input_shape=None):
        if input_shape is None:
            tdl.core.assert_initialized(
                self, 'copute_output_shape',
                ['input_shape', 'particles'])
            input_shape = self.input_shape
        input_shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([self.particles] + input_shape)

    def call(self, inputs):
        try:
            inputs = tf.convert_to_tensor(inputs)
        except TypeError:
            return inputs.sample(sample_shape=[self.particles])
        multiples = [self.particles] + [1]*inputs.shape.ndims
        return tf.tile(inputs[tf.newaxis, ...],
                       multiples)


class McNormalEstimate(tdl.core.Layer):
    @tdl.core.InputArgument
    def input_shape(self, value):
        '''Input tensor shape.'''
        if value is None:
            raise tdl.core.exceptions.ArgumentNotProvided(self)
        return tf.TensorShape(value)

    @tdl.core.InputArgument
    def sample_dim(self, value):
        '''Sample dimension across which mean and stddev are computed.'''
        if value is None:
            value = 0
        return value

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        if tdl.core.is_property_initialized(self, 'input_shape'):
            assert self.input_shape.is_compatible_with(inputs.shape)
        else:
            self.input_shape = inputs.shape
        # mean
        loc = tf.reduce_mean(inputs, axis=self.sample_dim)
        # stddev
        N = tf.cast(tf.shape(inputs)[0], tf.float32)
        diff = (inputs - loc)**2
        sample_variance = (tf.reduce_sum(diff, axis=self.sample_dim)
                           / (N - 1))
        scale = tf.sqrt(sample_variance)
        return tfp.distributions.Normal(loc=loc, scale=scale)
