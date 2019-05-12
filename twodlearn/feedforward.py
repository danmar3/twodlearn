#  ***********************************************************************
#   This file defines some common feedforward Neural Network
#   arquitectures:
#      - ConvNet: convolutional neural network
#      - MlpNet: multilayer perceptron
#
#   Wrote by: Daniel L. Marino (marinodl@vcu.edu)
#    Modern Heuristics Research Group (MHRG)
#    Virginia Commonwealth University (VCU), Richmond, VA
#    http://www.people.vcu.edu/~mmanic/
#
#   ***********************************************************************

from __future__ import division
from __future__ import print_function

import warnings
import functools
import collections
import numpy as np
import tensorflow as tf
import twodlearn as tdl
from twodlearn.normalizer import (Normalizer)
from twodlearn.losses import (Loss, EmpiricalLoss, AddNLosses,
                              ClassificationLoss, L2Regularizer,
                              L2Loss, EmpiricalWithRegularization,
                              ScaledLoss, EmpiricalLossWrapper)


class Options:
    def __init__(self, weight_initialization,
                 weight_initialization_alpha):
        self.weight_initialization = weight_initialization
        self.weight_initialization_alpha = weight_initialization_alpha


options = Options(weight_initialization='sum',
                  weight_initialization_alpha=1.0)

''' ------------------------- Activation functions ------------------------ '''


def selu01(x):
    ''' Self normalizing activation function
    Activation function proposed by Gunter Klambauer et. al.
    "Self-Normalizing Neural Networks", https://arxiv.org/abs/1706.02515
    '''
    with tf.name_scope('selu01'):
        alpha_01 = 1.6733
        lambda_01 = 1.0507
        x_pos = tf.nn.relu(x)
        x_neg = -tf.nn.relu(-x)
        y = lambda_01 * (x_pos + (alpha_01 * tf.exp(x_neg) - alpha_01))
    return y


def selu01_disc(x):
    ''' discontinuous selu '''
    alpha_01 = 1.6733
    lambda_01 = 1.0507
    x = x + alpha_01 * lambda_01
    x_pos = tf.nn.relu(x)

    return x_pos - alpha_01 * lambda_01


def selu01_disc2(x):
    ''' another version of discontinuous selu '''
    alpha_01 = 1.6733
    lambda_01 = 1.0507
    x = x + alpha_01
    x_pos = tf.nn.relu(x)

    return lambda_01 * x_pos - alpha_01 * lambda_01


def leaky_relu(x, leaky_slope=0.01):
    ''' leaky relu, with 0.01 slope for negative values'''
    x_pos = tf.nn.relu(x)
    x_neg = -tf.nn.relu(-x)

    return x_pos + leaky_slope * x_neg


''' -------------------------------- Layers ------------------------------- '''


class Transpose(tdl.core.TdlModel):
    @tdl.core.InferenceInput
    def inputs(self, value):
        return value

    @tdl.core.InputArgument
    def rightmost(self, value):
        if value is None:
            value = False
        return value

    @tdl.core.LazzyProperty
    def shape(self):
        tdl.core.assert_initialized(self, 'shape', ['value'])
        return self.value.shape

    @tdl.core.InputArgument
    def perm(self, value):
        tdl.core.assert_initialized(self, 'perm', ['rightmost'])
        if value is None:
            tdl.core.assert_any_available(self, 'axis', ['inputs'])
            x = tf.convert_to_tensor(self.inputs)
            if x.shape.ndims is not None:
                ndims = x.shape.ndims
                if self.rightmost:
                    left_axis = np.arange(ndims - 2)
                    right_axis = np.array([ndims-1, ndims-2])
                    value = np.concatenate([left_axis, right_axis], axis=0)
                else:
                    value = np.flip(np.arange(ndims), axis=0)
            else:
                ndims = tf.rank(x)
                if self.rightmost:
                    left_axis = tf.range(tf.rank(x) - 2)
                    right_axis = tf.convert_to_tensor([ndims-1, ndims-2])
                    value = tf.concat(left_axis, right_axis, axis=0)
                else:
                    value = tf.reverse(tf.range(ndims), axis=[0])
        else:
            assert self.rightmost is False
        return value

    @tdl.core.OutputValue
    def value(self, value):
        tdl.core.assert_initialized(self, 'value', ['inputs', 'perm'])
        return tf.transpose(self.inputs, perm=self.perm)


class TransposeLayer(tdl.core.TdlModel):
    @tdl.core.InputArgument
    def rightmost(self, value):
        if value is None:
            value = False
        return value

    @tdl.core.SubmodelInit
    def perm(self, inputs):
        tdl.core.assert_initialized(self, 'perm', ['rightmost'])
        x = tf.convert_to_tensor(inputs)
        if x.shape.ndims is not None:
            ndims = x.shape.ndims
            if self.rightmost:
                left_axis = np.arange(ndims - 2)
                right_axis = np.array([ndims-1, ndims-2])
                value = np.concatenate([left_axis, right_axis], axis=0)
            else:
                value = np.flip(np.arange(ndims), axis=0)
        else:
            ndims = tf.rank(x)
            if self.rightmost:
                left_axis = tf.range(tf.rank(x) - 2)
                right_axis = tf.convert_to_tensor([ndims-1, ndims-2])
                value = tf.concat(left_axis, right_axis, axis=0)
            else:
                value = tf.reverse(tf.range(ndims), axis=[0])
        return value

    def __call__(self, x, name=None):
        if not tdl.core.is_property_set(self, 'perm'):
            self.perm.init(inputs=x)
        return Transpose(inputs=x, perm=self.perm, name=name)


class AlexnetLayer(tdl.core.TdlModel):
    '''Creates a layer like the one used in (ImageNet Classification
    with Deep Convolutional Neural Networks).

    The format for filter_size is:
        [filter_size_dim0 , filter_size_dim1], it performs 2D convolution
    The format for n_maps is:
        [num_input_maps, num_output_maps]
    The format for pool_size is:
        [pool_size_dim0, pool_size_dim1]
    '''
    @tdl.core.SimpleParameter
    def weights(self, value):
        if value is None:
            initializer = tf.truncated_normal(
                shape=[self.filter_size[0], self.filter_size[1],
                       self.n_maps[0], self.n_maps[1]],
                stddev=0.1)
            value = tf.Variable(initializer, name='W')
        return value

    @tdl.core.SimpleParameter
    def bias(self, value):
        if value is None:
            initializer = tf.truncated_normal([self.n_maps[1]], stddev=0.1)
            value = tf.Variable(initializer, name='b')
        return value

    @property
    def filter_size(self):
        return self._filter_size

    @property
    def n_maps(self):
        return self._n_maps

    @property
    def pool_size(self):
        return self._pool_size

    @tdl.core.Regularizer
    def regularizer(self, scale=None):
        reg = tdl.losses.L2Regularizer(self.weights, scale=scale)
        return reg

    def __init__(self, filter_size, n_maps, pool_size, name=None):
        self._filter_size = filter_size
        self._n_maps = n_maps
        self._pool_size = pool_size
        self._n_inputs = n_maps[0]
        self._n_outputs = n_maps[1]
        super(AlexnetLayer, self).__init__(name=name)

    class AlexnetLayerSetup(tdl.core.ModelEvaluation):
        @property
        def weights(self):
            return self.model.weights

        @property
        def bias(self):
            return self.model.bias

        @property
        def pool_size(self):
            return self.model.pool_size

        def eval_layer(self, inputs):
            conv = tf.nn.conv2d(inputs,
                                self.weights,
                                strides=[1, 1, 1, 1],
                                padding='VALID')
            hidden = tf.nn.relu(conv + self.bias)

            # Perform Pooling if the size of the pooling layer is bigger than 1
            # note that the size of the pooling kernel and the stride is the same
            if (self.pool_size[0] == 1 and self.pool_size[1] == 1):
                return hidden

            else:
                pool = tf.nn.max_pool(hidden,
                                      ksize=[1, self.pool_size[0],
                                             self.pool_size[1], 1],
                                      strides=[1, self.pool_size[0],
                                               self.pool_size[1], 1],
                                      padding='VALID')
                return pool

        def __init__(self, model, inputs, name):
            self._model = model
            self._name = name
            with tf.name_scope(self.scope):
                self._y = self.eval_layer(inputs)

    def evaluate(self, inputs, name=None):
        if name is None:
            name = self.name
        return self.AlexnetLayerSetup(model=self,
                                      inputs=inputs,
                                      name=name)


class StridedConvLayer(object):
    '''Creates a convolutional layer that uses strided convolutions instead
    of pooling.

    The format for filter_size is:
        [filter_size_dim0 , filter_size_dim1], it performs 2D convolution
    The format for n_maps is:
        [num_input_maps, num_output_maps]

    The format for stride is:
        [stride_dim0, stride_dim1]
    '''

    def __init__(self, filter_size, n_maps, stride, name=''):
        self.name = name
        self.stride = stride

        with tf.name_scope(self.name) as scope:
            self.weights = tf.Variable(
                tf.truncated_normal([filter_size[0], filter_size[1],
                                     n_maps[0], n_maps[1]],
                                    stddev=0.1),
                name='W')

            self.bias = tf.Variable(tf.truncated_normal([n_maps[1]],
                                                        stddev=0.001),
                                    name='b')

    def evaluate(self, input_tensor, padding='VALID'):
        # Perform Convolution
        # the layers performs a 2D convolution, with a strides of 1
        with tf.name_scope(self.name) as scope:
            conv = tf.nn.conv2d(input_tensor, self.weights, strides=[
                1, self.stride[0], self.stride[1], 1], padding=padding)
            hidden = tf.nn.relu(conv + self.bias)

        return hidden


class ConvTransposeLayer(object):
    '''Creates a "deconvolutional" layer.

    The format for filter_size is:
        [filter_size_dim0 , filter_size_dim1], it performs 2D convolution
    The format for n_maps is:
        [num_input_maps, num_output_maps]

    The format for stride is:
        [stride_dim0, stride_dim1]
    '''

    def __init__(self, filter_size, n_maps, stride, afunction=None, name=''):
        self.name = name
        self.stride = stride
        self.n_in_maps = n_maps[0]
        self.n_out_maps = n_maps[1]

        if afunction is None:
            self.afunction = tf.nn.relu
        else:
            self.afunction = afunction
        with tf.name_scope(self.name) as scope:
            self.weights = tf.Variable(
                tf.truncated_normal([filter_size[0], filter_size[1], n_maps[1], n_maps[0]],
                                    stddev=0.1),
                name='W')

            self.bias = tf.Variable(tf.truncated_normal([n_maps[1]],
                                                        stddev=0.001),
                                    name='b')

    def evaluate(self, input_tensor, padding='SAME'):
        # Perform Convolution transpose
        # the layers performs a 2D convolution, with a strides of 1

        # conv = tf.nn.conv2d(input_tensor, self.weights, strides=[1, self.stride[0], self.stride[1], 1], padding= padding)
        in_shape = input_tensor.get_shape().as_list()
        out_shape = [in_shape[0],
                     in_shape[1] * self.stride[0],
                     in_shape[2] * self.stride[1],
                     self.n_out_maps]

        with tf.name_scope(self.name) as scope:
            deconv = tf.nn.conv2d_transpose(input_tensor,
                                            self.weights,
                                            output_shape=out_shape,
                                            strides=[1, self.stride[0],
                                                     self.stride[1], 1],
                                            padding=padding)

            hidden = self.afunction(deconv + self.bias)

        return hidden


class LinearLayer(tdl.core.TdlModel):
    '''Standard linear (W*X) fully connected layer'''
    @property
    def n_inputs(self):
        return self._n_inputs

    @property
    def n_units(self):
        return self._n_units

    @property
    def n_outputs(self):
        return self.n_units

    @property
    def alpha(self):
        ''' Gain used for '''
        return self._alpha

    @tdl.core.SimpleParameter
    def weights(self, value):
        return self._init_weights(alpha=value['alpha'])

    @tdl.core.Regularizer
    def regularizer(self, scale=None):
        with tf.name_scope(self.scope):
            reg = tdl.losses.L2Regularizer(self.weights, scale=scale)
        return reg

    def _init_sigma(self, init_method=None, alpha=None):
        ''' Defines several initialization methods for the
        linear layer '''
        if init_method == 'frob':
            sigma = np.sqrt((alpha * alpha) /
                            (self.n_inputs * self.n_units))
        elif init_method == 'sum':
            sigma = np.sqrt((alpha * alpha) /
                            (self.n_inputs + self.n_units))
        elif init_method == 'max':
            M = max(self.n_inputs, self.n_units)
            sigma = np.sqrt(alpha / M)
        elif init_method == 'singular':
            N = min(self.n_inputs, self.n_units)
            M = max(self.n_inputs, self.n_units)
            t = (N - 1) / (M - 1)
            t = 1 - np.exp(-10 * t)
            alpha = np.sqrt(np.minimum(N, M) * (alpha * ((1 - t) + 0.3 * t)))
            sigma = np.sqrt((alpha**2) / (N * M))
        return sigma

    def _init_weights(self, init_method=None, alpha=None, name='W'):
        if alpha is not None:
            self._alpha = alpha
        else:
            self._alpha = options.weight_initialization_alpha

        if init_method is None:
            self.weight_init_method = options.weight_initialization
        else:
            self.weight_init_method = init_method
        # weight initialization
        alpha = self.alpha  # 5.0
        sigma = self._init_sigma(self.weight_init_method, self.alpha)

        weights = tf.Variable(tf.truncated_normal([self.n_inputs,
                                                   self.n_units],
                                                  stddev=sigma),
                              name=name)
        return weights

    def __init__(self, n_inputs, n_units, alpha=None,
                 options=None, name=None):
        self._n_inputs = n_inputs
        self._n_units = n_units
        super(LinearLayer, self).__init__(weights={'alpha': alpha}, name=name,
                                          options=options)

    class Output(tdl.core.TdlModel):
        @property
        def y(self):
            return self.value

        @property
        def shape(self):
            return self.value.shape

        @property
        def n_units(self):
            return self.model.n_units

        @property
        def n_outputs(self):
            return self.n_units

        @property
        def weights(self):
            return self.model.weights

        @tdl.core.InputArgument
        def inputs(self, value):
            return value

        @tdl.core.OutputValue
        def value(self, _):
            return tf.matmul(self.inputs, self.weights)

        def __init__(self, model, inputs, options=None, name=None):
            self.model = model
            super(LinearLayer.Output, self).__init__(
                inputs=inputs, options=options, name=name)

    def evaluate(self, inputs, name=None):
        return LinearLayer.Output(self, inputs, name=name)

    def __call__(self, x):
        return self.evaluate(x)


class AffineLayer(LinearLayer):
    '''Standard affine (W*X+b) fully connected layer'''
    @property
    def parameters(self):
        ''' Returns the list of parameters for the layer'''
        return [self.weights, self.bias]

    @tdl.core.SimpleParameter
    def bias(self, value):
        return tf.Variable(tf.zeros([self.n_units]), name='b')

    def __init__(self, n_inputs, n_units, alpha=1.0,
                 options=None, name='AffineLayer'):
        super(AffineLayer, self).__init__(n_inputs, n_units, alpha,
                                          options=options, name=name)

    class Output(LinearLayer.Output):
        @property
        def bias(self):
            return self.model.bias

        @tdl.core.OutputValue
        def value(self, _):
            return tf.matmul(self.inputs, self.weights) + self.bias

    def evaluate(self, inputs, name=None):
        return AffineLayer.Output(self, inputs, name=name)


class DenseLayer(AffineLayer):
    '''Standard fully connected layer'''

    def __init__(self, n_inputs, n_units, afunction=tf.nn.relu, alpha=1.0,
                 options=None, name='DenseLayer'):
        super(DenseLayer, self).__init__(n_inputs, n_units, alpha,
                                         options=options, name=name)
        self.afunction = afunction

    class Output(AffineLayer.Output):
        @property
        def z(self):
            ''' activation before non-linearity '''
            return self.affine

        @property
        def y(self):
            ''' activation before non-linearity '''
            return self.value

        @property
        def afunction(self):
            return self.model.afunction

        @tdl.core.Submodel
        def affine(self, _):
            return tf.matmul(self.inputs, self.weights) + self.bias

        @tdl.core.OutputValue
        def value(self, _):
            return self.afunction(self.affine)

    def evaluate(self, inputs, options=None, name=None):
        return DenseLayer.Output(self, inputs, options=options, name=name)


''' ----------------------------- Networks -------------------------------- '''


class NetConf(object):
    '''This is a wrapper to any network configuration, it contains the
    references to the placeholders for inputs and labels, and the
    reference of the computation graph for the network

    inputs: placeholder for the inputs
    labels: placeholder for the labels
    y: output of the comptuation graph, ussually a linear map
       from the last layer (logits)
    loss: loss for the network
    '''

    def __init__(self, inputs, labels, y, loss):
        self.inputs = inputs
        self.labels = labels
        self.y = y  # TODO: change this to out
        self.loss = loss


class MultiLayer2DConvolution(tdl.core.TdlModel):
    ''' Creates a Convolutional neural network

    It performs a series of 2d Convolutions and pooling operations

    input_size: size of the input maps, [size_dim0, size_dim1]
    n_outputs: number of outputs
    n_input_maps: number of input maps
    n_filters: list with the number of filters for layer
    filter_size: list with the size of the kernel for each layer,
                 the format for the size of each layer is:
                 [filter_size_dim0 , filter_size_dim1]
    pool_size: list with the size of the pooling kernel foreach layer,
               the format for each layer is: [pool_size_dim0, pool_size_dim1]
    '''
    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def n_filters(self):
        return self._n_filters

    @property
    def filter_sizes(self):
        return self._filter_sizes

    @property
    def pool_sizes(self):
        return self._pool_sizes

    @property
    def weights(self):
        weights = [layer.weights for layer in self.layers]
        return weights

    @tdl.core.Submodel
    def layers(self, _):
        layers = list()
        _n_inputs = self.input_shape[-1]
        for idx, n_units in enumerate(self.n_filters):
            layers.append(
                AlexnetLayer(self.filter_sizes[idx],
                             [_n_inputs, n_units],
                             self.pool_sizes[idx],
                             name='conv_{}'.format(idx)))
            _n_inputs = n_units
        return layers

    @tdl.core.Regularizer
    def regularizer(self, scale=None):
        reg = [(layer.regularizer.value if layer.regularizer.is_set
                else layer.regularizer.init(scale))
               for layer in self.layers
               if hasattr(layer, 'regularizer')]
        if reg:
            reg = (reg[0] if len(reg) == 1
                   else tdl.losses.AddNLosses(reg))
        else:
            raise AttributeError(
                'None of the Layers has a regularizer defined')
        return reg

    def __init__(self, input_shape, n_filters, filter_sizes, pool_sizes,
                 name='MultiConv2D'):
        ''' All variables corresponding to the weights of the network are defined
        '''
        assert len(input_shape) == 3, \
            'input_shape must have 3 elements: '\
            'input_shape=[input_height, input_widht, input_maps]'
        input_size = input_shape[:2]
        n_input_maps = input_shape[2]
        self._n_inputs = n_input_maps
        self._n_outputs = n_filters[-1]
        self._input_shape = input_shape
        self._n_filters = n_filters
        self._filter_sizes = filter_sizes
        self._pool_sizes = pool_sizes

        super(MultiLayer2DConvolution, self).__init__(name=name)

        # Get size after convolution phase
        final_size = [input_size[0], input_size[1]]
        for i in range(len(filter_sizes)):
            final_size[0] = (final_size[0] -
                             (filter_sizes[i][0] - 1)) // pool_sizes[i][0]
            final_size[1] = (final_size[1] -
                             (filter_sizes[i][1] - 1)) // pool_sizes[i][1]

        if final_size[0] == 0:
            final_size[0] = 1
        if final_size[1] == 0:
            final_size[1] = 1
        self._output_shape = final_size + [self.n_filters[-1]]
        # print("Shape of the maps after convolution stage:", self.out_conv_shape)

    class Output(tdl.core.ModelEvaluation):

        @property
        def weights(self):
            return self.model.weights

        @property
        def hidden(self):
            return self._hidden

        @property
        def input_shape(self):
            return self.model.input_shape

        def setup_inputs(self, batch_size, input_shape):
            inputs = tf.placeholder(tf.float32,
                                    shape=[batch_size] + input_shape,
                                    name='inputs')
            return inputs

        def setup_conv_layers(self, inputs, layers):
            out = inputs
            hidden = list()
            for layer in layers:
                h = layer.evaluate(out)
                hidden.append(h)
                out = h.y

            return out, hidden

        def __init__(self, model, inputs=None, batch_size=None,
                     options=None, name='MultiConv2D'):
            super(MultiLayer2DConvolution.Output, self)\
                .__init__(model, options=options, name=name)

            with tf.name_scope(self.scope):
                if inputs is None:
                    inputs = self.setup_inputs(batch_size,
                                               self.input_shape)
                self._inputs = inputs
                self._y, self._hidden = \
                    self.setup_conv_layers(self.inputs,
                                           self.model.layers)

    def setup(self, inputs=None, batch_size=None, options=None, name=None):
        return MultiLayer2DConvolution.Output(model=self,
                                              inputs=inputs,
                                              batch_size=batch_size,
                                              options=options,
                                              name=name)


class AlexNet(tdl.core.TdlModel):
    _submodels = ['conv', 'mlp']

    @property
    def input_shape(self):
        return self.conv.input_shape

    @property
    def n_outputs(self):
        return self.mlp.n_outputs

    @property
    def weights(self):
        return self.conv.weights + self.mlp.weights

    @tdl.core.Submodel
    def conv(self, value):
        if isinstance(value, dict):
            conv = MultiLayer2DConvolution(
                input_shape=value['input_shape'],
                n_filters=value['n_filters'],
                filter_sizes=value['filter_sizes'],
                pool_sizes=value['pool_sizes'],
                name='conv')
        elif isinstance(value, MultiLayer2DConvolution):
            conv = value
        else:
            raise ValueError('Provided network is not a '
                             'MultiLayer2DConvolution')
        return conv

    @tdl.core.Submodel
    def mlp(self, value):
        n_inputs = functools.reduce(lambda x, y: x * y,
                                    self.conv.output_shape)
        if isinstance(value, dict):
            net = MlpNet(n_inputs=n_inputs,
                         n_outputs=value['n_outputs'],
                         n_hidden=value['n_hidden'],
                         afunction=tf.nn.relu,
                         output_function=value['output_function'],
                         name='mlp')
        elif isinstance(value, MlpNet):
            net = value
        else:
            raise ValueError('Provided network is not an MlpNet')
        return net

    @tdl.core.Regularizer
    def regularizer(self, scale=None):
        conv_reg = (self.conv.regularizer.value if self.conv.regularizer.is_set
                    else self.conv.regularizer.init(scale))
        mlp_reg = (self.mlp.regularizer.value if self.mlp.regularizer.is_set
                   else self.mlp.regularizer.init(scale))
        return tdl.losses.AddNLosses([conv_reg, mlp_reg])

    def __init__(self, input_shape, n_outputs, n_filters, filter_sizes,
                 pool_sizes, n_hidden, output_function=None, name='AlexNet'):

        super(AlexNet, self).__init__(
            conv={'input_shape': input_shape, 'n_filters': n_filters,
                  'filter_sizes': filter_sizes, 'pool_sizes': pool_sizes},
            mlp={'n_outputs': n_outputs, 'n_hidden': n_hidden,
                 'output_function': output_function},
            name=name)

    class AlexNetSetup(tdl.core.TdlModel):
        _submodels = ['conv', 'mlp']

        @property
        def weights(self):
            return self.model.weights

        @tdl.core.InputArgument
        def inputs(self, value):
            if value is None:
                value = tf.placeholder(tdl.core.global_options.float.tftype,
                                       shape=[None] + self.input_shape,
                                       name='inputs')
            return value

        @tdl.core.Submodel
        def conv(self, _):
            conv = self.model.conv.setup(self.inputs)
            return conv

        @tdl.core.Submodel
        def mlp(self, _):
            inputs = tf.reshape(self.conv.y, [-1, self.model.mlp.n_inputs])
            mlp = self.model.mlp.evaluate(inputs=inputs)
            return mlp

        @tdl.core.OutputValue
        def output(self, _):
            return self.mlp.output

        @property
        def value(self):
            return self.output.value

        @property
        def input_shape(self):
            return self.model.input_shape

        @tdl.core.OptionalProperty
        def loss(self, alpha=1e-5):
            empirical = L2Loss(self.value)
            regularizer = (self.model.regularizer.value
                           if self.model.regularizer.is_set
                           else self.model.regularizer.init())
            loss = EmpiricalWithRegularization(
                empirical=empirical,
                regularizer=regularizer,
                alpha=alpha)
            return loss

        def __init__(self, model, inputs=None, batch_size=None,
                     options=None, name='AlexNet'):
            self.model = model
            super(AlexNet.AlexNetSetup, self)\
                .__init__(inputs=inputs, options=options, name=name)

    def evaluate(self, inputs=None, options=None, name=None):
        return AlexNet.AlexNetOutput(model=self, inputs=inputs,
                                     options=options, name=name)


class AlexNetClassifier(AlexNet):
    @property
    def n_classes(self):
        return self._n_classes

    @tdl.core.Submodel
    def mlp(self, value):
        n_inputs = functools.reduce(lambda x, y: x * y,
                                    self.conv.output_shape)
        if isinstance(value, dict):
            net = MlpClassifier(
                n_inputs=n_inputs,
                n_classes=self.n_classes,
                n_hidden=value['n_hidden'],
                afunction=tf.nn.relu,
                name='mlp')
        elif isinstance(value, MlpNet):
            net = value
        else:
            raise ValueError('Provided network is not an MlpNet')
        return net

    def __init__(self, input_shape, n_classes, n_filters, filter_sizes,
                 pool_sizes, n_hidden, name='AlexNetClassifier'):
        self._n_classes = n_classes
        n_outputs = (1 if n_classes == 2
                     else n_classes)
        super(AlexNetClassifier, self).__init__(input_shape=input_shape,
                                                n_outputs=n_outputs,
                                                n_filters=n_filters,
                                                filter_sizes=filter_sizes,
                                                pool_sizes=pool_sizes,
                                                n_hidden=n_hidden,
                                                name=name)

    class AlexNetOutput(AlexNet.AlexNetSetup):
        @property
        def logits(self):
            return self.mlp.logits

        @tdl.core.OptionalProperty
        def loss(self, alpha=1e-5):
            empirical = ClassificationLoss(self.logits)
            regularizer = (self.model.regularizer.value
                           if self.model.regularizer.is_set
                           else self.model.regularizer.init())
            loss = EmpiricalWithRegularization(
                empirical=empirical,
                regularizer=regularizer,
                alpha=alpha)
            return loss

    def evaluate(self, inputs=None, options=None, name=None):
        return AlexNetClassifier\
            .AlexNetOutput(model=self, inputs=inputs, options=options,
                           name=name)


class ConvNet(object):
    ''' Creates a Convolutional neural network
    It creates a convolutional neural network similar to the one used
    in (ImageNet Classification with Deep Convolutional Neural Networks)

    It performs a series of 2d Convolutions and pooling operations, then
    a standard fully connected stage and finaly a softmax

    input_size: size of the input maps, [size_dim0, size_dim1]
    n_outputs: number of outputs
    n_input_maps: number of input maps
    n_filters: list with the number of filters for layer
    filter_size: list with the size of the kernel for each layer,
                 the format for the size of each layer is:
                 [filter_size_dim0 , filter_size_dim1]
    pool_size: list with the size of the pooling kernel foreach layer,
               the format for each layer is: [pool_size_dim0, pool_size_dim1]
    n_hidden: list with the number of units on each fully connected layer

    out_conv_shape: size of the output map from the convolution stage,
                    [size_dim0, size_dim1]

    '''

    def __init__(self, input_size, n_input_maps, n_outputs,
                 n_filters, filter_size,
                 pool_size,
                 n_hidden=[],
                 name=''):
        ''' All variables corresponding to the weights of the network are defined
        '''
        self.input_size = input_size
        self.n_input_maps = n_input_maps
        self.n_outputs = n_outputs
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.n_hidden = n_hidden

        # 1. Create the convolutional layers:
        self.conv_layers = list()
        self.conv_layers.append(
            AlexnetLayer(filter_size[0],
                         [n_input_maps, n_filters[0]],
                         pool_size[0],
                         name=name + '/conv_0'))

        for l in range(1, len(n_filters)):
            self.conv_layers.append(
                AlexnetLayer(filter_size[l],
                             [n_filters[l - 1], n_filters[l]],
                             pool_size[l],
                             name=name + '/conv_' + str(l))
            )

        # Get size after convolution phase
        final_size = [input_size[0], input_size[1]]
        for i in range(len(filter_size)):
            final_size[0] = (final_size[0] - (filter_size[i]
                                              [0] - 1)) // pool_size[i][0]
            final_size[1] = (final_size[1] - (filter_size[i]
                                              [1] - 1)) // pool_size[i][1]

        if final_size[0] == 0:
            final_size[0] = 1
        if final_size[1] == 0:
            final_size[1] = 1
        self.out_conv_shape = final_size
        print("Shape of the maps after convolution stage:", self.out_conv_shape)

        # 2. Create the fully connected layers:
        if len(n_hidden) > 0:
            self.full_layers = list()
            self.full_layers.append(
                DenseLayer(final_size[0] * final_size[1] * n_filters[-1],
                           n_hidden[0],
                           name=name + '/full_0'))
            for l in range(1, len(n_hidden)):
                self.full_layers.append(
                    DenseLayer(n_hidden[l - 1],
                               n_hidden[l],
                               name=name + '/full_' + str(l))
                )

            # 3. Create the final layer:
            self.out_layer = AffineLayer(
                n_hidden[-1], n_outputs, name=name + '/lin')

        elif (n_outputs is not None):
            # 3. Create the final layer: (TODO: test this!!!!!!)
            self.out_layer = AffineLayer(final_size[0] * final_size[1] * n_filters[-1],
                                         n_outputs,
                                         name=name + '/lin')

        # 4. Define the saver for the weights of the network
        saver_dict = dict()
        for l in range(len(self.conv_layers)):
            saver_dict.update(self.conv_layers[l].saver_dict)

        if len(n_hidden) != 0:
            for l in range(len(self.full_layers)):
                saver_dict.update(self.full_layers[l].saver_dict)
        elif (n_outputs is not None):
            saver_dict.update(self.out_layer.saver_dict)

        self.saver = tf.train.Saver(saver_dict)

    def setup(self, batch_size, drop_prob=None, l2_reg_coef=None, loss_type=None):
        ''' Defines the computation graph of the neural network for a specific
        batch size

        drop_prob: placeholder used for specify the probability for dropout.
                   If this coefficient is set, then dropout regularization is
                   added between all fully connected layers
                   (TODO: allow to choose which layers)
        l2_reg_coef: coeficient for l2 regularization
        loss_type: type of the loss being used for training the network,
                   the options are:
                - 'cross_entropy': for classification tasks
                - 'l2': for regression tasks
        '''
        inputs = tf.placeholder(tf.float32,
                                shape=(batch_size, self.input_size[0], self.input_size[1], self.n_input_maps))

        if (loss_type is not None) or (len(self.n_hidden) == 0):
            labels = tf.placeholder(
                tf.float32, shape=(batch_size, self.n_outputs))
        else:
            labels = None

        # 1. convolution stage
        out = inputs
        for layer in self.conv_layers:
            out = layer.evaluate(out)

        # 2. fully connected stage
        # 2.1 reshape
        shape = out.get_shape().as_list()
        print('Shape of input matrix entering to Fully connected layers:', shape)
        out = tf.reshape(out, [shape[0], shape[1] * shape[2] * shape[3]])

        # if no fully connected layers, return here:
        if (len(self.n_hidden) == 0 and self.n_outputs is None):
            return NetConf(inputs, None, out, None)
        elif len(self.n_hidden) == 0:  # TODO: check and add loss in this case
            out = self.out_layer.evaluate(out)
            return NetConf(inputs, None, out, None)

        # 2.2 mlp
        for layer in self.full_layers:
            out = layer.evaluate(out)
            if drop_prob is not None:
                out = tf.nn.dropout(out, drop_prob)

        # 3. linear stage
        y = self.out_layer.evaluate(out)

        # 4. loss # TODO: add number of parameters to loss so hyperparameters are more easy to tune, also put None as default and do not calculate loss if it is None
        # l2 regularizer
        l2_reg = 0
        if l2_reg_coef is not None:
            for layer in self.full_layers:
                l2_reg += tf.nn.l2_loss(layer.weights)
            l2_reg = l2_reg_coef * l2_reg

        # loss
        if loss_type is None:
            loss = None
        elif loss_type == 'cross_entropy':
            if self.n_outputs == 1:
                loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                            logits=y)) + l2_reg
            else:
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                            logits=y)) + l2_reg
        elif loss_type == 'l2':
            loss = tf.reduce_mean(tf.nn.l2_loss(y - labels)) + l2_reg

        return NetConf(inputs, labels, y, loss)


class LinearClassifier(tdl.core.TdlModel):
    @property
    def n_inputs(self):
        return self._n_inputs

    @property
    def n_outputs(self):
        return (1 if self.n_classes == 2
                else self.n_classes)

    @property
    def n_classes(self):
        return self._n_classes

    @tdl.core.Submodel
    def linear_layer(self, value):
        if value is None:
            value = AffineLayer(self.n_inputs, self.n_outputs)
        return value

    @tdl.core.Regularizer
    def regularizer(self, scale):
        reg = (self.linear_layer.regularizer.value
               if self.linear_layer.regularizer.is_set
               else self.linear_layer.regularizer.init(scale))
        return reg

    @property
    def weights(self):
        return [self.linear_layer.weights]

    def __init__(self, n_inputs, n_classes, name='linear_classifier',
                 **kargs):
        self._n_inputs = n_inputs
        self._n_classes = n_classes
        super(LinearClassifier, self).__init__(name=name, **kargs)

    class LinearClassifierSetup(tdl.core.OutputModel):
        @property
        def n_inputs(self):
            return self.model.n_inputs

        @property
        def n_outputs(self):
            return self.model.n_outputs

        @property
        def labels(self):
            return self._labels

        @tdl.core.OptionalProperty
        def loss(self, alpha):
            empirical = ClassificationLoss(self.logits)
            loss = EmpiricalWithRegularization(
                empirical=empirical,
                regularizer=self.model.regularizer.value,
                alpha=alpha)
            return loss

        @property
        def weights(self):
            return self.model.weights

    @tdl.ModelMethod(['logits', 'value', 'inputs'], ['inputs'],
                     LinearClassifierSetup)
    def evaluate(self, object, inputs=None):
        if inputs is None:
            inputs = tf.placeholder(tf.float32,
                                    shape=(None, self.n_inputs),
                                    name='inputs')
        logits = self.linear_layer(inputs)
        if self.n_outputs == 1:
            output = tf.nn.sigmoid(logits)
        else:
            output = tf.nn.softmax(logits)
        return logits, output, inputs


class StridedConvNet(object):   # TODO!!!!!!!!!!!!!!!!!!!
    ''' Creates a Convolutional neural network using strided convolutions.
    It does not use pooling

    It performs a series of 2d strided Convolution operations, then
    a standard fully connected stage and finaly an affine mapping

    input_size: size of the input maps: [size_dim0, size_dim1]
    n_outputs: number of outputs
    n_input_maps: number of input maps
    n_filters: list with the number of filters for layer
    filter_size: list with the size of the kernel for each layer,
                 the format for the size of each layer is:
                 [filter_size_dim0 , filter_size_dim1]
    strides: list with the size of the strides for each layer,
               the format for each layer is: [stride_dim0, stride_dim1]
    n_hidden: list with the number of units on each fully connected layer

    out_conv_shape: size of the output map from the convolution stage:
                    [size_dim0, size_dim1]

    '''

    def define_fullyconnected_layers(self):
        ''' defines the fully connected layers for the architecture
        This function populates the self.full_layers list and
        self.out_layer variable
        '''
        conv_as_vector_size = (self.out_conv_shape[0] *
                               self.out_conv_shape[1] *
                               self.n_filters[-1])
        self.full_layers = list()
        if self.n_hidden:
            self.full_layers.append(
                DenseLayer(conv_as_vector_size,
                           self.n_hidden[0],
                           name='Dense_0'))
            for l in range(1, len(self.n_hidden)):
                self.full_layers.append(
                    DenseLayer(self.n_hidden[l - 1],
                               self.n_hidden[l],
                               name='Dense_{}'.format(l)))

            # 3. Create the final layer:
            self.out_layer = AffineLayer(self.n_hidden[-1],
                                         self.n_outputs,
                                         name='lin')

        elif (self.n_outputs is not None):
            # 3. Create the final layer: (TODO: test this!!!!!!)
            self.out_layer = AffineLayer(conv_as_vector_size,
                                         self.n_outputs,
                                         name='lin')

    def define_conv_layers(self):
        ''' defines the convolution layers for the architecture
        This function populates the self.conv_layers list
        '''
        self.conv_layers = list()
        self.conv_layers.append(
            StridedConvLayer(self.filter_size[0],
                             [self.n_input_maps, self.n_filters[0]],
                             self.strides[0], name='conv_0'))

        for l in range(1, len(self.n_filters)):
            self.conv_layers.append(
                StridedConvLayer(self.filter_size[l],
                                 [self.n_filters[l - 1], self.n_filters[l]],
                                 self.strides[l],
                                 name='conv_{}'.format(str(l))))

    def __init__(self, input_size, n_input_maps, n_outputs,
                 n_filters, filter_size,
                 strides,
                 n_hidden=[],
                 name=''):
        ''' All variables corresponding to the weights of the network are defined
        '''
        self.input_size = input_size
        self.n_input_maps = n_input_maps
        self.n_outputs = n_outputs
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.strides = strides
        self.n_hidden = n_hidden
        self.name = name

        with tf.name_scope(self.name) as scope:
            # 1. Create the convolutional layers:
            self.define_conv_layers()

            # Get size after convolution phase
            final_size = [input_size[0], input_size[1]]
            for i in range(len(filter_size)):
                final_size[0] = (
                    final_size[0] - (filter_size[i][0] - 1)) // strides[i][0]
                final_size[1] = (
                    final_size[1] - (filter_size[i][1] - 1)) // strides[i][1]

            if final_size[0] == 0:
                final_size[0] = 1
            if final_size[1] == 0:
                final_size[1] = 1
            self.out_conv_shape = final_size
            print("Shape of the maps after convolution stage:", self.out_conv_shape)

            # 2. Create the fully connected layers:
            self.define_fullyconnected_layers()

        # 4. Define the saver for the weights of the network
        saver_dict = dict()
        for l in range(len(self.conv_layers)):
            saver_dict.update(self.conv_layers[l].saver_dict)

        if len(n_hidden) != 0:
            for l in range(len(self.full_layers)):
                saver_dict.update(self.full_layers[l].saver_dict)
        elif (n_outputs is not None):
            saver_dict.update(self.out_layer.saver_dict)

        self.parameters = saver_dict
        self.saver = tf.train.Saver(saver_dict)

    def setup(self, batch_size, drop_prob=None, l2_reg_coef=None,
              loss_type=None, inputs=None):
        ''' Defines the computation graph of the neural network for a specific
            batch size

        drop_prob: placeholder used for specify the probability for dropout. If
            this coefficient is set, then dropout regularization is added
            between all fully connected layers(TODO: allow to choose which
            layers)
        l2_reg_coef: coeficient for l2 regularization
        loss_type: type of the loss being used for training the network, the
            options are:
                - 'cross_entropy': for classification tasks
                - 'l2': for regression tasks
        '''
        if inputs is None:
            inputs = tf.placeholder(
                dtype=tf.float32,
                shape=(batch_size, self.input_size[0], self.input_size[1],
                       self.n_input_maps))

        if (loss_type is not None) or (len(self.n_hidden) == 0):
            labels = tf.placeholder(
                tf.float32, shape=(batch_size, self.n_outputs))
        else:
            labels = None

        # 1. convolution stage
        out = inputs
        for layer in self.conv_layers:
            out = layer.evaluate(out)

        # 2. fully connected stage
        # 2.1 reshape
        shape = out.get_shape().as_list()
        print('Shape of input matrix entering to Fully connected layers:',
              shape)
        out = tf.reshape(out, [shape[0], shape[1] * shape[2] * shape[3]])

        # if no fully connected layers, return here:
        if (len(self.n_hidden) == 0 and self.n_outputs is None):
            return NetConf(inputs, None, out, None)

        # 2.2 mlp
        for layer in self.full_layers:
            out = layer.evaluate(out)
            if drop_prob is not None:
                out = tf.nn.dropout(out, drop_prob)

        # 3. linear stage
        y = self.out_layer.evaluate(out)

        # 4. loss # TODO: add number of parameters to loss so hyperparameters are more easy to tune, also put None as default and do not calculate loss if it is None
        # l2 regularizer
        l2_reg = 0
        if l2_reg_coef is not None:
            for layer in self.full_layers:
                l2_reg += tf.nn.l2_loss(layer.weights)
            l2_reg = l2_reg_coef * l2_reg

        # loss
        if loss_type is None:
            loss = None
        elif loss_type == 'cross_entropy':
            if self.n_outputs == 1:
                loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                            logits=y)) + l2_reg
            else:
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                            logits=y)) + l2_reg
        elif loss_type == 'l2':
            loss = tf.reduce_mean(tf.nn.l2_loss(y - labels)) + l2_reg

        return NetConf(inputs, labels, y, loss)


class StridedDeconvNet(object):
    ''' Creates a Deconvolutional neural network using upsampling
    TODO: implement this using new format
    It performs a \'deconvolutional\' neural network similar to the one used
    in "UNSUPERVISED REPRESENTATION LEARNING WIT\H DEEP CONVOLUTIONAL
    GENERATIVE ADVERSARIAL NETWORKS"
    (http://arxiv.org/pdf/1511.06434v2.pdf)

    The network maps a vector of size n_inputs to a 2d map with several chanels

    First a linear mapping is performed, then a reshape to form an initial
    tensor of 2d maps with chanels, then a series of upscaling and convolutions
    are performed

    n_inputs: size of the input vectors
    input_size: size of the maps after linear stage: [size_dim0, size_dim1]
    n_input_maps: number of maps after linear stage
    n_filters: list with the number of filters for each layer
    filter_size: list with the size of the kernel for each layer,
                 the format for the size of each layer is:
                 [filter_size_dim0 , filter_size_dim1]
    upsampling: list with the size for the upsampling in each deconv layer:
                [upsampling_dim0, upsampling_dim1]



    in_layer: input layer, a linear layer for mapping the inputs to the desired
        output
    '''

    def __init__(self, n_inputs, input_size, n_input_maps,
                 n_filters, filter_size,
                 upsampling,
                 name=''):
        ''' All variables corresponding to the weights of the network are
        defined '''
        self.n_inputs = n_inputs
        self.input_size = input_size
        self.n_input_maps = n_input_maps
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.upsampling = upsampling

        # 1. Create the linear layer
        self.in_layer = LinearLayer(n_inputs,
                                    n_input_maps *
                                    input_size[0] * input_size[1],
                                    name=name + '/lin')

        # 2. Create the convolutional layers:
        self.conv_layers = list()
        self.conv_layers.append(
            ConvTransposeLayer(filter_size[0],
                               [n_input_maps, n_filters[0]],
                               upsampling[0],
                               name=name + '/conv_0'))
        for l in range(1, len(n_filters) - 1):
            self.conv_layers.append(
                ConvTransposeLayer(filter_size[l],
                                   [n_filters[l - 1], n_filters[l]],
                                   upsampling[l],
                                   name=name + '/conv_' + str(l)))

        # last conv layer has tanh activation function
        self.conv_layers.append(
            ConvTransposeLayer(filter_size[-1],
                               [n_filters[-2], n_filters[-1]],
                               upsampling[-1],
                               afunction=tf.tanh,
                               name=name + '/conv_' + str(len(n_filters) - 1)))

        # 4. Define the saver for the weights of the network
        saver_dict = dict()
        for l in range(len(self.conv_layers)):
            saver_dict.update(self.conv_layers[l].saver_dict)

        self.saver = tf.train.Saver(saver_dict)

    def setup(self, batch_size, drop_prob=None):
        ''' Defines the computation graph of the neural network for a specific
            batch size

        drop_prob: placeholder used for specify the probability for dropout. If
            this coefficient is set, then dropout regularization is added
            between all fully connected layers(TODO: allow to choose which
            layers)
        '''
        inputs = tf.placeholder(tf.float32,
                                shape=(batch_size, self.n_inputs))

        # 1. linear stage
        out = self.in_layer.evaluate(inputs)

        # 1.1 reshape
        shape = out.get_shape().as_list()
        out = tf.reshape(out, [shape[0],
                               self.input_size[0],
                               self.input_size[1],
                               self.n_input_maps]
                         )

        # 2. convolution stage
        for layer in self.conv_layers:

            out = layer.evaluate(out, 'SAME')

        return NetConf(inputs, None, out, None)


class StackedModel(tdl.TdlModel):
    _submodels = ['layers']

    @tdl.Submodel
    def layers(self, value):
        if value is None:
            value = list()
        return value

    @tdl.core.LazzyProperty
    def _layer_names(self):
        ''' name of the layers '''
        return dict()

    def __getitem__(self, item):
        return self.layers[item]

    def __len__(self):
        return len(self.layers)

    def add(self, layer, name=None):
        assert callable(layer), \
            'Model {} is not callable. StackedModel only works with '\
            'callable models'.format(layer)
        if name is not None:
            if not isinstance(name, str):
                raise TypeError('name should be a string')
            if name in self._layer_names:
                raise ValueError('a layer with the same name has been '
                                 'previously added')
            self._layer_names[name] = len(self.layers)
        self.layers.append(layer)
        return layer

    @tdl.core.Regularizer
    def regularizer(self, scale=None):
        reg = [(layer.regularizer.value if layer.regularizer.is_set
                else layer.regularizer.init(scale))
               for layer in self.layers
               if hasattr(layer, 'regularizer')]
        if reg:
            reg = (reg[0] if len(reg) == 1
                   else tdl.losses.AddNLosses(reg))
        else:
            raise AttributeError(
                'None of the Layers has a regularizer defined')
        return reg

    def get_save_data(self):
        init_args = {'layers': tdl.core.save.get_save_data(self.layers),
                     'name': self.scope}
        data = tdl.core.save.ModelData(cls=type(self),
                                       init_args=init_args)
        return data

    class StackedOutput(tdl.core.OutputModel):
        def __getattr__(self, value):
            if not tdl.core.get_context(self).initialized:
                raise AttributeError('Unable to get property {}, model {} has '
                                     'not been initialized.'.format(self))
            if value not in self.model._layer_names:
                raise AttributeError('model {} does not have {} property.'
                                     ''.format(self, value))
            idx = self.model._layer_names[value]
            if idx == len(self.hidden):
                return self.output
            return self.hidden[idx]

    @tdl.ModelMethod(['output', 'hidden', 'value', 'shape'], ['inputs'],
                     StackedOutput)
    def evaluate(self, object, inputs):
        hidden = list()
        x = inputs
        for layer in self.layers:
            # x = layer(tf.convert_to_tensor(x))
            x = layer(x)
            hidden.append(x)
        y = hidden[-1]
        hidden = hidden[:-1]
        try:
            value = tf.convert_to_tensor(y)
        except ValueError:
            value = None
        shape = (value.shape if value is not None
                 else None)
        return y, hidden, value, shape

    def __call__(self, x, name=None):
        return self.evaluate(inputs=x, name=name)

    def __init__(self, layers=None, options=None, name='Stacked'):
        super(StackedModel, self).__init__(
            layers=layers, options=options, name=name)


class ParallelModel(tdl.TdlModel):
    @property
    def models(self):
        return self._models

    def __init__(self, models, name='Parallel'):
        assert isinstance(models, collections.Iterable), \
            'Provided models is not a list'
        self._models = models
        super(ParallelModel, self).__init__(name=name)

    @tdl.ModelMethod(['output', 'value'], ['inputs'])
    def evaluate(self, object, inputs):
        assert isinstance(inputs, collections.Iterable), \
            'Provided input for ParallelModel {} is not iterable'\
            ''.format(self)
        assert len(inputs) == len(self.models),\
            'Number of inputs ({}) do not coincide with the number of models '\
            '({}) for ParallelModel {}'.format(self)

        y = [self.models[i](inputs[i]) for i in range(len(inputs))]
        value = list()
        for y_i in y:
            value_i = (None if not hasattr(y_i, 'value')
                       else y_i.value)
            value.append(value_i)
        return y, value

    def __call__(self, inputs, name='parallel_model'):
        return self.evaluate(inputs, name=name)


class Concat(tdl.TdlModel):
    @property
    def axis(self):
        return self._axis

    def __init__(self, axis, name='Concat'):
        self._axis = axis
        super(Concat, self).__init__(name=name)

    @tdl.ModelMethod(['value'], ['inputs'])
    def evaluate(self, object, inputs):
        if isinstance(inputs, tdl.core.TdlOp):
            inputs = inputs.value
        return tf.concat(inputs, axis=self.axis)

    def __call__(self, inputs, name=None):
        return self.evaluate(inputs, name=name)


class BoundedOutput(tdl.TdlModel):
    _parameters = ['lower', 'upper']

    @tdl.core.SimpleParameter
    def lower(self, value):
        assert value is not None,\
            'BoundedOutput must at least have a lower bound'
        return value

    @tdl.core.SimpleParameter
    def upper(self, value):
        return value

    class Output(tdl.core.OutputModel):
        @property
        def lower(self):
            return self.model.lower

        @property
        def upper(self):
            return self.model.upper

    @tdl.core.ModelMethod(['value'], ['inputs'], Output)
    def evaluate(self, object, inputs):
        def tanh_bound(x):
            y_delta = (self.upper - self.lower)/2.0
            y_mean = (self.upper + self.lower)/2.0
            return y_delta*tf.nn.tanh((x-y_mean)/y_delta) + y_mean

        def softplus_bound(x):
            return tf.nn.softplus(x - self.lower) + self.lower

        if self.upper is None:
            value = softplus_bound(inputs)
        else:
            value = tanh_bound(inputs)

        return value

    def __call__(self, inputs, name=None):
        return self.evaluate(inputs, name=name)

    def __init__(self, lower=1e-7, upper=None, name='BoundedOutput'):
        super(BoundedOutput, self).__init__(
            lower=lower, upper=upper, name=name)


class MlpNet(StackedModel):
    '''
    full_layers: list of fully connected layers
    out_layer: output layer, for the moment, linear layer
    '''
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
    def weights(self):
        weights = [layer.weights for layer in self.layers
                   if hasattr(layer, 'weights')]
        return weights

    def add(self, layer):
        raise NotImplementedError('Cannot add layers to mlp net. Use a '
                                  'StackedModel and add an mlp to it')

    @tdl.Submodel
    def layers(self, value):
        n_hidden = value['n_hidden']
        afunction = (value['afunction'] if isinstance(value['afunction'], list)
                     else [value['afunction'] for i in range(len(n_hidden))])
        layers = list()
        _n_inputs = self.n_inputs
        for idx, n_units in enumerate(n_hidden):
            layer_i = DenseLayer(n_inputs=_n_inputs,
                                 n_units=n_units,
                                 afunction=afunction[idx])
            _n_inputs = n_units
            layers.append(layer_i)

        if value['output_function'] is None:
            output_layer = AffineLayer(n_inputs=_n_inputs,
                                       n_units=self.n_outputs)
        else:
            output_layer = DenseLayer(n_inputs=_n_inputs,
                                      n_units=self.n_outputs,
                                      afunction=value['output_function'])
        layers.append(output_layer)
        return layers

    def __init__(self, n_inputs, n_outputs, n_hidden, afunction=tf.nn.relu,
                 output_function=None, name='MlpNet'):
        '''All variables corresponding to the weights of the network are defined
        @param n_inputs: number of inputs
        @param n_outputs: number of outputs
        @param n_hidden: list with the number of hidden units in each layer
        @param afunction: function, or list of functions specifying the
                          activation function being used. if not specified,
                          the default is relu
        '''
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        super(MlpNet, self).__init__(
            layers={'n_hidden': n_hidden, 'afunction': afunction,
                    'output_function': output_function},
            name=name)

    class Output(tdl.core.TdlModel):
        @property
        def n_inputs(self):
            return self.model.n_inputs

        @property
        def n_outputs(self):
            return self.model.n_outputs

        @tdl.core.InputArgument
        def inputs(self, value):
            if value is None:
                value = tf.placeholder(tdl.core.global_options.float.tftype,
                                       shape=(None, self.n_inputs),
                                       name='inputs')
            return value

        @tdl.core.InputArgument
        def keep_prob(self, value):
            ''' probability of not droping an activation during dropout '''
            return value

        @tdl.core.Submodel
        def hidden(self, _):
            keep_prob = self.keep_prob
            if (not isinstance(keep_prob, list)):
                # by default, dropout is not applied in the input layer
                keep_prob = [None] + \
                    [keep_prob for i in range(len(self.model.layers) - 1)]
            hidden = list()
            x = self.inputs
            for idx, layer in enumerate(self.model.layers[:-1]):
                x = (x if keep_prob[idx] is None
                     else tf.nn.dropout(x, keep_prob[idx]))
                x = layer(x)
                hidden.append(x)
            return hidden

        @tdl.core.Submodel
        def output(self, _):
            ''' output from the network, after output_function '''
            x = (self.hidden[-1] if self.hidden
                 else self.inputs)
            keep_prob = (self.keep_prob[-1] if isinstance(self.keep_prob, list)
                         else self.keep_prob)
            x = (x if keep_prob is None
                 else tf.nn.dropout(x, keep_prob))
            output = self.model.layers[-1](x)
            return output

        @tdl.core.OutputValue
        def value(self, _):
            try:
                value = tf.convert_to_tensor(self.output)
            except ValueError:
                value = None
            return value

        @property
        def shape(self):
            return (self.value.shape if self.value is not None
                    else None)

        @property
        def weights(self):
            return self.model.weights

        @tdl.core.OptionalProperty
        def loss(self, alpha=1e-5):
            empirical = L2Loss(self.value)
            regularizer = (self.model.regularizer.value
                           if self.model.regularizer.is_set
                           else self.model.regularizer.init())
            loss = EmpiricalWithRegularization(
                empirical=empirical,
                regularizer=regularizer,
                alpha=alpha)
            return loss

        def __init__(self, model, inputs=None, keep_prob=None, name=None):
            ''' Defines the computation graph of the neural network
            for a specific batch size

            @param keep_prob: placeholder used for specify the probability for
                              dropout. If this coefficient is set, then dropout
                              regularization is added between all fully connected
                              layers. By default, dropout is not applied to the
                              inputs.
                              If a list is provided, dropout is applied according
                              to the list.
            '''
            self.model = model
            super(MlpNet.Output, self)\
                .__init__(inputs=inputs, keep_prob=keep_prob,
                          options=options, name=name)

    def evaluate(self, inputs=None, keep_prob=None, name=None):
        return MlpNet.Output(self, inputs=inputs,
                             keep_prob=keep_prob, name=name)

    def __call__(self, x, keep_prob=None, name=None):
        return self.evaluate(x, keep_prob=keep_prob, name=name)


class MlpClassifier(MlpNet):
    @property
    def n_classes(self):
        return self._n_classes

    def __init__(self, n_inputs, n_classes, n_hidden, afunction=tf.nn.relu,
                 name=None):
        self._n_classes = n_classes
        n_outputs = (1 if n_classes == 2
                     else n_classes)
        if n_outputs == 1:
            output_function = tf.nn.sigmoid
        else:
            output_function = tf.nn.softmax
        super(MlpClassifier, self).__init__(n_inputs=n_inputs,
                                            n_outputs=n_outputs,
                                            n_hidden=n_hidden,
                                            afunction=afunction,
                                            output_function=output_function,
                                            name=name)

    class Output(MlpNet.Output):
        @property
        def logits(self):
            return self.output.affine

        @tdl.core.OptionalProperty
        def loss(self, alpha=1e-5):
            empirical = ClassificationLoss(self.logits)
            regularizer = (self.model.regularizer.value
                           if self.model.regularizer.is_set
                           else self.model.regularizer.init())
            loss = EmpiricalWithRegularization(
                empirical=empirical,
                regularizer=regularizer,
                alpha=alpha)
            return loss

    def evaluate(self, inputs=None, keep_prob=None, name=None):
        return MlpClassifier.Output(
            self, inputs=inputs, keep_prob=keep_prob, name=name)

    def __call__(self, x, keep_prob=None, name=None):
        return self.evaluate(x, keep_prob=keep_prob, name=name)
