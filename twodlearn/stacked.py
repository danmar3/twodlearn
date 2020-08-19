from __future__ import division
from __future__ import print_function

import warnings
import functools
import collections
import numpy as np
import tensorflow as tf
import twodlearn as tdl


@tdl.core.create_init_docstring
class StackedLayers(tdl.core.Layer):
    '''Layer that is composed of several stacked layers.'''

    @tdl.core.InputArgument
    def input_shape(self, value):
        '''Input tensor shape.'''
        if value is None:
            raise tdl.core.exceptions.ArgumentNotProvided(self)
        if isinstance(value, tf.TensorShape):
            return value
        elif isinstance(value, (list, tuple)):
            if all(isinstance(vi, int) for vi in value):
                return tf.TensorShape(value)
            elif all(isinstance(vi, tf.TensorShape) for vi in value):
                return value
        elif isinstance(value, dict):
            if all(isinstance(vi, tf.TensorShape) for vi in value.values()):
                return value
        if not isinstance(value, tf.TensorShape):
            value = tf.TensorShape(value)
        return value

    @tdl.Submodel
    def layers(self, value):
        '''Layers that are executed sequentially.'''
        if value is None:
            value = list()
        return value

    def add(self, layer):
        '''Add a new layer to the stacked model.
        The only condition is that layer must be callable.
        '''
        assert callable(layer), \
            'Model {} is not callable. StackedLayers only works with '\
            'callable models'.format(layer)
        assert self.built is False, 'StackedLayers has been already built'
        self.layers.append(layer)
        return layer

    def compute_output_shape(self, input_shape=None):
        tdl.core.assert_initialized(self, 'copute_output_shape', ['layers'])
        if input_shape is None:
            tdl.core.assert_initialized(
                self, 'copute_output_shape', ['input_shape'])
            input_shape = self.input_shape
        input_shape = tf.TensorShape(input_shape)
        for layer in self.layers:
            if isinstance(layer, tf.keras.layers.Layer):
                output_shape = layer.compute_output_shape(
                    input_shape=input_shape)
                input_shape = output_shape
        return output_shape

    def build(self, input_shape=None):
        '''Build the model. Note that this function does not build the
        layers.'''
        if not tdl.core.is_property_initialized(self, 'input_shape'):
            self.input_shape = input_shape
        tdl.core.assert_initialized(self, 'build', ['layers'])
        self.built = True

    def _update_shapes(self, inputs):
        if not tdl.core.is_property_initialized(self, 'input_shape'):
            if not isinstance(inputs, (list, tuple)):
                try:
                    inputs = tf.convert_to_tensor(inputs)
                    self.input_shape = inputs.shape
                except TypeError:
                    pass

    def call(self, inputs, *args, **kargs):
        '''Call the layers in sequential order.'''
        # assert self.input_shape == inputs.shape TODO
        tdl.core.assert_initialized(self, 'call', ['layers'])
        self._update_shapes(inputs=inputs)
        output_type = kargs['output'] if 'output' in kargs else 'output'

        hidden = list()
        output = inputs
        for layer in self.layers:
            output = layer(output)
            hidden.append(output)

        if output_type == 'output':
            return output
        elif output_type == 'hidden':
            return hidden
        else:
            raise ValueError(f'invalid output option "{output_type}"')


class StackedModel(tdl.core.TdlModel):
    @tdl.Submodel
    def layers(self, value):
        if value is None:
            value = list()
        return value

    @tdl.core.InputArgument
    def return_layers(self, value):
        '''True if the return value of the stacked model is the layers'''
        if value is None:
            value = False
        if value not in (True, False):
            raise ValueError('return_layers must be either True or False')
        return value

    def __getitem__(self, item):
        return self.layers[item]

    def __len__(self):
        return len(self.layers)

    def add(self, layer):
        assert callable(layer), \
            'Model {} is not callable. StackedModel only works with '\
            'callable models'.format(layer)
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
        pass

    @tdl.ModelMethod(['output', 'hidden', 'value', 'shape'], ['inputs'],
                     StackedOutput)
    def evaluate(self, object, inputs):
        hidden = list()
        x = inputs
        for layer in self.layers:
            x = layer(x)
            hidden.append(x)
        y = hidden[-1]
        hidden = hidden[:-1]
        try:
            value = tf.convert_to_tensor(y)
        except (ValueError, TypeError):
            value = None
        shape = (value.shape if value is not None
                 else None)
        return y, hidden, value, shape

    def __call__(self, x, name=None):
        layers = self.evaluate(inputs=x, name=name)
        if self.return_layers:
            return layers
        else:
            return layers.output

    def __init__(self, layers=None, return_layers=None,
                 options=None, name='Stacked'):
        super(StackedModel, self).__init__(
            layers=layers, return_layers=return_layers,
            options=options, name=name)
