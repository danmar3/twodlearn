from __future__ import division
from __future__ import print_function

import warnings
import functools
import collections
import numpy as np
import tensorflow as tf
import twodlearn as tdl
from .core import nest


@tdl.core.create_init_docstring
class ParallelLayers(tdl.core.Layer):
    '''Layer that is composed of several parallel layers.'''

    @tdl.core.InputArgument
    def input_shape(self, value):
        '''Input tensor shape.'''
        if value is None:
            raise tdl.core.exceptions.ArgumentNotProvided(self)
        if isinstance(value, (list, tuple)):
            if all(isinstance(vi, int) for vi in value):
                return tf.TensorShape(value)
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
        flatten_shape = nest.flatten(input_shape)
        flatten_layers = nest.flatten(self.layers)
        assert len(flatten_shape) == len(flatten_layers),\
            'the number of input_shape elements must be the same than '\
            'the number of layers'
        flatten_output = list()
        for (layer, shape) in zip(flatten_layers, flatten_shape):
            if isinstance(layer, tf.keras.layers.Layer):
                output_shape = layer.compute_output_shape(
                    input_shape=shape)
                flatten_output.append(output_shape)
        return nest.pack_sequence_as(
            structure=self.layers,
            flat_sequence=flatten_output)

    def build(self, input_shape=None):
        '''Build the model. Note that this function does not build the
        layers.'''
        if not tdl.core.is_property_initialized(self, 'input_shape'):
            self.input_shape = input_shape
        tdl.core.assert_initialized(self, 'build', ['layers'])
        self.built = True

    def call(self, inputs, *args, **kargs):
        '''Call the layers in sequential order.'''
        # assert self.input_shape == inputs.shape TODO
        tdl.core.assert_initialized(self, 'call', ['layers'])
        flatten_inputs = nest.flatten(inputs)
        flatten_layers = nest.flatten(self.layers)
        flatten_output = list()
        for layer, input in zip(flatten_layers, flatten_inputs):
            flatten_output.append(layer(input))
        return nest.pack_sequence_as(
            structure=self.layers,
            flat_sequence=flatten_output)
