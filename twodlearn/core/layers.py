'''Definition of tf.keras compatible Layers

The tdl Layer is compatible with tf.keras layers and supports
autoinitialization with tdl descriptors.
'''
import types
import functools
import collections
import tensorflow as tf
from . import common
from .common import (InputArgument, InputParameter, InputModel, InferenceInput,
                     InputModelInit,
                     Submodel, SubmodelWithArgs, SubmodelInit, ParameterInit,
                     OutputValue,
                     SimpleParameter,
                     Regularizer, OptionalProperty,
                     is_property_initialized,
                     nest
                     )
from .search import get_variables, is_trainable, SEARCHABLE_CLASSES


def _assert_known_kargs(layer, kwargs):
    '''Checks if the provided arguments have an initialization method.'''
    known = common._find_tdl_attrs(
        type(layer), AttrClass=(),
        AttrBaseClass=common.TDL_INIT_DESCRIPTORS)
    for kwarg in kwargs:
        if kwarg not in known:
            raise TypeError('Unknown keyword argument:', kwarg)


@common.add_autoinit_class
class Layer(tf.keras.layers.Layer):
    '''TDL Layer that is compatible with tf.keras specification.'''

    def _tdl_check_kwargs(self, kwargs):
        '''custom defined function to check arguments before initialization'''
        return

    @InputArgument
    def input_shape(self, value):
        if value is None:
            # raise ValueError('the value of attribute {}.input_shape must be '
            #                  'provided'.format(type(self).__name__))
            return None
        if isinstance(value, (tuple, list)):
            if all(isinstance(i, (int, None)) for i in value):
                value = tf.TensorShape(value)
        flat = nest.flatten(value)
        if not all(isinstance(i, tf.TensorShape) for i in flat):
            raise ValueError('input_shape should be a TensorShape or a nested'
                             'structure of TensorShapes.')
        return value

    @property
    def scope(self):
        '''Scope for the model, used to define all operations.'''
        if not hasattr(self.__tdl__, '_scope'):
            with tf.name_scope(self.name) as scope:
                self.__tdl__._scope = scope
        return self.__tdl__._scope

    def _get_unique_name(self, name):
        def is_absolute(name):
            return name[0] == '/' or name[-1] == '/'
        name = (name if name is not None
                else type(self).__name__)
        if is_absolute(name):
            self.__tdl__._scope = name
        else:
            with tf.name_scope(name) as scope:
                self.__tdl__._scope = scope
        return self.scope[:-1]

    def __init__(self, trainable=True, name=None, *args,  **kwargs):
        '''Base initialization of a Tdl operation.
        Arguments should be explicitly specified. The basic acguments are:
            name: name of the model/operation, used to create a scope,
                  if no name is provided, the function will look for
                  self._default_name
            parameters corresponding to the specific model
            submodels corresponding to the specific model.
        '''
        if args:
            raise TypeError('arguments for tdl layers must be explicitly '
                            'stated')
        # initialize TDL instance for the descriptors
        if not hasattr(self, '__tdl__'):
            setattr(self, '__tdl__', common.__TDL__(self))
        # initialize name and scope
        name = self._get_unique_name(name)
        super(Layer, self).__init__(trainable=trainable, name=name)

        # auto initialize using the tdl descriptors
        self._tdl_check_kwargs(kwargs)
        _assert_known_kargs(self, kwargs)
        assert common.get_context(self).given_attrs is None
        common.get_context(self).given_attrs = \
            set([key for key, value in kwargs.items()])

        with tf.name_scope(self.scope), common.DisableAutoinit(self):
            attrs_done = common._init_tdl_attrs(
                self, kwargs, '_input_args',
                (InputArgument, InputParameter, InputModel, InferenceInput,
                 InputModelInit))
            allowed_autoinit = attrs_done
            attrs_done = common._init_tdl_attrs(
                self, kwargs, '_parameters',
                (SimpleParameter, ParameterInit),
                allowed_autoinit=allowed_autoinit)
            allowed_autoinit = set.union(attrs_done, allowed_autoinit)
            attrs_done = common._init_tdl_attrs(
                self, kwargs, '_submodels',
                (Submodel, SubmodelWithArgs, SubmodelInit),
                allowed_autoinit=allowed_autoinit)
            allowed_autoinit = set.union(attrs_done, allowed_autoinit)
            attrs_done = common._init_tdl_attrs(
                self, kwargs, '_model_outputs', OutputValue,
                allowed_autoinit=allowed_autoinit)
            allowed_autoinit = set.union(attrs_done, allowed_autoinit)
            attrs_done = common._init_tdl_attrs(
                self, kwargs, '_optional', (Regularizer, OptionalProperty),
                allowed_autoinit=allowed_autoinit)
        common.assert_initialized(self, '__init__',
                                  self.__tdl__._model_outputs)
        common.get_context(self).initialized = True

        # Wrap call method to update variables when finished
        def update_vars_wrapper(call_fn):
            @functools.wraps(call_fn)
            def call(obj, *args, **kargs):
                output = call_fn(obj, *args, **kargs)
                obj._update_variables()
                return output
            return call
        self.call = types.MethodType(
            update_vars_wrapper(self.call.__func__), self)
        self.build = types.MethodType(
            update_vars_wrapper(self.build.__func__), self)

    def build(self, input_shape=None):
        if input_shape is not None:
            if not is_property_initialized(self, 'input_shape'):
                self.input_shape = input_shape
        common.build(self)
        self.built = True
        return self

    def add_weight(self, *args, **kwargs):
        '''add a weight to the layer. If only one argument is provided, it
        should be an instance of tf.Variable. Otherwise, the signature is
        the same as tf.keras.Layer add_weight.'''
        if len(args) == 1 and not kwargs:
            variable = args[0]
            if self.trainable and isinstance(variable, tf.Variable):
                self._trainable_weights.append(variable)
            elif isinstance(variable, tf.Variable):
                self._non_trainable_weights.append(variable)
            else:
                raise ValueError('weight must be a variable instance')
            return variable
        else:
            return super(Layer, self).add_weight(*args, **kwargs)

    def _update_variables(self):
        '''Update the variables and weights list by searching through the
        attributes with tdl descriptors
        '''
        attr_names = collections.deque(
            common._find_tdl_attrs(
                type(self), AttrClass=common.TDL_INIT_DESCRIPTORS))
        vars = set()
        for name in attr_names:
            if common.is_property_initialized(self, name):
                attr = getattr(self, name)
                if isinstance(attr, (list, tuple)):
                    for attr_i in attr:
                        if isinstance(attr_i, SEARCHABLE_CLASSES):
                            vars |= set(get_variables(attr_i))
                elif isinstance(attr, SEARCHABLE_CLASSES):
                    vars |= set(get_variables(attr))
        vars = vars - set.union(set(self.trainable_weights),
                                set(self.non_trainable_weights))
        self._trainable_weights.extend(
            [v for v in vars if is_trainable(v)])
        self._non_trainable_weights.extend(
            [v for v in vars if not is_trainable(v)])
