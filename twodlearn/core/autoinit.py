import numpy as np
import tensorflow as tf
from . import variable
from .options import global_options


class AutoinitType(object):
    ''' Base class to identify auto initializers '''
    pass


class AutoInit(object):
    ''' Indicates that the property should be auto initialized

    Example: ::

        TdlModel(prop=AutoInit())  # Runs auto initialization for prop

    If the property initializer accepts AutoType, the Type can be provided
    using a tuple: ::

        TdlModel(prop=(AutoInit(), AutoType))
    '''


class AutoTensor(AutoinitType):
    ''' auto initialize properties as tensorflow Tensors
    '''
    def __call__(self, value):
        if isinstance(value, tf.Tensor):
            return value
        else:
            return tf.convert_to_tensor(value)


class AutoConstant(AutoinitType):
    ''' auto initialize properties as tensorflow constants
    '''
    def __call__(self, *args, **kargs):
        return tf.constant(*args, **kargs)


class AutoVariable(AutoinitType):
    ''' auto initialize properties as variables

    If an initializer is provided, then shape must be specified: ::

        init = AutoVariable(initializer=tf.keras.initializer.glorot_uniform())
        var = init(shape=shape)

    Otherwise, calling AutoVariable expects an initial value or an initializer

    '''
    def __init__(self, initializer=None):
        self.initializer = initializer

    def __call__(self, *args, **kargs):
        if self.initializer is not None:
            if 'shape' not in kargs:
                raise TypeError('shape must be specified for an AutoVariable '
                                'that has an initializer.')
            if args:
                raise TypeError('arguments must be explicitly stated when '
                                'AutoVariable with an initializer.')
            shape = kargs['shape']
            kargs = {key: value for key, value in kargs.items()
                     if key != 'shape'}
            return variable.variable(self.initializer(shape=shape),
                                     **kargs)
        else:
            return variable.variable(*args, **kargs)


class AutoConstantVariable(AutoinitType):
    ''' auto initialize properties as non-trainable vairables
    '''
    def __call__(self, *args, **kargs):
        return variable.variable(*args, trainable=False, **kargs)


class AutoTrainable(AutoinitType):
    ''' auto initialize properties as trainable vairables
    '''
    def __call__(self, *args, **kargs):
        return variable.variable(*args, trainable=True, **kargs)


class AutoPlaceholder(AutoinitType):
    def __call__(self, **kargs):
        ''' auto initialize properties as placeholders
        '''
        if 'dtype' not in kargs:
            kargs['dtype'] = global_options.float.tftype
        return tf.placeholder(**kargs)


class AutoZeros(AutoinitType):
    def __call__(self, **kargs):
        ''' auto initialize properties as placeholders
        '''
        if 'dtype' not in kargs:
            kargs['dtype'] = global_options.float.tftype
        return tf.zeros(**kargs)


class AutoNormalVar(AutoinitType):
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape, **kargs):
        ''' auto initialize properties as variables
        '''
        return variable.variable(
            tf.random_normal(shape=shape, mean=self.mean, stddev=self.stddev),
            **kargs)
