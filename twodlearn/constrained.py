import numpy as np
import tensorflow as tf
import twodlearn as tdl
import twodlearn.core


class TransformedVariable(tdl.core.TdlModel):
    ''' Transformed Variable
    1. Initialization is performed using the inverse method
        raw = tf.Variable(inverse(initial_value))
    2. the value is defined using the transform
        value = transform(raw)
    '''
    @classmethod
    def init_wrapper(cls, initializer, trainable, **kargs):
        '''wraps the object to be used as an initializer.'''
        return lambda shape: cls(initial_value=initializer(shape=shape),
                                 trainable=trainable,
                                 **kargs)

    @tdl.core.SimpleParameter
    def raw(self, kargs):
        ''' raw value before making a transformation '''
        initial_value = kargs['initial_value']
        if callable(initial_value):
            kargs['initial_value'] = lambda: self.inverse(initial_value())
        else:
            kargs['initial_value'] = self.inverse(initial_value)
        variable = tdl.core.variable(**kargs)
        return variable

    @property
    def initializer(self):
        return self.raw.initializer

    @tdl.core.OutputValue
    def value(self, _):
        return self.transform(self.raw)

    @property
    def shape(self):
        return self.value.shape

    def inverse(self, value):
        raise NotImplementedError('inverse not specified')

    def transform(self, value):
        raise NotImplementedError('transform not specified')

    def __pow__(self, other):
        return self.value ** other

    def __add__(self, other):
        return self.value + other

    def __mul__(self, other):
        return self.value * other

    def __init__(self, initial_value, trainable=True,
                 collections=None, validate_shape=True,
                 caching_device=None, name=None,
                 variable_def=None, dtype=None,
                 expected_shape=None, import_scope=None,
                 constraint=None,
                 **kargs):

        variable_args = {'initial_value': initial_value,
                         'trainable': trainable,
                         'collections': collections,
                         'validate_shape': validate_shape,
                         'caching_device': caching_device,
                         'name': name,
                         'variable_def': variable_def,
                         'dtype': dtype,
                         'expected_shape': expected_shape,
                         'import_scope': import_scope,
                         'constraint': constraint}
        super(TransformedVariable, self).__init__(
            raw=variable_args,
            name=name,
            **kargs)


class PositiveVariable(TransformedVariable):
    ''' Creates a variable that can only take positive values '''
    @tdl.core.InputArgument
    def tolerance(self, value):
        '''tolerance for positive variables.'''
        return value

    def inverse(self, value):
        if isinstance(value, tf.Tensor):
            return tf.log(tf.exp(tf.abs(value)) - 1)
        else:
            return np.log(np.exp(np.abs(value)) - 1).astype(np.float32)
        return value

    def transform(self, value):
        output = tf.nn.softplus(value)
        if self.tolerance is not None:
            output = output + self.tolerance
        return output


class PositiveVariableExp(TransformedVariable):
    ''' Creates a variable that can only take positive values.
    This function uses exp() as a reparameterization of the variable'''
    @tdl.core.InputArgument
    def tolerance(self, value):
        '''tolerance for positive variables.'''
        if value is None:
            value = tdl.core.global_options.tolerance
        return value

    @tdl.core.InputArgument
    def max(self, value):
        return value

    @tdl.core.ParameterInit
    def raw(self, initial_value, trainable=True, **kargs):
        ''' raw value before making a transformation '''
        tdl.core.assert_initialized(self, 'raw', ['max', 'tolerance'])
        if callable(initial_value):
            initial_value_fn = initial_value
            initial_value = lambda: self.inverse(initial_value_fn())
        else:
            initial_value = self.inverse(initial_value)
        if self.max is None:
            variable = tdl.core.variable(
                initial_value, trainable=trainable,
                constraint=lambda x: tf.clip_by_value(
                    x, clip_value_min=self.inverse(self.tolerance),
                    clip_value_max=np.infty),
                **kargs)
        else:
            variable = tdl.core.variable(
                initial_value,
                trainable=trainable,
                constraint=lambda x: tf.clip_by_value(
                    x, clip_value_min=self.inverse(self.tolerance),
                    clip_value_max=self.inverse(self.max)),
                **kargs)
        return variable

    def inverse(self, value):
        if isinstance(value, tf.Tensor):
            return tf.log(value)
        else:
            return np.log(value).astype(np.float32)

    def transform(self, value):
        output = tf.exp(value)
        return output

    def __init__(self, initial_value, max=None, trainable=True,
                 collections=None, validate_shape=True,
                 caching_device=None, name=None,
                 tolerance=None):
        super(TransformedVariable, self).__init__(
            name=name, max=max,
            tolerance=tolerance,
            raw={'initial_value': initial_value,
                 'trainable': trainable,
                 'collections': collections,
                 'validate_shape': validate_shape,
                 'caching_device': caching_device})


class PositiveVariable2(TransformedVariable):
    ''' Creates a variable that can only take positive values.
    This function uses pow(x,2) as a reparameterization of the variable'''
    @tdl.core.InputArgument
    def tolerance(self, value):
        '''tolerance for positive variables.'''
        return value

    def inverse(self, value):
        if isinstance(value, tf.Tensor):
            return tf.sqrt(value)
        else:
            return np.sqrt(value).astype(np.float32)

    def transform(self, value):
        output = tf.pow(value, 2.0)
        if self.tolerance is not None:
            output = output + self.tolerance
        return output


class BoundedVariable(TransformedVariable):
    ''' Creates a variable that can only take values inside a range '''
    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    def inverse(self, value):
        return value

    def transform(self, value):
        y_delta = (self.max - self.min)/2.0
        y_mean = (self.max + self.min)/2.0
        return y_delta*tf.nn.tanh((value-y_mean)/y_delta) + y_mean

    def __init__(self, min, max, initializer,
                 initial_value=None, trainable=True,
                 collections=None, validate_shape=True,
                 caching_device=None, name='variable',
                 variable_def=None, dtype=None,
                 expected_shape=None, import_scope=None,
                 constraint=None, options=None):
        assert np.all(max > min), 'max should be > than min'
        self._min = min
        self._max = max
        super(BoundedVariable, self)\
            .__init__(initializer=initializer,
                      initial_value=initial_value, trainable=trainable,
                      collections=collections, validate_shape=validate_shape,
                      caching_device=caching_device, name=name,
                      variable_def=variable_def, dtype=dtype,
                      expected_shape=expected_shape, import_scope=import_scope,
                      constraint=constraint, options=options)


class ConstrainedVariable(tdl.core.TdlModel):
    @property
    def value(self):
        return self.variable.value()

    @property
    def shape(self):
        return self.value.shape

    @property
    def initializer(self):
        return self.variable.initializer

    @tdl.core.InputArgument
    def min(self, value):
        if isinstance(value, tf.Tensor):
            return value
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        return value.astype(tdl.core.global_options.float.nptype)

    @tdl.core.InputArgument
    def max(self, value):
        if isinstance(value, tf.Tensor):
            return value
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        return value.astype(tdl.core.global_options.float.nptype)

    @tdl.core.InputArgument
    def initial_value(self, value):
        return value

    def projection(self, x):
        return tf.clip_by_value(x, clip_value_min=self.min,
                                clip_value_max=self.max)

    @tdl.core.SimpleParameter
    def variable(self, value):
        if value is None:
            value = tf.Variable(self.initial_value,
                                constraint=self.projection)
        else:
            raise NotImplementedError('Providing variable directly is not yet '
                                      'implemented')
        return value

    def __init__(self, initial_value, min=0.0, max=np.infty, name=None):
        super(ConstrainedVariable, self).__init__(
            initial_value=initial_value,
            min=min, max=max, name=name)
