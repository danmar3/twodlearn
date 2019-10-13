"""This module defines common structures used in twodlearn
"""
import sys
try:
    set
except NameError:
    from sets import Set as set
import types
import typing
import inspect
import warnings
import functools
import numpy as np
import collections
import tensorflow as tf
from .options import global_options
from . import exceptions
from . import autoinit
from . import variable
from .context import (add_autoinit_class, __TDL__,
                      get_context, is_autoinit_enabled,
                      DisableAutoinit)

try:
    from types import SimpleNamespace as PySimpleNamespace
except ImportError:
    from argparse import Namespace as PySimpleNamespace


PYTHON_VERSION = sys.version_info[0]
TDL_DESCRIPTORS = set()        # list of all available descriptors
TDL_INIT_DESCRIPTORS = set()

py_hasattr = hasattr


def add_init_descriptor(descriptor):
    '''Adds the descriptor to the list of initialization descriptors
    This list is checked by the class initializers to verify if the provided
    arguments have an initialization method. '''
    assert hasattr(descriptor, '__get__'), \
        'provided class does not implement a __get__ method'
    TDL_INIT_DESCRIPTORS.add(descriptor)
    return descriptor


def add_tdl_descriptor(descriptor):
    '''Adds the descriptor to the list of existing descriptors
    This list is checked by the class initializers to verify if the provided
    arguments have an initialization method '''
    assert hasattr(descriptor, '__get__'), \
        'provided class does not implement a __get__ method'
    TDL_DESCRIPTORS.add(descriptor)
    return descriptor


def _get_func_args(func):
    if PYTHON_VERSION >= 3:
        return inspect.getfullargspec(func).args
    else:
        return inspect.getargspec(func).args


def merge_dicts(a, b):
    # if PYTHON_VERSION >= 3:
    #    print('using python: {}'.format(PYTHON_VERSION))
    #    return {**train_stats, **valid_stats}
    # else:
    z = a.copy()
    z.update(b)
    return z


class SimpleNamespace(PySimpleNamespace):
    ''' SimpleNamespace that works with tf.convert_to_tensor and is searchable
    '''


class Options(dict):
    def __setitem__(self, key, item):
        super(Options, self).__setitem__(key, item)

    def __getitem__(self, key):
        self.n_access[key] = (self.n_access[key] + 1 if key in self.n_access
                              else 1)
        return super(Options, self).__getitem__(key)

    def __init__(self, *argv, **kargs):
        self.n_access = dict()
        super(Options, self).__init__(*argv, **kargs)


class reuse_scope:
    def __enter__(self):
        global_options.reuse_scope = True

    def __exit__(self, type, value, traceback):
        global_options.reuse_scope = False


def check_defaults(options, default):
    """Adds the values in default to options.

    Args:
        options (dict): base options.
        default (dict): default options.
    Returns:
        dict: options with the missing values that are in default but not in
        options.
    """
    if options is None:
        options = dict()
    if default is not None:
        for key, value in default.items():
            if key not in options:
                options[key] = value
    return options


class TdlOp(object):
    ''' Base class for defining operations
    The operation is encapsulated inside a scope '''
    @property
    def name(self):
        ''' name for the model '''
        return self._name

    @name.setter
    def name(self, value):
        assert not hasattr(self, '_name'),\
            'name can only be set once'

        def is_absolute(name):
            return name[0] == '/' or name[-1] == '/'

        if is_absolute(value):
            with tf.name_scope(value) as scope:
                self._scope = scope
            self._name = self.scope.split('/')[-2]
        else:
            self._name = value

    @property
    def scope(self):
        ''' scope for the model, used to define all operations '''
        if not hasattr(self, '_scope'):
            assert hasattr(self, '_name'), \
                'attempting to create scope with undefined name'
            with tf.name_scope(self.name) as scope:
                self._scope = scope
        return self._scope

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, opt):
        def assert_dict_equal(opt1, opt2):
            for key, value in opt2.items():
                if isinstance(value, dict):
                    assert isinstance(opt1[key], dict),\
                        'New options do not match old ones'
                    assert_dict_equal(opt1[key], value)
                else:
                    assert opt1[key] == value,\
                        'New options do not match old ones'

        if hasattr(self, '_options'):
            assert_dict_equal(opt, self._options)
        self._options = opt

    def _init_options(self, options, default=None):
        if options is None:
            options = dict()
        if default is not None:
            for key, value in default.items():
                if key not in options:
                    options[key] = value
        return options

    def __init__(self, name, options=None):
        self.name = name
        self.options = self._init_options(options)

    def __pow__(self, other):
        return self.value**other

    def __add__(self, other):
        return self.value+other

    def __sub__(self, other):
        return self.value-other

    def __mul__(self, other):
        return self.value*other

    __radd__ = __add__
    __rmul__ = __mul__


# def encapsulated_op(func):
#     def encapsulate(*args, **kargs):
#         input_vars = locals()
#         print(input_vars)
#         if 'name' in kargs:
#             name = kargs['name']
#         else:
#             name = func.func_name
#         output = TdlOp(name=name, options=None)
#         y = func(*args)
#         setattr(output, 'y', y)
#         return output
#     return encapsulate
class Parameter(object):
    def __init__(self, fget=None, finit=None):
        self.fget = fget
        self.finit = finit
        self.name = fget.__name__

    def init(self, finit):
        assert self.finit is None,\
            'the initialization has already been specified'
        return type(self)(self.fget, finit)

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable parameter")
        return self.fget(obj)

    def __set__(self, obj, val):
        if self.finit is None:
            raise AttributeError('initializer for parameter {} '
                                 'not specified'.format(self.name))
        self.finit(obj, val)


class UnsetProperty(object):
    is_set = False

    def __init__(self, obj, attr_name, finit):
        self._finit = finit
        self._obj = obj
        self._attr_name = attr_name

    def default_init(self):
        raise NotImplementedError(
            'No default initialization method specified for {}'
            ''.format(type(self)))

    def init(self, *args, **kargs):
        # set the attribute using finit method
        if hasattr(self._obj, 'scope'):
            with tf.name_scope(self._obj.scope):
                with tf.name_scope(self._attr_name):
                    setattr(self._obj, self._attr_name,
                            self._finit(self._obj, *args, **kargs))
        else:
            setattr(self._obj, self._attr_name,
                    self._finit(self._obj, *args, **kargs))


@add_tdl_descriptor
@add_init_descriptor
class TdlDescriptor(object):
    ''' Decorator used to specify a parameter inside a model.
    The decorator works similar to @property, but the specified
    method correponds to the initialization of the parameter.

    Examples:
        The tdl descriptor allows to specify a default initialization function
        for a class attribute ::

            class Layer(core.Layer):
                @tdl.core.TdlDescriptor
                def submodel(self, value):
                    # This is the initialization method for the attribute
                    # submodel. Value is an attribute given as a key-word
                    # parameter when the class is initialized.
                    # If no value is provided as parameter, value is None.
                    # The return value is stored and is returned by the
                    # __get__ function.
                    if value is None:
                        # No parameter provided, give adefault value
                        value = 0.0
                    # The return value is stored and is the value of the
                    # attribute after initialization
                    return value

        The required and optional methods use some default initialization
        functions that are commonly used. They allow to reduce the number of
        lines ::

            class Layer(core.Layer):
                n_dims = core.InputArgument.optional(
                    'n_dims', doc='dimensions of the GMM model')
                n_components = core.InputArgument.required(
                    'n_components', doc='number of mixture components')
                alpha = core.InputArgument.optional(
                    'alpha', doc='is alpha',
                    default=True)

    '''
    @classmethod
    def required(cls, name, doc=None):
        def finit(self, value):
            if value is None:
                raise ValueError('the value of attribute {}.{} must be '
                                 'provided'.format(type(self).__name__, name))
            else:
                return value
        return cls(finit=finit, doc=doc, name=name)

    @classmethod
    def optional(cls, name, doc=None, default=None):
        def finit(self, value):
            if value is None:
                return default
            return value
        return cls(finit=finit, doc=doc, name=name)

    def __init__(self, finit=None, doc=None, name=None):
        """Creates a new SingleParameter.

        Args:
            finit (callable): Function that initialize the parameter.
                The function should return the value for the parameter.
                Defaults to None.
            doc (str): docstring for the attribute.
            name (str): name of the method. Ussualy infered from finit
        """
        self.finit = finit
        if name is None:
            name = finit.__name__
        assert isinstance(name, str), 'attribute name should be a string'
        self.name = name
        if doc is None and finit is not None:
            doc = finit.__doc__
        self.__doc__ = doc

    def __get__(self, obj, objtype):
        '''returns the value of the attribute.
        If the attribute has not been initialized, autoinit is called
        and the attribute is returned
        '''
        if obj is None:
            return self
        # initialize if the property has not been set
        if not hasattr(obj.__tdl__, self.name):
            self.autoinit(obj)
        return getattr(obj.__tdl__, self.name)

    def __set__(self, obj, val):
        '''sets the value of the attribute.
        Calls the initialization function with the given value.
        '''
        if self.finit is None:
            raise AttributeError('initializer for parameter {} '
                                 'not specified'.format(self.name))
        if not hasattr(obj, '__tdl__'):
            setattr(obj, '__tdl__', __TDL__(obj))
        # TODO: remove finit, which must be called only from init method
        # setattr(obj.__tdl__, self.name, self.finit(obj, val))
        self.init(obj, val)

    def init(self, obj, val):
        ''' '''
        def run_init():
            try:
                if isinstance(val, tuple) and len(val) == 2:
                    if isinstance(val[0], autoinit.AutoInit):
                        return self.finit(obj, None, val[1])
                    else:
                        return self.finit(obj, val[0], val[1])
                elif isinstance(val, autoinit.AutoinitType):
                    return self.finit(obj, None, val)
                else:
                    return self.finit(obj, val)
            except exceptions.UnsetProperty as error:
                raise exceptions.InitPreconditionsFailed(
                    obj, self.name, [error.property])

        # if isinstance(val, autoinit.AutoInit):
        #     self.autoinit(obj, force=True)
        #     return

        if self.finit is None:
            raise AttributeError('initializer for parameter {} '
                                 'not specified'.format(self.name))
        if not hasattr(obj, '__tdl__'):
            setattr(obj, '__tdl__', __TDL__(obj))

        # check attribute has not been initialized
        if hasattr(obj.__tdl__, self.name):
            raise exceptions.PropertyRedefinition(
                property=self.name, object=obj)
        # run initialization
        if hasattr(obj, 'scope'):
            with tf.name_scope(obj.scope), tf.name_scope(self.name):
                param = run_init()
        else:
            param = run_init()
        setattr(obj.__tdl__, self.name, param)

    def autoinit(self, obj, force=False):
        '''initialized the attribute with None'''
        if is_autoinit_enabled(obj) or force:
            self.init(obj, None)
        else:
            raise exceptions.UnsetProperty(property=self.name, object=obj)


class OptionalPropertyWrapper(TdlOp):
    @property
    def is_set(self):
        return self._prop_value is not None

    @property
    def value(self):
        return self._prop_value

    def __init__(self, obj, finit, fset, value):
        self._obj = obj
        self._finit = finit
        self._fset = fset
        self._prop_value = value
        super(OptionalPropertyWrapper, self).__init__(name=finit.__name__)

    def __getattr__(self, attr):
        return getattr(self._prop_value, attr)

    def init(self, *args, **kargs):
        if hasattr(self._obj, 'scope'):
            with tf.name_scope(self._obj.scope):
                with tf.name_scope(self._finit.__name__):
                    self._prop_value = self._finit(*args, **kargs)
        else:
            self._prop_value = self._finit(*args, **kargs)
        return self._prop_value

    def set_value(self, value):
        self._prop_value = (self._fset(value) if self._fset is not None
                            else value)
        return self._prop_value

    def call(self, *args, **kargs):
        return self._prop_value(*args, **kargs)


@add_tdl_descriptor
@add_init_descriptor
class OptionalProperty(object):
    ''' Decorator used to specify an optional property inside a model.
    The decorator works similar to @property, but the specified
    method correponds to the initialization of the property '''

    def __init__(self, finit=None, fset=None):
        """
        Args:
            finit (callable): Function that initialize the parameter.
                The function should return the value for the parameter.
                Defaults to None.
        """
        self.finit = finit
        self.fset = fset
        self.name = finit.__name__

    def setter(self, fset):
        assert self.fset is None,\
            'the evaluation method has already been specified'
        return type(self)(finit=self.finit, fset=fset)

    def _check_wrapper_exist(self, obj):
        if not hasattr(obj, '__tdl__'):
            setattr(obj, '__tdl__', __TDL__(obj))
        if not hasattr(obj.__tdl__, self.name):
            wrapper = OptionalPropertyWrapper(
                obj=obj,
                finit=types.MethodType(self.finit, obj),
                fset=(types.MethodType(self.fset, obj) if self.fset
                      else None),
                value=None)
            setattr(obj.__tdl__, self.name, wrapper)

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        self._check_wrapper_exist(obj)
        return getattr(obj.__tdl__, self.name)

    def __set__(self, obj, val):
        if self.finit is None:
            raise AttributeError('initializer for parameter {} '
                                 'not specified'.format(self.name))
        self._check_wrapper_exist(obj)
        getattr(obj.__tdl__, self.name).set_value(val)

    def init(self, obj, val):
        if self.finit is None:
            raise AttributeError('initializer for parameter {} '
                                 'not specified'.format(self.name))
        self._check_wrapper_exist(obj)
        getattr(obj.__tdl__, self.name).set_value(val)


@add_tdl_descriptor
@add_init_descriptor
class LazzyProperty(object):
    def __init__(self, finit=None):
        self.finit = finit
        self.name = finit.__name__

    def init_property(self, obj):
        if hasattr(obj, 'scope'):
            with tf.name_scope(obj.scope):
                with tf.name_scope(self.name):
                    value = self.finit(obj)
        else:
            value = self.finit(obj)
        setattr(obj.__tdl__, self.name, value)

    def autoinit(self, obj, force=True):
        self.init_property(obj)

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        if not hasattr(obj, '__tdl__'):
            setattr(obj, '__tdl__', __TDL__(obj))
        if not hasattr(obj.__tdl__, self.name):
            self.init_property(obj)
        return getattr(obj.__tdl__, self.name)


@add_tdl_descriptor
@add_init_descriptor
class Regularizer(OptionalProperty):
    '''Decorator used to specify a regularizer for a model.
    The decorator works similar to @property, but the specified
    method correponds to the initialization of the regularizer.
    '''


@add_tdl_descriptor
@add_init_descriptor
class SimpleParameter(TdlDescriptor):
    ''' Decorator used to specify a parameter inside a model.
    The decorator works similar to @property, but the specified
    method correponds to the initialization of the parameter '''


@add_tdl_descriptor
@add_init_descriptor
class Submodel(TdlDescriptor):
    '''Decorator used to specify a submodel inside a model.
    The decorator works similar to @property, but the specified
    method correponds to the initialization of the submodel.
    '''


@add_tdl_descriptor
@add_init_descriptor
class OutputValue(TdlDescriptor):
    '''Decorator used to specify the output value of a model.
    The decorator works similar to @property, but the specified
    method correponds to the definition of the output value.
    These methods are called at the end of the auto initialization
    procedure.
    '''


@add_tdl_descriptor
@add_init_descriptor
class SubmodelWithArgs(Submodel):
    '''Decorator used to specify a submodel inside a model.
    The decorator works similar to @property, but the specified
    method correponds to the initialization of the submodel.
    '''
    def __set__(self, obj, val):
        if self.finit is None:
            raise AttributeError('initializer for parameter {} '
                                 'not specified'.format(self.name))
        if not hasattr(obj, '__tdl__'):
            setattr(obj, '__tdl__', __TDL__(obj))
        setattr(obj.__tdl__, self.name, self.finit(obj, **val))

    def init(self, obj, val):
        if self.finit is None:
            raise AttributeError('initializer for parameter {} '
                                 'not specified'.format(self.name))
        if not hasattr(obj, '__tdl__'):
            setattr(obj, '__tdl__', __TDL__(obj))
        if hasattr(obj, 'scope'):
            with tf.name_scope(obj.scope):
                with tf.name_scope(self.name):
                    param = self.finit(obj, **val)
        else:
            param = self.finit(obj, **val)
        setattr(obj.__tdl__, self.name, param)


@add_tdl_descriptor
@add_init_descriptor
class InputArgument(TdlDescriptor):
    ''' Decorator used to specify the input arguments for a model.
    The decorator works similar to @property, but the specified
    method correponds to the initialization of the argument, ussually
    checking for types and setting default values.
    '''


@add_tdl_descriptor
@add_init_descriptor
class InputParameter(InputArgument):
    ''' Decorator used to specify the input arguments for a model.
    These inputs will serve as parameters.
    The decorator works similar to @property, but the specified
    method correponds to the initialization of the argument, ussually
    checking for types and setting default values '''


@add_tdl_descriptor
@add_init_descriptor
class InputModel(InputArgument):
    ''' Decorator used to specify the input models for a model.
    The decorator works similar to @property, but the specified
    method correponds to the initialization of the argument, ussually
    checking for types and setting default values '''


@add_tdl_descriptor
@add_init_descriptor
class InferenceInput(InputArgument):
    ''' Decorator used to specify a model input required to perform inference.
    The decorator works similar to @property, but the specified
    method correponds to the initialization of the argument, ussually
    checking for types and setting default values '''


@add_tdl_descriptor
@add_init_descriptor
class SubmodelInit(object):
    '''Indicate the initialization function for a property

    Examples:
        Some examples of how can be used ::

            class TestObject0(tdl.common.TdlObject):
                @tdl.core.SubmodelInit
                def submodel(self, x, y):
                    return tdl.core.SimpleNamespace(x=x, y=y)

            class TestObject1(tdl.common.TdlObject):
                submodel = tdl.core.SubmodelInit(inference_input=True)

                @submodel.initializer
                def submodel(self, x, y):
                    return tdl.core.SimpleNamespace(x=x, y=y)

            class TestObject2(tdl.common.TdlObject):
                @tdl.core.SubmodelInit(inference_input=True)
                def submodel(self, x, y):
                    return tdl.core.SimpleNamespace(x=x, y=y)
    '''
    class Initializer(object):
        def __init__(self, obj, attr_name, finit):
            self._finit = finit
            self._obj = obj
            self._attr_name = attr_name
            self._given_args = None

            def wrap_init(initializer, finit):
                @functools.wraps(finit)
                def init(self, *args, **kargs):
                    if not args and not kargs and self._given_args:
                        kargs = self._given_args
                    # set the attribute using finit method
                    if hasattr(self._obj, 'scope'):
                        with tf.name_scope(self._obj.scope):
                            with tf.name_scope(self._attr_name):
                                setattr(self._obj, self._attr_name,
                                        self._finit(self._obj, *args, **kargs))
                    else:
                        setattr(self._obj, self._attr_name,
                                self._finit(self._obj, *args, **kargs))

        def set_args(self, given_args):
            assert isinstance(given_args, dict)
            self._given_args = given_args

        def init(self, *args, **kargs):
            if self._given_args:
                kargs.update(self._given_args)
            # set the attribute using finit method
            if hasattr(self._obj, 'scope'):
                with tf.name_scope(self._obj.scope):
                    with tf.name_scope(self._attr_name):
                        setattr(self._obj, self._attr_name,
                                self._finit(self._obj, *args, **kargs))
            else:
                setattr(self._obj, self._attr_name,
                        self._finit(self._obj, *args, **kargs))

    def __init__(self, finit=None, inference_input=None, lazzy=False,
                 doc=None):
        """Defines the initialization method of a property.

        Args:
            finit (callable): Function that initializes the parameter.
                The function should return the value for the parameter.
                Defaults to None.
            inference_input (bool): indicates if the property should be
                interpreted as inference inputs
            lazzy (bool): indicates if the property should be initialized
                only when assert_initialized is called.
        """
        self.finit = finit
        self.inference_input = (False if inference_input is None
                                else inference_input)
        self.lazzy = lazzy
        self.name = (None if self.finit is None
                     else finit.__name__)
        if not doc and finit is not None:
            if finit.__doc__:
                doc = finit.__doc__
            else:
                doc = ('Autoinit with arguments {}'
                       ''.format([arg for arg in _get_func_args(finit)
                                  if arg != 'self']))
        self.__doc__ = doc

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        if not hasattr(obj.__tdl__, self.name):
            value = SubmodelInit.Initializer(
                obj=obj, attr_name=self.name, finit=self.finit)
            setattr(obj.__tdl__, self.name, value)
        return getattr(obj.__tdl__, self.name)

    def __set__(self, obj, val):
        if self.finit is None:
            raise AttributeError('initializer for parameter {} '
                                 'not specified'.format(self.name))
        if not hasattr(obj, '__tdl__'):
            setattr(obj, '__tdl__', __TDL__(obj))
        if hasattr(obj.__tdl__, self.name):
            if not isinstance(getattr(obj.__tdl__, self.name),
                              SubmodelInit.Initializer):
                raise exceptions.PropertyRedefinition(
                    property=self.name, object=obj)
        setattr(obj.__tdl__, self.name, val)

    def initializer(self, finit):
        assert self.finit is None,\
            'the evaluation method has already been specified'
        return type(self)(finit=finit,
                          inference_input=self.inference_input,
                          doc=self.__doc__)

    def __call__(self, finit=None):
        assert self.finit is None,\
            'the evaluation method has already been specified'
        return type(self)(finit=finit,
                          inference_input=self.inference_input,
                          lazzy=self.lazzy,
                          doc=self.__doc__)

    def init(self, obj, val):
        """initialization method called when TdlModel is initialized

        Args:
            obj (TdlModel): object to which the property belongs to.
            val (type): value used for initialization. If value is
                a dictionary, the initialization method will be
                called using the dictionary values, otherwise, the
                property will be set with the provided value.
        """
        if isinstance(val, dict):
            if self.lazzy:
                self.__get__(obj, type(obj)).set_args(val)
            else:
                self.__get__(obj, type(obj)).init(**val)
        else:
            self.__set__(obj, val)

    def autoinit(self, obj, force=False):
        if is_autoinit_enabled(obj) or force:
            try:
                self.__get__(obj, type(obj)).init()
            except TypeError:
                raise exceptions.AutoInitFailed(property=self.name, object=obj)
        else:
            raise exceptions.UnsetProperty(property=self.name, object=obj)


@add_tdl_descriptor
@add_init_descriptor
class InputModelInit(SubmodelInit):
    pass


@add_tdl_descriptor
@add_init_descriptor
class ParameterInit(SubmodelInit):
    pass


def is_property_set(obj, prop):
    """Checks if a property has already been set.
    This function checks only if the property has been set. For attributes
    like MethodInit that are set using an initializer, this function
    returns True, even when they have not been initialized.

    Args:
        obj (TdlModel): model object.
        prop (str): property name.
    Returns:
        bool: True if property is set, False otherwise.
    """
    with DisableAutoinit(obj):
        try:
            attr = getattr(obj, prop)
        except exceptions.UnsetProperty:
            return False
        except exceptions.InitPreconditionsFailed:
            return False
    if isinstance(attr, SubmodelInit.Initializer):
        return False
    if isinstance(attr, OptionalPropertyWrapper):
        return attr.is_set
    return True


def is_property_initialized(obj, prop):
    """Checks if a property has been initialized.
    This function checks initialization even for properties that are set but
    not initialized, like MethodInit.

    Args:
        obj (TdlModel): model object.
        prop (str): property name.
    Returns:
        bool: True if property is set, False otherwise.
    """
    with DisableAutoinit(obj):
        try:
            attr = getattr(obj, prop)
        except exceptions.UnsetProperty:
            return False
        except exceptions.InitPreconditionsFailed:
            return False
    if isinstance(attr, SubmodelInit.Initializer):
        return False
    if isinstance(attr, OptionalPropertyWrapper):
        return attr.is_set
    if isinstance(attr, MethodInit.MethodData):
        return attr.initialized
    return True


def any_initialized(obj, props: typing.List[str]) -> bool:
    """checks if any property is initialized

    Args:
        obj: object that owns the properties.
        props (typing.List[str]): list of properties to be checked.
    Returns:
        bool: True if any property is already initialized.
    """
    return any(is_property_initialized(obj, prop) for prop in props)


def assert_initialized(object, prop, reqs):
    """Check if the requirements have been initialized.

    Args:
        object: object being initialized
        prop: property being initialized (string)
        reqs: list of properties that are required to initialize prop
            (list of strings)
    Raises:
        InitPreconditionsFailed: the exception is raised
            if any of the requirements is not initialized.
            During initialization, the exeption is handled by _init_tdl_attrs
    """
    initialized = [is_property_initialized(object, p) for p in reqs]
    if all(initialized):
        return
    elif is_autoinit_enabled(object):
        # attempt to auto initialize
        not_init = list(filter(
            lambda p: not is_property_initialized(object, p),
            reqs))
        for p in not_init:
            if (hasattr(getattr(type(object), p), 'autoinit')
                    and not is_property_initialized(object, p)):
                try:
                    getattr(type(object), p).autoinit(object)
                except exceptions.AutoInitFailed:
                    raise exceptions.InitPreconditionsFailed(
                        object=object, property=prop, reqs=not_init)

    initialized = [is_property_initialized(object, p) for p in reqs]
    if not all(initialized):
        not_init = list(filter(
            lambda p: not is_property_initialized(object, p),
            reqs))
        raise exceptions.InitPreconditionsFailed(
            object=object, property=prop, reqs=not_init)


def assert_any_available(object, property=None, reqs=None):
    """Check if requirements are available.

    This function checks if any of the requirements is already set.
    If no requirement is set, we rise an exeption and left _init_tdl_attrs
    function to handle it.

    The requirements will be initialized only if they were provided by the
    user, no auto-initialization is performed.

    Args:
        object (TdlModel): object being initialized/defined.
        reqs (str): requirements.
    """
    if reqs is None:
        raise ValueError('list of requirements must be provided')
    initialized = [is_property_set(object, p) for p in reqs]
    if any(initialized):
        return
    else:
        raise exceptions.NonePropertyAvailable(
            object=object, property=property, reqs=reqs)


def assert_initialized_if_available(object, property=None, reqs=None):
    """check that requirements are initialized if they were provided by the
    user, no auto-initialization is performed.

    Args:
        object (TdlModel): object being initialized/defined.
        reqs (str): requirements.
    """
    if reqs is None:
        raise ValueError('list of requirements must be provided')
    # get user provided attributes
    context = get_context(object)
    given_attrs = context.given_attrs
    if context.initialized is True:
        return
    # filter only requirements provided by the user
    reqs = [p for p in reqs if p in given_attrs]
    # check if initialized
    initialized = [is_property_initialized(object, p) for p in reqs]
    if all(initialized):
        return
    else:
        not_init = [p for p in reqs if not is_property_set(object, p)]
        raise exceptions.InitPreconditionsFailed(
            object=object, property=property, reqs=not_init)


def _find_tdl_attrs(cls, AttrClass, ignore=None, AttrBaseClass=None,
                    return_attr_class=False):
    """find attributes of class AttrClass.

    Args:
        cls (type): class to find the attributes.
        AttrClass (type, list): Types of descriptors to look for.
            The method looks for an exact Type match on this list.
        ignore (type): names to ignore.
        AttrBaseClass(type, list): Types of descriptors to look for.
            The method looks for instances that match this list.
        return_attr_class (bool): If false, return only the names.
            If true, return a list of tuples (name, class).
    Returns:
        list: list of attributes names.
    """
    if not isinstance(AttrClass, collections.Iterable):
        AttrClass = (AttrClass,)
    if AttrBaseClass is None:
        AttrBaseClass = ()
    if return_attr_class:
        names = [(x[0], type(x[1])) for x in inspect.getmembers(cls)
                 if (type(x[1]) in AttrClass or
                     any(isinstance(x[1], BaseClass)
                         for BaseClass in AttrBaseClass))]
    else:
        names = [x[0] for x in inspect.getmembers(cls)
                 if (type(x[1]) in AttrClass or
                     any(isinstance(x[1], BaseClass)
                         for BaseClass in AttrBaseClass))]

    return set(names)


def _init_tdl_attrs(obj, kargs, attr_name, AttrClass, allowed_autoinit=None):
    """Initialization of tdl parameters. This function automatically searches
    for tdl parameters in the class.

    Args:
        obj (type): object which owns the attributes to be initialized.
        kargs (type): dictionary with the user provided values.
        attr_name (type): name of the attributes (e.g. _parameters).
        AttrClass (type): class of the tdl attribute (e.g. SimpleParameter).
        allow_autoinit: list with attributes that are allowed to be
            auto-initialized
    Returns:
        (set): set of processed attributes
    """
    # get attributes
    _found = _find_tdl_attrs(type(obj), AttrClass)
    if hasattr(obj, attr_name):
        _given = set(getattr(obj, attr_name))
        assert _given <= _found, \
            'given {} attributes ({}) must be a subset of defined '\
            'attributes ({})'.format(attr_name, _given, _found)
        _diff = _found - _given
        _value = getattr(obj, attr_name) + list(_diff)
        setattr(obj.__tdl__, attr_name, _value)
    else:
        setattr(obj.__tdl__, attr_name, _found)
    # init attributes
    if EagerMethod == AttrClass:   # TODO: decide which interface to use
        for name in getattr(obj.__tdl__, attr_name):
            with tf.name_scope(name):
                if name in kargs:
                    if isinstance(kargs[name], collections.Iterable):
                        setattr(obj, name, kargs[name])
                    else:
                        setattr(obj, name, [kargs[name]])
                else:
                    setattr(obj, name, [None])
    else:
        init_queue = collections.deque(getattr(obj.__tdl__, attr_name))
        autoinit_set = set()
        autoinit_failed_set = set()
        allowed_autoinit = (
            set(init_queue) if allowed_autoinit is None
            else set.union(set(init_queue), set(allowed_autoinit)))
        COUNTER = 0
        while init_queue:
            COUNTER += 1
            if COUNTER > 100000:
                # raise ValueError('something went wrong!!!')
                import pdb
                pdb.set_trace()
            name = init_queue.popleft()
            if (name not in kargs) and (name not in autoinit_set):
                continue
            try:    # Attempt to initialize name
                if name in autoinit_set:
                    getattr(type(obj), name).autoinit(obj, force=True)
                    autoinit_set.remove(name)
                elif isinstance(kargs[name], autoinit.AutoInit):
                    getattr(type(obj), name).autoinit(obj, force=True)
                else:
                    getattr(type(obj), name).init(obj, kargs[name])
            except exceptions.InitPreconditionsFailed as error:
                reqs = error.reqs
                # check autoinit has not been tried before
                if name in autoinit_set:
                    if name in autoinit_failed_set:
                        raise exceptions.InitPreconditionsFailed(
                                obj, name, reqs)
                    autoinit_failed_set.add(name)
                # check autoinit is allowed for the requirements
                if not all(r in allowed_autoinit for r in reqs):
                    raise exceptions.InitPreconditionsFailed(
                        obj, name, reqs)
                for req in reqs:
                    # add requirement to autoinit set if it is not provided
                    if req not in kargs:
                        autoinit_set.add(req)
                    # Add requirement in init queue if not in there
                    if req not in init_queue:
                        init_queue.appendleft(req)
                init_queue.append(name)
            except exceptions.NonePropertyAvailable as error:
                if any([req in kargs for req in error.reqs]):
                    init_queue.appendleft(req)
                else:
                    raise exceptions.NonePropertyAvailable(
                        object=obj, property=name, reqs=error.reqs)
    return set(getattr(obj.__tdl__, attr_name))


def init_attrs(model, attrs=None, AttrTypes='default'):
    """Run auto-initialization of the model attributes.

    Args:
        model (TdlModel): model to initialize attributes.
        attrs (list): list with the names of the attributes to initialize.
        AttrTypes (str): tdl decorator or list of tdl decorators.
            Use 'default' to initialize (InputArgument, InputParameter,
            InputModel, SimpleParameter, Submodel).
    """
    if attrs is None:
        attrs = set()
    if AttrTypes is 'default':
        AttrTypes = (InputArgument, InputParameter, InputModel,
                     SimpleParameter, Submodel)
    if AttrTypes is not None:
        attrs_found = _find_tdl_attrs(type(model), AttrTypes)
        attrs.update(attrs_found)

    for attr_i in attrs:
        if not is_property_set(model, attr_i):
            getattr(type(model), attr_i).autoinit(model)


@add_autoinit_class
class TdlModel(TdlOp):

    def _tdl_check_kwargs(self, kwargs):
        '''custom defined function to check arguments before initialization'''
        return

    def __init__(self, **kargs):
        ''' Base initialization of a Tdl operation.

        Arguments should be explicitly specified. The basic acguments are:

        Args:
            name: name of the model/operation, used to create a scope,
                  if no name is provided, the function will look for
                  self._default_name.
            options: options for the model/operation.
            kargs: parameters corresponding to the specific model.
                submodels corresponding to the specific model.
        '''
        self._tdl_check_kwargs(kargs)
        name = (kargs['name'] if 'name' in kargs
                else self._default_name if hasattr(self, '_default_name')
                else None)
        name = (name if name is not None
                else type(self).__name__)

        options = (kargs['options'] if 'options' in kargs
                   else None)
        super(TdlModel, self).__init__(name=name, options=options)
        if not hasattr(self, '__tdl__'):
            setattr(self, '__tdl__', __TDL__(self))

        assert get_context(self).given_attrs is None
        get_context(self).given_attrs = \
            set([key for key, value in kargs.items()])

        with tf.name_scope(self.scope), DisableAutoinit(self):
            attrs_done = _init_tdl_attrs(
                self, kargs, '_input_args',
                (InputArgument, InputParameter, InputModel, InferenceInput,
                 InputModelInit))
            allowed_autoinit = attrs_done
            attrs_done = _init_tdl_attrs(
                self, kargs, '_parameters', (SimpleParameter, ParameterInit),
                allowed_autoinit=allowed_autoinit)
            allowed_autoinit = set.union(attrs_done, allowed_autoinit)
            attrs_done = _init_tdl_attrs(
                self, kargs, '_submodels',
                (Submodel, SubmodelWithArgs, SubmodelInit),
                allowed_autoinit=allowed_autoinit)
            allowed_autoinit = set.union(attrs_done, allowed_autoinit)
            attrs_done = _init_tdl_attrs(
                self, kargs, '_model_outputs', OutputValue,
                allowed_autoinit=allowed_autoinit)
            allowed_autoinit = set.union(attrs_done, allowed_autoinit)
            attrs_done = _init_tdl_attrs(
                self, kargs, '_optional', (Regularizer, OptionalProperty,
                                           MethodInit),
                allowed_autoinit=allowed_autoinit)
        assert_initialized(self, '__init__', self.__tdl__._model_outputs)
        get_context(self).initialized = True

    def add_weight(self, name, initializer, shape, trainable, **kargs):
        return variable(initial_value=initializer(shape=shape),
                        trainable=trainable,
                        name=name,
                        **kargs)


class TdlModelCallable(TdlModel):
    def __call__(self, inputs, *args, **kargs):
        with tf.name_scope(self.scope):
            if hasattr(self, 'input_shape'):
                try:
                    input_shape = tf.convert_to_tensor(inputs).shape
                except TypeError:
                    input_shape = None
                if input_shape is not None:
                    if is_property_initialized(self, 'input_shape'):
                        assert self.input_shape.is_compatible_with(input_shape)
                    else:
                        self.input_shape = input_shape
            build(self)
            output = self.call(inputs=inputs, *args, **kargs)
        return output


class encapsulate_op(object):
    def __init__(self, input_vars, output_vars=None):
        self._input_vars = input_vars
        if output_vars is None:
            self._output_vars = 'y'
        else:
            self._output_vars = output_vars

    def __call__(self, func):
        def encapsulate(*args, **kargs):
            # get name for the operation
            if 'name' in kargs:
                name = kargs['name']
            else:
                name = func.__name__
            # define class and set input arguments
            output = TdlOp(name=name, options=None)
            input_vars = dict()  # locals()
            for i, var_i in enumerate(self._input_vars):
                if i < len(args):
                    input_vars[var_i] = args[i]
                else:
                    input_vars[var_i] = kargs[var_i]
                setattr(output, var_i, input_vars[var_i])
            # add remaining explicitly specified variables
            for var_name, var_value in kargs.items():
                if (var_name not in input_vars and var_name != 'name'):
                    input_vars[var_name] = var_value
            # call function
            with tf.name_scope(output.scope):
                y = func(**input_vars)
            # set outputs
            if isinstance(y, collections.Iterable):
                for i, out_i in enumerate(self._output_vars):
                    setattr(output, out_i, y[i])
            else:
                setattr(output, self._output_vars, y)
            return output
        return encapsulate


class OutputModel(TdlModel):
    @InputModel
    def model(self, value):
        return value

    @InputArgument
    def _feval(self, value):
        return value

    @InputArgument
    def _outputs(self, value):
        return value

    @InputArgument
    def _inputs(self, value):
        return value

    def eval(self, input_vars):
        y = self._feval(self.model, **input_vars)


class ModelMethod(object):
    ''' Decorator used to specify an operation inside a model.

    The decorator works similar to @property, but the specified
    method correponds to the definition of the operation.

    Examples:
        Model methods can be used as follows ::

            class MyModel(tdl.TdlModel):
                _submodels = ['evaluate']
                @tdl.ModelMethod(['y'],     # list of outputs
                                 ['x' ,'y'] # list of inputs
                                )
                def evaluate(self, x, u):
                    return x+u

    '''

    _DefaultOutputClass = OutputModel

    def __init__(self, output_vars, input_vars, OutputClass=None):
        """Short summary.

        Args:
            output_vars ([str]): names of the output variables.
            input_vars ([str]): naes of the input variables.
            OutputClass (class): Class of the method's output.
                Defaults to None.
        """
        if isinstance(output_vars, str):
            output_vars = [output_vars]
        self._output_vars = output_vars

        if isinstance(input_vars, str):
            input_vars = [input_vars]
        self._input_vars = input_vars

        if OutputClass is None:
            self._OutputClass = type("DerivedModelInst",
                                     (self._DefaultOutputClass,), {})
        elif issubclass(OutputClass, self._DefaultOutputClass):
            self._OutputClass = OutputClass
        else:
            raise NotImplementedError("ModelMethod not implemented for {}"
                                      "".format(OutputClass))
        # if not hasattr(self._OutputClass, 'model'):
        #     # def model(self, value): return value
        #     self._OutputClass.model = Submodel(lambda self, value: None)
        # for input_i in self._input_vars:
        #     if not hasattr(self._OutputClass, input_i):
        #         setattr(self._OutputClass, input_i,
        #                 InputArgument(lambda self, value: None))

    def __call__(self, method):
        def encapsulate(model, *args, **kargs):
            # get name for the operation
            if method.__name__ == 'evaluate':
                default_name = model.name
            else:
                default_name = '{}/{}'.format(model.name, method.__name__)
            if 'name' in kargs:
                name = (kargs['name'] if kargs['name'] is not None
                        else default_name)
                del kargs['name']
            else:
                name = default_name
            # get input args
            assert (len(args) + len(kargs) <= len(self._input_vars)),\
                'provided input attributes exceed the number of defined '\
                'attributes'
            input_vars = {self._input_vars[i]: value
                          for i, value in enumerate(args)}
            input_vars.update({key: value
                               for key, value in kargs.items()
                               if key in self._input_vars})
            # define class and set input arguments
            output = self._OutputClass(model=model, name=name, options=None,
                                       _inputs=list(input_vars.keys()),
                                       _outputs=list(self._output_vars),
                                       _feval=method)
            # set input variables
            for key, value in input_vars.items():
                setattr(output, key, value)
            # add remaining explicitly specified variables
            for var_name, var_value in kargs.items():
                if (var_name not in input_vars and var_name != 'name'):
                    raise ValueError('provided argument {} has not been '
                                     'specified in the method definition'
                                     ''.format(var_name))
                    input_vars[var_name] = var_value
            # call method
            with tf.name_scope(output.scope):
                y = method(model, object=output, **input_vars)
            # set outputs
            if self._output_vars:
                y = ([y] if len(self._output_vars) == 1
                     else y)
                for i, out_i in enumerate(self._output_vars):
                    setattr(output, out_i, y[i])
            return output
        return encapsulate


class TdlProgram(object):
    ''' Defines a program that executes operations over TdlModel instances '''
    @property
    def name(self):
        ''' name for the model '''
        return self._name

    @name.setter
    def name(self, value):
        def is_absolute(name):
            return name[0] == '/' or name[-1] == '/'
        assert not hasattr(self, '_name'),\
            'name can only be set once'
        if is_absolute(value):
            with tf.name_scope(value) as scope:
                self._scope = scope
            self._name = self.scope.split('/')[-2]
        else:
            self._name = value

    @property
    def scope(self):
        ''' scope for the model, used to define all operations '''
        if not hasattr(self, '_scope'):
            assert hasattr(self, '_name'), \
                'attempting to create scope with undefined name'
            with tf.name_scope(self.name) as scope:
                self._scope = scope
        return self._scope

    @property
    def options(self):
        return self._options

    def _init_options(self, options):
        options = (dict() if options is None
                   else options)
        return options

    def __init__(self, options=None, name=None, **kargs):
        """Initialization of TdlProgram. This method calls the Initialization
        methods for all attributes defined using twodlearn decorators

        Args:
            **kargs (type): arguments for attributes defined using tdl
                decorators (e.g. arguments for EagerMethod attributes).
        """
        # name
        self.name = (type(self).__name__ if name is None
                     else name)
        # tdl descriptors
        if not hasattr(self, '__tdl__'):
            self.__tdl__ = __TDL__(self)
        # initialize options if needed
        if not hasattr(self, '_options'):
            self._options = self._init_options(options)
        # initialize models
        with tf.name_scope(self.scope), DisableAutoinit(self):
            _init_tdl_attrs(self, kargs, '_input_args', InputArgument)
            _init_tdl_attrs(self, kargs, '_parameters', SimpleParameter)
            _init_tdl_attrs(self, kargs, '_submodels', Submodel)
            # initialize eager methods
            for name, value in kargs.items():
                if not isinstance(value, collections.Iterable):
                    kargs[name] = [kargs[name]]
            _init_tdl_attrs(self, kargs, '_eager', EagerMethod)


class EagerMethod(object):
    ''' Decorator used to specify methods that perform operations using
    tensorflow, but are evaluated immediatly.
    These methods consist of an initialization method (speficied with the
    function given to the decorator) and an execute '''
    def __init__(self, finit=None, feval=None):
        self.finit = finit
        self.feval = feval
        self.name = (finit.__name__ if finit is not None
                     else None)

    def eval(self, feval):
        assert self.feval is None,\
            'the evaluation method has already been specified'
        return type(self)(finit=self.finit, feval=feval)

    def __set__(self, obj, argv):
        if obj is None:
            return self
        if self.finit is None:
            raise AttributeError('initializer for EagerMethod {}'
                                 'has not been specified'.format(self))
        if self.feval is None:
            raise AttributeError('evaluation function for EagerMethod {}'
                                 'has not been specified'.format(self))
        if not hasattr(obj, '__tdl__'):
            setattr(obj, '__tdl__', __TDL__(obj))
        # setattr(obj.__tdl__, self.name,
        #         lambda *args, **kargs: self.feval(obj, *args, **kargs))
        setattr(obj.__tdl__, self.name,
                types.MethodType(self.feval, obj))
        return self.finit(obj, *argv)

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        return getattr(obj.__tdl__, self.name)


class EncapsulatedMethod(object):
    ''' Decorator used to specify methods that have a set of local variables.

    These methods consist of an initialization method (speficied with the
    function given to the decorator) and an execute method.
    '''
    class MethodData(object):
        def __init__(self, func):
            self.func = func
            self.local = SimpleNamespace()

        def __call__(self, *args, **kargs):
            return self.func(self.local, *args, **kargs)

    def __init__(self, finit=None, feval=None):
        self.finit = finit
        self.feval = feval
        self.name = (finit.__name__ if finit is not None
                     else None)

    def eval(self, feval):
        assert self.feval is None,\
            'the evaluation method has already been specified'
        return type(self)(self.finit, feval)

    def init_local(self, obj, value):
        if self.finit is None:
            raise AttributeError(
                'initializer for EncapsulatedMethod {}'
                'has not been specified'.format(self))
        if self.feval is None:
            raise AttributeError(
                'evaluation function for EncapsulatedMethod {}'
                'has not been specified'.format(self))
        if not hasattr(obj, '__tdl__'):
            setattr(obj, '__tdl__', __TDL__(obj))

        method_data = EncapsulatedMethod.MethodData(
            func=types.MethodType(self.feval, obj))
        setattr(obj.__tdl__, self.name, method_data)
        self.finit(obj, method_data.local, value)

    def init(self, obj, value):
        self.init_local(obj, value)

    def __set__(self, obj, val):
        raise AttributeError('EncapsulatedMethod cannot be set')

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        if not hasattr(obj, '__tdl__'):
            self.init_local(obj, None)
        if not hasattr(obj.__tdl__, self.name):
            self.init_local(obj, None)
        return getattr(obj.__tdl__, self.name)


@add_tdl_descriptor
@add_init_descriptor
class MethodInit(object):
    '''Descriptor that encapsulates a set of local variables for the method.
    The following is an example of how this descriptor can be used:

    Examples:
        The descriptor can be used to define both, an initialization method,
        and an evaluation method ::

            class Example(tdl.core.TdlModel):
                @tdl.core.MethodInit
                def evaluate(self, local, units=20):
                    local.model = tdf.core.dense.DenseLayer(units=units)

                @evaluate.eval
                def evaluate(self, local, inputs, n_samples=100):
                    return local.model(inputs)/n_samples
    '''
    class MethodData(object):
        def __init__(self, obj, attr_name, finit, feval):
            self._finit = finit
            self._feval = types.MethodType(feval, obj)
            self._obj = obj
            self._attr_name = attr_name
            self.initialized = False
            self.local = SimpleNamespace()

        def init(self, *args, **kargs):
            # set the attribute using finit method
            assert self.initialized is False,\
                'Method {} from object {} already initialized'\
                ''.format(self._attr_name, self._obj)
            if hasattr(self._obj, 'scope'):
                with tf.name_scope(self._obj.scope):
                    with tf.name_scope(self._attr_name):
                        self._finit(self._obj, local=self.local,
                                    *args, **kargs)
            else:
                self._finit(self._obj, local=self.local,
                            *args, **kargs)
            self.initialized = True

        def __call__(self, *args, **kargs):
            if not self.initialized:
                raise exceptions.InitPreconditionsFailed(
                    object=self._obj, property=self._attr_name,
                    reqs=['{}.init'.format(self._attr_name)])
            return self._feval(self.local, *args, **kargs)

    def __init__(self, finit=None, feval=None, doc=None):
        """Initialization of MethodInit descriptor.

        Args:
            finit (callable): Function that initialize the locals.
                Defaults to None.
            feval (callable): Function that executes the function.
        """
        self.finit = finit
        self.feval = feval
        self.name = finit.__name__
        if doc is None and finit is not None:
            doc = finit.__doc__
        self.__doc__ = doc

    def eval(self, feval):
        assert self.feval is None,\
            'the evaluation method has already been specified'
        return type(self)(finit=self.finit, feval=feval, doc=self.__doc__)

    def init(self, obj, val=None):
        assert (self.finit is not None), 'Unspecified finit method'
        assert (self.feval is not None), 'Unspecified feval method'
        if not hasattr(obj, '__tdl__'):
            setattr(obj, '__tdl__', __TDL__(obj))
        if hasattr(obj.__tdl__, self.name):
            raise exceptions.PropertyRedefinition(
                property=self.name, object=obj)
        else:
            functor = MethodInit.MethodData(obj=obj, attr_name=self.name,
                                            finit=self.finit, feval=self.feval)
            if val is not None:
                functor.init(**val)
            setattr(obj.__tdl__, self.name, functor)

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        if not hasattr(obj, '__tdl__'):
            setattr(obj, '__tdl__', __TDL__(obj))
        if not hasattr(obj.__tdl__, self.name):
            self.init(obj)
        return getattr(obj.__tdl__, self.name)

    def __set__(self, obj, val):
        raise AttributeError('MethodInit cannot be set')

    def autoinit(self, obj, force=False):
        if is_autoinit_enabled(obj) or force:
            initializer = getattr(obj, self.name)
            initializer.init()
        else:
            raise exceptions.UnsetProperty(property=self.name, object=obj)


def hasattr(object, name):
    '''Checks if the object has the attribute name bypassing the
    autoinitialization procedure.'''
    with DisableAutoinit(object):
        try:
            hasit = py_hasattr(object, name)
        except exceptions.UnsetProperty:
            return True
        except exceptions.InitPreconditionsFailed:
            return True
    return hasit


def build(model, recursive=False):
    '''initialize parameters and submodels of a given model.

    Args:
        recursive: True for building the parameters and submodels.
    '''
    input_desc = (InputArgument, InputParameter, InputModel, InferenceInput,
                  InputModelInit)
    param_desc = (SimpleParameter, ParameterInit)
    model_desc = (Submodel, SubmodelWithArgs, SubmodelInit)
    init_queue = collections.deque(
        _find_tdl_attrs(type(model), AttrClass=input_desc))
    init_queue.extendleft(
        _find_tdl_attrs(type(model), AttrClass=param_desc))
    init_queue.extendleft(
        _find_tdl_attrs(type(model), AttrClass=model_desc))
    failed = set()
    while init_queue:
        attr_name = init_queue.popleft()
        try:
            if not is_property_initialized(model, attr_name):
                getattr(type(model), attr_name).autoinit(model)
            if recursive:
                attr = getattr(model, attr_name)
                if isinstance(attr, (TdlModel)):
                    build(attr)
        except exceptions.InitPreconditionsFailed as error:
            reqs = error.reqs
            if attr_name in failed:
                raise exceptions.InitPreconditionsFailed(
                    model, attr_name, reqs)
            failed.add(attr_name)
            init_queue.append(attr_name)


class TdlObject(object):
    def __init__(self, **kargs):
        """Initialization of TdlObject. This method calls the Initialization
        methods for all attributes defined using twodlearn decorators

        Args:
            **kargs (type): arguments for attributes defined using tdl
                decorators (e.g. arguments for EncapsulatedMethod attributes).
        """
        if not hasattr(self, '__tdl__'):
            self.__tdl__ = __TDL__(self)
        # initialize encapsulated methods
        with DisableAutoinit(self):
            _init_tdl_attrs(self, kargs, '_encapsulated', EncapsulatedMethod)
            _init_tdl_attrs(self, kargs, '_optional', OptionalProperty)


class ScopedMethod(object):
    ''' adds a tf.name_scope before executing the method '''
    # def scoped_method(obj, *args, **kargs):
    #     with tf.name_scope(obj.scope), tf.name_scope(feval.__name__):
    #         return feval(obj, *args, **kargs)
    # return scoped_method
    class ScopedWrapper(object):
        def __init__(self, instance, feval, name):
            self.instance = instance
            self.feval = types.MethodType(feval, instance)
            self.name = name

        def __call__(self, *args, **kwargs):
            with tf.name_scope(self.instance.scope), tf.name_scope(self.name):
                return self.feval(*args, **kwargs)

    def __init__(self, feval=None):
        """adds a tf.name_scope before executing the method.
        Args:
            feval (callable): method that executes scoped operations.
        """
        self.feval = feval
        self.name = feval.__name__

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        if self.feval is None:
            raise AttributeError('function for the method {} has not been '
                                 'specified'.format(self.name))
        feval = self.feval
        feval_name = self.name
        def scoped_method(self, *args, **kwargs):
            with tf.name_scope(obj.scope), tf.name_scope(feval_name):
                return feval(self, *args, **kwargs)
        return types.MethodType(scoped_method, obj)
        #return ScopedMethod.ScopedWrapper(instance=obj, feval=self.feval,
        #                                  name=self.name)

    def __set__(self, obj, val):
        raise AttributeError('EncapsulatedMethod cannot be set')


# -------------------------- Common Models ------------------------ #

class TransformedVariable(TdlModel):
    _parameters = ['raw']

    @SimpleParameter
    def raw(self, kargs):
        ''' raw value before making a transformation '''
        return tf.Variable(**kargs)

    @property
    def value(self):
        return self._value

    @property
    def initializer(self):
        return self.raw.initializer

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

    def __init__(self, initializer,
                 initial_value=None, trainable=True,
                 collections=None, validate_shape=True,
                 caching_device=None, name='variable',
                 variable_def=None, dtype=None,
                 expected_shape=None, import_scope=None,
                 constraint=None, options=None):

        initial_value = self.inverse(initial_value)
        variable_args = {'initial_value': initializer(initial_value),
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
        super(TransformedVariable, self).__init__(raw=variable_args,
                                                  name=name, options=options)
        with tf.name_scope(self.scope):
            self._value = self.transform(self.raw)


class PositiveVariable(TransformedVariable):
    ''' Creates a variable that can only take positive values '''

    def inverse(self, value):
        # if isinstance(value, tf.Tensor):
        #    return tf.sqrt(value)
        # else:
        #    return np.sqrt(value)
        return value

    def transform(self, value):
        return tf.nn.softplus(value)


class PositiveVariableExp(TransformedVariable):
    ''' Creates a variable that can only take positive values.
    This function uses exp() as a reparameterization of the variable'''

    def inverse(self, value):
        if isinstance(value, tf.Tensor):
            return tf.log(value)
        else:
            return np.log(value).astype(np.float32)

    def transform(self, value):
        return tf.exp(value)


class PositiveVariable2(TransformedVariable):
    ''' Creates a variable that can only take positive values.
    This function uses pow(x,2) as a reparameterization of the variable'''

    def inverse(self, value):
        if isinstance(value, tf.Tensor):
            return tf.sqrt(value)
        else:
            return np.sqrt(value).astype(np.float32)

    def transform(self, value):
        return tf.pow(value, 2.0)


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


class ConstrainedVariable(TdlModel):
    @property
    def value(self):
        return self.variable.value()

    @property
    def initializer(self):
        return self.variable.initializer

    @InputArgument
    def min(self, value):
        if isinstance(value, tf.Tensor):
            return value
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        return value.astype(global_options.float.nptype)

    @InputArgument
    def max(self, value):
        if isinstance(value, tf.Tensor):
            return value
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        return value.astype(global_options.float.nptype)

    @InputArgument
    def initial_value(self, value):
        return value

    def projection(self, x):
        return tf.clip_by_value(x, clip_value_min=self.min,
                                clip_value_max=self.max)

    @SimpleParameter
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


class Identity(TdlModel):
    def evaluate(self, x):
        return x

    def __call__(self, x):
        return self.evaluate(x)

# ------------------------ Conversion functions ------------------------ #


def convert_variable_to_tensor(value, dtype=None, name=None, as_ref=False):
    if dtype is None:
        return value.value
    else:
        return tf.convert_to_tensor(value.value, dtype=dtype, name=name)


tf.register_tensor_conversion_function(
    base_type=TransformedVariable,
    conversion_func=convert_variable_to_tensor,
    priority=100)


def convert_output_to_tensor(value, dtype=None, name=None, as_ref=False):
    output_value = value.value
    if isinstance(output_value, tf.Tensor):
        value = output_value
    elif isinstance(value.value, np.ndarray):
        value = tf.convert_to_tensor(output_value, dtype=dtype)
    else:
        value = convert_output_to_tensor(output_value, dtype=dtype, name=name,
                                         as_ref=False)
    return (value if dtype is None
            else tf.convert_to_tensor(value, dtype=dtype, name=name))


tf.register_tensor_conversion_function(
    base_type=(TdlOp, SimpleNamespace),
    conversion_func=convert_output_to_tensor,
    priority=100)


# -------------------------- Common operations ------------------------ #

def tensor_rank(tensor):
    assert isinstance(tensor, (tf.Tensor, tf.Variable, np.ndarray)),\
        'Unrecognized tensor type {}'.format(type(tensor))
    return sum([d > 1 for d in tensor.shape])


def create_init_docstring(cls):
    '''Add a docstring to the initialization method of a tdl class.'''
    def get_first_line(doc):
        return doc.split('\n\n')[0].replace('\n', ' ')

    tdl_attrs = _find_tdl_attrs(cls, AttrClass=(),
                                AttrBaseClass=TDL_INIT_DESCRIPTORS,
                                return_attr_class=True)
    # doc = inspect.getdoc(cls.__init__)
    doc = ('' if cls.__doc__ is None
           else inspect.getdoc(cls) + '\n\n')
    doc = (doc +
           'Tdl autoinitialization with arguments: \n\n'
           'Attributes:\n')
    for name, descriptor_cls in tdl_attrs:
        descriptor_doc = inspect.getdoc(getattr(cls, name))
        if descriptor_doc is None:
            descriptor_doc = ''
        doc = doc + '    {}: (:class:`~twodlearn.core.common.{}`) {} \n\n'.format(
            name, descriptor_cls.__name__,
            get_first_line(descriptor_doc))
    if PYTHON_VERSION >= 3:
        cls.__doc__ = doc
    else:
        cls.__init__.__func__.__doc__ = doc
    return cls
