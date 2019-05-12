#  ***********************************************************************
#   This file defines common structures used in twodlearn
#
#   Wrote by: Daniel L. Marino (marinodl@vcu.edu)
#    Modern Heuristics Research Group (MHRG)
#    Virginia Commonwealth University (VCU), Richmond, VA
#    http://www.people.vcu.edu/~mmanic/
#
#   ***********************************************************************
import sys
try:
    set
except NameError:
    from sets import Set as set
import types
import inspect
import warnings
import numpy as np
import collections
import tensorflow as tf
from .options import global_options, DisableAutoinit
from . import exceptions
from . import autoinit
try:
    from types import SimpleNamespace as PySimpleNamespace
except ImportError:
    from argparse import Namespace as PySimpleNamespace

import pdb  # DEBUG

PYTHON_VERSION = sys.version_info[0]


def merge_dicts(a, b):
    # if PYTHON_VERSION >= 3:
    #    print('using python: {}'.format(PYTHON_VERSION))
    #    return {**train_stats, **valid_stats}
    # else:
    z = a.copy()
    z.update(b)
    return z


class SimpleNamespace(PySimpleNamespace):
    ''' SimpleNamespace that works with tf.convert_to_tensor '''


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


class ModelBase(object):
    @property
    def name(self):
        ''' name for the model '''
        return self._name

    @name.setter
    def name(self, value):
        assert not hasattr(self, '_name'),\
            'name can only be set once'
        self._name = value

    @property
    def scope(self):
        ''' scope for the model, used to define all operations '''
        assert hasattr(self, '_name'), \
            'attempting to create scope with undefined name'
        if not hasattr(self, '_scope'):
            with tf.name_scope(self.name) as scope:
                self._scope = scope
        return self._scope

    @property
    def n_inputs(self):
        return self._n_inputs

    @property
    def n_outputs(self):
        return self._n_outputs

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

    def setup(self, *args, **kargs):
        assert len(args) == 0,\
            'arguments for setup must be explicitly specified'
        return self.ModelOutput(self, **kargs)

    def __call__(self, x, name=None):
        return self.evaluate(x, name=name)

    # TODO: @classmethod
    # def get_default_options(cls):
    #    return cls._init_options(None, None, None)


class Placeholders(object):
    pass


class ModelEvaluation(ModelBase):
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        assert not hasattr(self, '_model'),\
            'model can only be set once'
        self._model = value

    @property
    def n_inputs(self):
        return self.model.n_inputs

    @property
    def n_outputs(self):
        return self.model.n_outputs

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        assert not hasattr(self, '_y'),\
            'y can only be set once'
        self._y = value

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        assert not hasattr(self, '_inputs'),\
            'inputs can only be set once'
        self._inputs = value

    @property
    def placeholders(self):
        return self._placeholders

    # def __init__(self, model, options=None, name=None):
    def __init__(self, *args, **kargs):
        def check_inputs(args, kargs):
            assert len(args) <= 1,\
                'To initialize ModelEvaluation base class, explicitly '\
                'indicate options=options and name=name'
            if 'model' in kargs:
                model = kargs['model']
            else:
                model = args[0]
            if 'options' in kargs:
                options = kargs['options']
            else:
                options = None
            if 'name' in kargs:
                name = kargs['name']
            else:
                name = None
            return model, options, name

        model, options, name = check_inputs(args, kargs)
        if name is None:
            if global_options.reuse_scope:
                name = model.scope
            else:
                name = model.name
        super(ModelEvaluation, self).__init__(name, options)
        self.model = model
        self._placeholders = Placeholders()


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


class __TDL__(object):
    ''' class that stores all parameters and methods that are created using
    tdl decorators '''
    def __init__(self, obj):
        self.obj = obj
        self.context = SimpleNamespace(
            initialized=False,   # is the model initialized
            given_attrs=None,    # set with the user provided attrs
        )


def get_context(model):
    ''' get tdl context for a model.
    The context can be used to know if the model has been initialized
    '''
    if not isinstance(model, TdlModel):
        raise TypeError('context is only available to TdlModel objects')
    if '__tdl__' not in model.__dict__:
        raise AttributeError('It seems that model {} has not been initialized '
                             '(missing __tdl__ attribute)'.format(model))
    return model.__tdl__.context


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


class TdlDescriptor(object):
    ''' Decorator used to specify a parameter inside a model.
    The decorator works similar to @property, but the specified
    method correponds to the initialization of the parameter '''

    def __init__(self, finit=None):
        """Creates a new SingleParameter.

        Args:
            finit (callable): Function that initialize the parameter.
                The function should return the value for the parameter.
                Defaults to None.
        """
        self.finit = finit
        self.name = finit.__name__

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        # initialize if the property has not been set
        if not hasattr(obj.__tdl__, self.name):
            self.autoinit(obj)
        return getattr(obj.__tdl__, self.name)

    def __set__(self, obj, val):
        if self.finit is None:
            raise AttributeError('initializer for parameter {} '
                                 'not specified'.format(self.name))
        if not hasattr(obj, '__tdl__'):
            setattr(obj, '__tdl__', __TDL__(obj))
        # TODO: remove finit, which must be called only from init method
        # setattr(obj.__tdl__, self.name, self.finit(obj, val))
        self.init(obj, val)

    def init(self, obj, val):
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
        if global_options.is_autoinit_enabled(obj) or force:
            self.init(obj, None)
        else:
            raise exceptions.UnsetProperty(property=self.name, object=obj)


# class OptionalProperty(TdlDescriptor):
#     ''' Decorator used to specify an optional property inside a model.
#     The decorator works similar to @property, but the specified
#     method correponds to the initialization of the property '''
#     def __get__(self, obj, objtype):
#         if obj is None:
#             return self
#         if hasattr(obj.__tdl__, self.name):
#             value = getattr(obj.__tdl__, self.name)
#             return (value if value is not None
#                     else types.MethodType(self.finit, obj))
#         else:
#             return types.MethodType(self.finit, obj)
#
#     def __set__(self, obj, val):
#         if self.finit is None:
#             raise AttributeError('initializer for parameter {} '
#                                  'not specified'.format(self.name))
#         if not hasattr(obj, '__tdl__'):
#             setattr(obj, '__tdl__', __TDL__(obj))
#         setattr(obj.__tdl__, self.name, val)


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


class Regularizer(OptionalProperty):
    ''' Decorator used to specify a regularizer for a model.
    The decorator works similar to @property, but the specified
    method correponds to the initialization of the regularizer'''


class SimpleParameter(TdlDescriptor):
    ''' Decorator used to specify a parameter inside a model.
    The decorator works similar to @property, but the specified
    method correponds to the initialization of the parameter '''


class Submodel(TdlDescriptor):
    ''' Decorator used to specify a submodel inside a model.
    The decorator works similar to @property, but the specified
    method correponds to the initialization of the submodel '''


class OutputValue(TdlDescriptor):
    ''' Decorator used to specify the output value of a model.
    The decorator works similar to @property, but the specified
    method correponds to the definition of the output value '''


class SubmodelWithArgs(Submodel):
    ''' Decorator used to specify a submodel inside a model.
    The decorator works similar to @property, but the specified
    method correponds to the initialization of the submodel '''
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


class InputArgument(TdlDescriptor):
    ''' Decorator used to specify the input arguments for a model.
    The decorator works similar to @property, but the specified
    method correponds to the initialization of the argument, ussually
    checking for types and setting default values '''


class InputParameter(InputArgument):
    ''' Decorator used to specify the input arguments for a model.
    These inputs will serve as parameters.
    The decorator works similar to @property, but the specified
    method correponds to the initialization of the argument, ussually
    checking for types and setting default values '''


class InputModel(InputArgument):
    ''' Decorator used to specify the input models for a model.
    The decorator works similar to @property, but the specified
    method correponds to the initialization of the argument, ussually
    checking for types and setting default values '''


class InferenceInput(InputArgument):
    ''' Decorator used to specify a model input required to perform inference.
    The decorator works similar to @property, but the specified
    method correponds to the initialization of the argument, ussually
    checking for types and setting default values '''


class SubmodelInit(object):
    '''Indicate the initialization function for a property

    Some examples of how can be used:

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

    def __init__(self, finit=None, inference_input=None):
        """Defines the initialization method of a property.
        Args:
            finit (callable): Function that initializes the parameter.
                The function should return the value for the parameter.
                Defaults to None.
            inference_input (bool): indicates if the property should be
                interpreted as inference inputs
        """
        self.finit = finit
        self.inference_input = (False if inference_input is None
                                else inference_input)
        self.name = (None if self.finit is None
                     else finit.__name__)

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        if hasattr(obj.__tdl__, self.name):
            return getattr(obj.__tdl__, self.name)
        else:
            # setter: lambda val: setattr(obj, self.name, val),
            return SubmodelInit.Initializer(
                obj=obj, attr_name=self.name, finit=self.finit)

    def __set__(self, obj, val):
        if self.finit is None:
            raise AttributeError('initializer for parameter {} '
                                 'not specified'.format(self.name))
        if not hasattr(obj, '__tdl__'):
            setattr(obj, '__tdl__', __TDL__(obj))
        if hasattr(obj.__tdl__, self.name):
            raise exceptions.PropertyRedefinition(property=self.name,
                                                  object=obj)
        setattr(obj.__tdl__, self.name, val)

    def initializer(self, finit):
        assert self.finit is None,\
            'the evaluation method has already been specified'
        return type(self)(finit=finit,
                          inference_input=self.inference_input)

    def __call__(self, finit=None):
        assert self.finit is None,\
            'the evaluation method has already been specified'
        return type(self)(finit=finit,
                          inference_input=self.inference_input)

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
            self.__get__(obj, type(obj)).init(**val)
        else:
            self.__set__(obj, val)

    def autoinit(self, obj, force=False):
        if global_options.is_autoinit_enabled(obj) or force:
            try:
                self.__get__(obj, type(obj)).init()
            except TypeError:
                raise exceptions.AutoInitFailed(property=self.name, object=obj)
        else:
            raise exceptions.UnsetProperty(property=self.name, object=obj)


class InputModelInit(SubmodelInit):
    pass


def is_property_set(obj, prop):
    """Checks if a property has already been set.
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
    initialized = [is_property_set(object, p) for p in reqs]
    if all(initialized):
        return
    elif global_options.is_autoinit_enabled(object):
        # attempt to auto initialize
        not_init = list(filter(lambda p: not is_property_set(object, p),
                               reqs))
        for p in not_init:
            if (hasattr(getattr(type(object), p), 'autoinit')
                    and not is_property_set(object, p)):
                try:
                    getattr(type(object), p).autoinit(object)
                except exceptions.AutoInitFailed:
                    raise exceptions.InitPreconditionsFailed(
                        object=object, property=prop, reqs=not_init)

    initialized = [is_property_set(object, p) for p in reqs]
    if not all(initialized):
        not_init = list(filter(lambda p: not is_property_set(object, p),
                               reqs))
        raise exceptions.InitPreconditionsFailed(
            object=object, property=prop, reqs=not_init)


def assert_any_available(object, property=None, reqs=None):
    """check if requirements are available.
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
    initialized = [is_property_set(object, p) for p in reqs]
    if all(initialized):
        return
    else:
        not_init = [p for p in reqs if not is_property_set(object, p)]
        raise exceptions.InitPreconditionsFailed(
            object=object, property=property, reqs=not_init)


def _find_tdl_attrs(cls, AttrClass, ignore=None):
    """find attributes of class AttrClass.
    Args:
        cls (type): class to find the attributes.
        AttrClass (type): attributes class.
        ignore (type): names to ignore.
    Returns:
        list: list of attributes names.
    """
    if not isinstance(AttrClass, collections.Iterable):
        AttrClass = (AttrClass,)
    names = [x[0] for x in inspect.getmembers(cls)
             if type(x[1]) in AttrClass]
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
        while init_queue:
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
    """ run auto-initialization of the model attributes
    Args:
        model (TdlModel): model to initialize attributes.
        attrs: list with the names of the attributes to initialize
        AttrTypes: tdl decorator or list of tdl decorators.
            Use 'default' to initialize
            (InputArgument, InputParameter, InputModel,
             SimpleParameter, Submodel)
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


class TdlModel(TdlOp):

    def __init__(self, **kargs):
        ''' Base initialization of a Tdl operation.
        Arguments should be explicitly specified. The basic acguments are:
            name: name of the model/operation, used to create a scope,
                  if no name is provided, the function will look for
                  self._default_name
            options: options for the model/operation.
            parameters corresponding to the specific model
            submodels corresponding to the specific model
        '''
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
                self, kargs, '_parameters', SimpleParameter,
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
                self, kargs, '_optional', (Regularizer, OptionalProperty),
                allowed_autoinit=allowed_autoinit)
        get_context(self).initialized = True


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
    method correponds to the definition of the operation
    Usage:
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
    function given to the decorator) and an execute method '''
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


class MethodInit(TdlDescriptor):
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

    def __init__(self, finit=None, feval=None):
        """Initialization of MethodInit descriptor.
        Args:
            finit (callable): Function that initialize the locals.
                Defaults to None.
            feval (callable): Function that executes the function.
        """
        self.finit = finit
        self.feval = feval
        self.name = finit.__name__

    def eval(self, feval):
        assert self.feval is None,\
            'the evaluation method has already been specified'
        return type(self)(finit=self.finit, feval=feval)

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        assert (self.finit is not None), 'Unspecified finit method'
        assert (self.feval is not None), 'Unspecified feval method'
        if not hasattr(obj, '__tdl__'):
            setattr(obj, '__tdl__', __TDL__(obj))
        if not hasattr(obj.__tdl__, self.name):
            functor = MethodInit.MethodData(obj=obj, attr_name=self.name,
                                            finit=self.finit, feval=self.feval)
            setattr(obj.__tdl__, self.name, functor)
        return getattr(obj.__tdl__, self.name)

    def __set__(self, obj, val):
        raise AttributeError('MethodInit cannot be set')


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
    if not hasattr(value, 'value'):
        raise AttributeError('ModelOutput {} does not have a \'value\' '
                             'property. Unable to convert output to tensor'
                             ''.format(value))
    if isinstance(value.value, tf.Tensor):
        value = value.value
    elif isinstance(value.value, np.ndarray):
        value = tf.convert_to_tensor(value.value, dtype=dtype)
    else:
        value = convert_output_to_tensor(value.value, dtype=dtype, name=name,
                                         as_ref=False)
    return (value if dtype is None
            else tf.convert_to_tensor(value, dtype=dtype, name=name))


tf.register_tensor_conversion_function(
    base_type=(ModelEvaluation, TdlOp, SimpleNamespace),
    conversion_func=convert_output_to_tensor,
    priority=100)


# -------------------------- Common operations ------------------------ #

def variables_initializer(var_list, name="init"):
    if var_list:
        return tf.group(*[v.initializer for v in var_list], name=name)
    return tf.no_op(name=name)


class NoScopeParam(object):
    def __init__(self, object, name, value):
        self.scope = '{}/{}'.format(object.scope, name)
        self.value = value

    def __eq__(self, other):
        return (self.__class__ == other.__class__ and
                self.scope == other.scope)

    def __hash__(self):
        return hash(self.scope)

    def __str__(self):
        return ("Parameter {}: {}"
                "".format(self.scope, type(self.value)))

    def __repr__(self):
        return ("Parameter {}: {}"
                "".format(self.scope, type(self.value)))


def get_parameters(model, include_inputs=False):
    def is_a_valid_model(attr):
        return isinstance(attr, (tf.Variable, tf.Tensor, TdlModel,
                                 NoScopeParam))

    def is_a_valid_list(attr):
        if not isinstance(attr, list):
            return False
        return all([is_a_valid_model(val) for val in attr])

    def list_with_any_valid_model(attr):
        if not isinstance(attr, list):
            return False
        return any([is_a_valid_model(val) for val in attr])

    if isinstance(model, SimpleNamespace):
        model = [m for m in model.__dict__.values()
                 if is_a_valid_model(m) or is_a_valid_list(m)]

    assert (is_a_valid_model(model) or is_a_valid_list(model)),\
        'model must be an instance of TdlModel, tf.Variable, tf.Tensor'\
        'or a list of them. '\
        'Got {} ({}) instead'.format(type(model), model)

    if not isinstance(model, (TdlModel, list)):
        return set([model])

    if isinstance(model, TdlModel):
        def _getparam(object, name):
            attr = getattr(object, name)
            if is_a_valid_model(attr):
                return set([attr])
            elif is_a_valid_list(attr):
                return set(attr)
            elif list_with_any_valid_model(attr):
                attr = [(val if is_a_valid_model(val)
                         else NoScopeParam(object=object,
                                           name='{}_{}'.format(name, idx),
                                           value=val))
                        for idx, val in enumerate(attr)]
                return set(attr)
            else:
                return set([NoScopeParam(object=object, name=name,
                                         value=attr)])

        def _getsubmodelparams(object, name):
            attr = getattr(object, name)
            if is_a_valid_model(attr) or is_a_valid_list(attr):
                return get_parameters(attr, include_inputs=include_inputs)
            elif list_with_any_valid_model(attr):
                attr = [(val if is_a_valid_model(val)
                         else NoScopeParam(object=object,
                                           name='{}_{}'.format(name, idx),
                                           value=val))
                        for idx, val in enumerate(attr)]
                return get_parameters(attr, include_inputs=include_inputs)
            else:
                return set([NoScopeParam(object=object, name=name,
                                         value=attr)])

        def _getinputsparams(object, name):
            return _getsubmodelparams(object, name)

        params = [_getparam(model, name)
                  for name in model.__tdl__._parameters]
        params = (set.union(*params) if params
                  else set())
        submodel_params = [_getsubmodelparams(model, mi)
                           for mi in model.__tdl__._submodels]
        params = (params | set.union(*submodel_params) if submodel_params
                  else params)
        if include_inputs:
            input_params = [_getinputsparams(model, mi)
                            for mi in model.__tdl__._input_args]
            params = (params | set.union(*input_params) if input_params
                      else params)
        return params
    elif isinstance(model, collections.Iterable):
        if not model:
            return set([])
        params = set.union(*[get_parameters(mi, include_inputs=include_inputs)
                             for mi in model])
        return params


def get_variables(model, include_inputs=True):
    params = get_parameters(model, include_inputs=include_inputs)
    if params:
        params = set.union(*[get_parameters(p, include_inputs=include_inputs)
                             for p in params
                             if isinstance(p, (TdlModel, tf.Variable))])
        params = [mi for mi in params
                  if isinstance(mi, tf.Variable)]
    return params


if [int(s) for s in tf.__version__.split('.')][0:2] < [1, 10]:
    def is_trainable(variable, scope=None):
        if isinstance(variable, tf.Tensor):
            return False
        elif isinstance(variable, tf.Variable):
            return variable in tf.trainable_variables()
        else:
            raise TypeError("Type {} not recognized".format(type(variable)))
else:
    def is_trainable(variable, scope=None):
        if isinstance(variable, tf.Tensor):
            return False
        elif isinstance(variable, tf.Variable):
            return variable.trainable
        else:
            raise TypeError("Type {} not recognized".format(type(variable)))


def get_trainable(model, include_inputs=True):
    params = get_parameters(model, include_inputs=include_inputs)
    params = set.union(*[get_parameters(p, include_inputs=include_inputs)
                         for p in params
                         if isinstance(p, (TdlModel, tf.Variable))])
    params = [mi for mi in params
              if isinstance(mi, tf.Variable)]
    params = [mi for mi in params if is_trainable(mi)]
    return params


def get_placeholders(model):
    """find the placeholders of a TdlModel following attributes defined
        using the InferenceInput decorator.
    Args:
        model (TdlModel): TdlModel instance..
    Returns:
        set: set of found placeholders.
    """
    def isplaceholder(obj):
        if isinstance(obj, tf.Tensor):
            return obj.op.type == u'Placeholder'
        else:
            return False

    def is_inference(obj, attr):
        descriptor = getattr(type(obj), attr)
        if isinstance(descriptor, InferenceInput):
            return True
        else:
            if hasattr(descriptor, 'inference_input'):
                return descriptor.inference_input
            else:
                return False

    if isplaceholder(model):
        return set([model])
    elif isinstance(model, list):
        plhdr = set([m for m in model if isplaceholder(m)])
        plhdr = set.union(plhdr, *[get_placeholders(m) for m in model
                                   if isinstance(m, TdlModel)])
        return plhdr
    else:
        assert isinstance(model, TdlModel),\
            'model is not an instance of TdlModel'
        valid_descriptors = (InferenceInput, InputModelInit)
        inputs = [getattr(model, name)
                  for name in _find_tdl_attrs(type(model), valid_descriptors)
                  if is_inference(model, name)]

        placeholders = set([input for input in inputs
                            if isplaceholder(input)])
        models = [input for input in inputs
                  if isinstance(input, (TdlModel, list))]
        placeholders = set.union(placeholders,
                                 *[get_placeholders(m) for m in models])
        return placeholders


def get_placeholder(model):
    """find placeholder of model. Works similar to get_placeholders,
        but raises an error if more than one placeholder is found.
    Args:
        model (type): Description of parameter `model`.
    Returns:
        type: Description of returned object.
    """
    placeholders = get_placeholders(model)
    if len(placeholders) == 1:
        return list(placeholders)[0]
    elif len(placeholders) == 0:
        raise ValueError('No placeholder was found')
    else:
        raise ValueError('Provided model has more than one placeholder')


def tensor_rank(tensor):
    assert isinstance(tensor, (tf.Tensor, tf.Variable, np.ndarray)),\
        'Unrecognized tensor type {}'.format(type(tensor))
    return sum([d > 1 for d in tensor.shape])
