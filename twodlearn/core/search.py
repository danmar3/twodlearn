import collections
import tensorflow as tf
import numpy as np
from . import nest
from . import common
from .common import (TdlModel, SimpleNamespace,
                     _find_tdl_attrs,
                     InputArgument, InputParameter, InputModel,
                     InferenceInput, InputModelInit,
                     SimpleParameter, ParameterInit,
                     Submodel, SubmodelWithArgs, SubmodelInit,
                     )


# Classes that can be sarched for variables
SEARCHABLE_CLASSES = (tf.Variable, TdlModel, tf.keras.layers.Layer,
                      SimpleNamespace)


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
    '''find the parameters in the model.

    Args:
        model (TdlModel, TdlLayer): model or nested sequence of models.
            Supports SingleNamespaces as well.
        include_inputs (bool): propagate the search through the inputs of the
            model.
    Returns:
        list: list of parameters.
    '''
    def is_a_valid_model(attr):
        return isinstance(attr, SEARCHABLE_CLASSES + (tf.Tensor, NoScopeParam))

    def is_a_valid_list(attr):
        if not isinstance(attr, list):
            return False
        return all([is_a_valid_model(val) for val in attr])

    def list_with_any_valid_model(attr):
        if not isinstance(attr, list):
            return False
        return any([is_a_valid_model(val) for val in attr])

    def namespace_to_list(model):
        return [m for m in model.__dict__.values()
                if is_a_valid_model(m) or is_a_valid_list(m)]

    # convert nested structures to list
    if isinstance(model, SimpleNamespace):
        model = namespace_to_list(model)
    elif nest.is_nested(model):
        model = [mj
                 for m in nest.flatten(model)
                 for mj in (namespace_to_list(m)
                            if isinstance(m, SimpleNamespace)
                            else [m])
                 if is_a_valid_model(mj)]

    assert (is_a_valid_model(model) or is_a_valid_list(model)),\
        'model must be an instance of TdlModel, tf.Variable, tf.Tensor'\
        'or a list of them. '\
        'Got {} ({}) instead'.format(type(model), model)

    if not isinstance(model, (TdlModel, tf.keras.layers.Layer, list)):
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
    elif isinstance(model, tf.keras.layers.Layer):
        return set.union(*[set(model.trainable_weights),
                           set(model.trainable_variables),
                           set(model.non_trainable_weights),
                           set(model.non_trainable_variables)])


def get_variables_set(model):
    def namespace_to_list(model):
        return [m for m in model.__dict__.values()]

    this = get_variables_set

    if isinstance(model, tf.Variable):
        return set([model])
    elif isinstance(model, (tf.TensorShape, tf.Tensor, np.ndarray)):
        return set([])
    elif isinstance(model, SimpleNamespace):
        return this(namespace_to_list(model))
    elif isinstance(model, str):
        return set([])
    elif nest.is_nested(model):
        vars = [this(mi) for mi in nest.flatten(model)]
    elif isinstance(model, dict):
        return this(list(model.values()))
    elif isinstance(model, collections.Iterable):
        if not isinstance(model, (list, tuple)):
            raise ValueError(f'unexpected iterable ({type(model)}): {model}')
        vars = [this(mi) for mi in model]
    elif isinstance(model, tf.keras.layers.Layer):
        vars = [set(model.trainable_weights),
                set(model.trainable_variables),
                set(model.non_trainable_weights),
                set(model.non_trainable_variables)]
    elif isinstance(model, TdlModel):
        attr_names = _find_tdl_attrs(
            type(model),  AttrClass=common.TDL_DESCRIPTORS)
        return this([
            getattr(model, name) for name in attr_names
            if common.is_property_initialized(model, name)])
    else:
        return set([])
    if vars:
        return set.union(*vars)
    else:
        return set([])


def get_variables(model, *args, **kargs):
    '''get tf variables of a model

    Args:
        model (TdlModel, TdlLayer): model or nested sequence of models.
            Supports SingleNamespaces as well.
    Returns:
        list: list of tf variables.
    '''
    if args or kargs:
        raise ValueError('get_variables now does not support include_inputs '
                         'anymore.')
    return list(get_variables_set(model))


def _get_variables_deprecated(model, include_inputs=True):
    '''get tf variables of a model

    Args:
        model (TdlModel, TdlLayer): model or nested sequence of models.
            Supports SingleNamespaces as well.
        include_inputs (bool): propagate the search through the inputs of the
            model.
    Returns:
        list: list of tf variables.
    '''
    params = get_parameters(model, include_inputs=include_inputs)
    if params:
        extra = [get_parameters(p, include_inputs=include_inputs)
                 for p in params
                 if isinstance(p, (TdlModel, tf.Variable))]
        if extra:
            params = set.union(*extra)
        params = [mi for mi in params
                  if isinstance(mi, tf.Variable)]
    return params


if [int(s) for s in tf.__version__.split('.')[0:2]] < [1, 10]:
    def is_trainable(variable, scope=None):
        '''check if a variable is trainable'''
        if isinstance(variable, tf.Tensor):
            return False
        elif isinstance(variable, tf.Variable):
            return variable in tf.trainable_variables()
        else:
            raise TypeError("Type {} not recognized".format(type(variable)))
else:
    def is_trainable(variable, scope=None):
        '''check if a variable is trainable'''
        if isinstance(variable, tf.Tensor):
            return False
        elif isinstance(variable, tf.Variable):
            return variable.trainable
        else:
            raise TypeError("Type {} not recognized".format(type(variable)))


def get_trainable(model, *args, **kargs):
    '''get trainable variables of a model

    Args:
        model (TdlModel, TdlLayer): model or nested sequence of models.
            Supports SingleNamespaces as well.
        include_inputs (bool): propagate the search through the inputs of the
            model.
    Returns:
        list: list of trainable variables.
    '''
    if args or kargs:
        raise ValueError('get_trainable now does not support include_inputs '
                         'anymore.')
    params = get_variables(model)
    params = [mi for mi in params if is_trainable(mi)]
    return params


def get_placeholders(model):
    """Find the placeholders of a TdlModel following attributes defined
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
    """find placeholder in a model.

    Works similar to get_placeholders, but raises an error if more than one
    placeholder is found.

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


def initialize_variables(model):
    '''Initialize variables for a given model that have not been initialized.
    '''
    tf.compat.v1.report_uninitialized_variables(get_variables(model)).eval()
    model_vars = get_variables(model)
    name_notinit = tf.compat.v1.report_uninitialized_variables(model_vars)\
                     .eval()
    uninitialized = [var for var in model_vars
                     if var.name.split(':')[0].encode() in name_notinit]
    variables_initializer(uninitialized).run()


def find_instances(model, classinfo):
    '''Finds any attribute in the model that is an instance of classinfo.
    Objects must be Hashable.

    Args:
        classinfo: class or tuple of type objects.
    Returns:
        set: set with the instances of classinfo found in the
            model.
    '''
    if isinstance(model, classinfo):
        return set([model])
    if nest.is_nested(model):
        model = [mi for mi in nest.flatten(model)
                 if isinstance(mi, collections.Hashable)]
        if isinstance(model, dict):
            model = model.values()
        model_set = set(model)
        direct_instances = set([mi for mi in model_set
                                if isinstance(mi, classinfo)])
        searchable = [mi for mi in (model_set - direct_instances)
                      if isinstance(mi, SEARCHABLE_CLASSES)]
        found_instances = [find_instances(mi, classinfo)
                           for mi in searchable]
        if found_instances:
            return direct_instances | set.union(*found_instances)
        else:
            return direct_instances
    if isinstance(model, SEARCHABLE_CLASSES):
        attr_names = _find_tdl_attrs(
            type(model),  AttrClass=common.TDL_DESCRIPTORS)
        attrs = [getattr(model, name) for name in attr_names
                 if common.is_property_initialized(model, name)]
        instances = [find_instances(attr_i, classinfo)
                     for attr_i in attrs]
        return (set.union(*instances) if instances
                else set())
    else:
        return set()
