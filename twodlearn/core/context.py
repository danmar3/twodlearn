import collections
from types import SimpleNamespace
TDL_AUTOINIT_CLASSES = set()


def add_autoinit_class(cls):
    '''Adds the class to the list of classes that have enabled
    autoinitialization.
    '''
    TDL_AUTOINIT_CLASSES.add(cls)
    # import pdb
    # pdb.set_trace()
    return cls


class __TDL__(object):
    ''' class that stores all parameters and methods that are created using
    tdl decorators '''
    def __init__(self, obj):
        self.obj = obj
        self.context = SimpleNamespace(
            initialized=False,   # is the model initialized
            given_attrs=None,    # set with the user provided attrs
            autoinit=SimpleNamespace(
                disable=0,
                trainable=True)
        )


def get_context(model):
    ''' get tdl context for a model.
    The context can be used to know if the model has been initialized
    '''
    # if not any(isinstance(model, AutoClass)
    #            for AutoClass in TDL_AUTOINIT_CLASSES):
    #     raise TypeError('context is not available for the object: {}'
    #                     ''.format(model))
    if '__tdl__' not in model.__dict__:
        model.__tdl__ = __TDL__(model)
    #   raise AttributeError('It seems that model {} has not been initialized '
    #                        '(missing __tdl__ attribute)'.format(model))
    return model.__tdl__.context


def is_autoinit_enabled(obj):
    '''Check if autoinitialization of parameters is enabled for
    the model obj '''
    context = get_context(obj)
    assert context.autoinit.disable >= 0,\
        'autoinit for {} is negative'.format(obj)
    return context.autoinit.disable == 0


def disable_autoinit(obj):
    get_context(obj).autoinit.disable += 1


def enable_autoinit(obj):
    context = get_context(obj)
    context.autoinit.disable -= 1
    assert context.autoinit.disable >= 0,\
        'autoinit for {} is negative'.format(obj)


class DisableAutoinit(object):
    ''' Disables autoinitialization of parameters '''
    def __init__(self, obj):
        self.obj = obj

    def __enter__(self):
        disable_autoinit(self.obj)

    def __exit__(self, type, value, traceback):
        enable_autoinit(self.obj)
