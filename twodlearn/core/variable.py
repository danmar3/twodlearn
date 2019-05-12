import tensorflow as tf
from .options import global_options
# from . import common

if [int(s) for s in tf.__version__.split('.')][0:2] < [1, 12]:
    tf_Variable = tf.Variable
else:
    tf_Variable = object


def variable(initial_value=None,
             trainable=None,
             collections=None,
             validate_shape=True,
             caching_device=None,
             name=None,
             variable_def=None,
             dtype=None,
             expected_shape=None,
             import_scope=None,
             constraint=None,
             **kargs):
    if dtype is None:
        if isinstance(initial_value, (int, float)):
            dtype = global_options.float.tftype
    if trainable is None:
        trainable = global_options.autoinit.trainable

    return tf.Variable(
        initial_value=initial_value,
        trainable=trainable,
        collections=collections,
        validate_shape=validate_shape,
        caching_device=caching_device,
        name=name,
        variable_def=variable_def,
        dtype=dtype,
        expected_shape=expected_shape,
        import_scope=import_scope,
        constraint=constraint,
        **kargs)
