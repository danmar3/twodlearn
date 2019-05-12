#  ***********************************************************************
#   This file defines global properties used by TDL library
#
#   Wrote by: Daniel L. Marino (marinodl@vcu.edu)
#    Modern Heuristics Research Group (MHRG)
#    Virginia Commonwealth University (VCU), Richmond, VA
#    http://www.people.vcu.edu/~mmanic/
#
#   ***********************************************************************
import numpy as np
import collections
try:
    from types import SimpleNamespace as PySimpleNamespace
except ImportError:
    from argparse import Namespace as PySimpleNamespace


class GlobalOptions:
    def __init__(self, reuse_scope=False,
                 float_nptype=np.float32,
                 float_tftype=np.float32):
        self.reuse_scope = reuse_scope
        self.float = PySimpleNamespace(nptype=float_nptype,
                                       tftype=float_tftype)
        self.tolerance = 1e-6
        self.autoinit = PySimpleNamespace(
            _disable_autoinit=collections.defaultdict(lambda: 0),
            trainable=True
        )

    def is_autoinit_enabled(self, obj):
        ''' check if autoinitialization of parameters is enabled for
        the model obj '''
        assert self.autoinit._disable_autoinit[obj] >= 0,\
            'autoinit for {} is negative'.format(obj)
        return self.autoinit._disable_autoinit[obj] == 0

    def disable_autoinit(self, obj):
        self.autoinit._disable_autoinit[obj] += 1

    def enable_autoinit(self, obj):
        self.autoinit._disable_autoinit[obj] -= 1
        assert self.autoinit._disable_autoinit[obj] >= 0,\
            'autoinit for {} is negative'.format(obj)


global_options = GlobalOptions(reuse_scope=False)


class DisableAutoinit(object):
    ''' Disables autoinitialization of parameters '''
    def __init__(self, obj):
        self.obj = obj

    def __enter__(self):
        global_options.disable_autoinit(self.obj)

    def __exit__(self, type, value, traceback):
        global_options.enable_autoinit(self.obj)


class NotTrainable(object):
    ''' Variables are instantiated as not trainable '''
    def __enter__(self):
        global_options.autoinit.trainable = False

    def __exit__(self, type, value, traceback):
        global_options.autoinit.trainable = True
