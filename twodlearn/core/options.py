""" This file defines global properties used by TDL library
"""

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
            trainable=True
        )


global_options = GlobalOptions(reuse_scope=False)


class NotTrainable(object):
    ''' Variables are instantiated as not trainable '''
    def __enter__(self):
        global_options.autoinit.trainable = False

    def __exit__(self, type, value, traceback):
        global_options.autoinit.trainable = True
