import sys
import pdb
import logging
import warnings
import traceback
try:
    from types import SimpleNamespace
except ImportError:
    from argparse import Namespace as SimpleNamespace


class TracePrintsSysOut(object):
    def __init__(self):
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write("Writing %r\n" % s)
        traceback.print_stack(file=self.stdout)


class TracePrints:
    ''' class to trace print statements
    Usage:
    with TracePrints:
        run_your_function()
    '''

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = TracePrintsSysOut()

    def __exit__(self, type, value, traceback):
        # restore old stdout
        sys.stdout = self._stdout


def debug_func(func):
    try:
        return func()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


def stop_at_error(func):
    ''' Define decorator to stop at error '''
    def debug_func(*args):
        try:
            return func(*args)
        except:
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
    return debug_func


def stop_at_warning(func):
    ''' Define decorator to stop at warnings '''
    def debug_func(*args):
        warnings.simplefilter('error')
        try:
            return func(*args)
        except:
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        warnings.simplefilter('default')
    return debug_func


debug_loggers = dict()


def get_logger(name):
    def new_debug_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')

        # Create file handler that logs all activity
        fh = logging.FileHandler('{}.log'.format(name), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    if name not in debug_loggers:
        debug_loggers[name] = new_debug_logger(name)
    return debug_loggers[name]
