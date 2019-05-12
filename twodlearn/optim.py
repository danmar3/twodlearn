#  ***********************************************************************
#   General purpose optimizer
#
#   Wrote by: Daniel L. Marino (marinodl@vcu.edu)
#    Modern Heuristics Research Group (MHRG)
#    Virginia Commonwealth University (VCU), Richmond, VA
#    http://www.people.vcu.edu/~mmanic/
#
#   ***********************************************************************
from __future__ import division
from __future__ import print_function

import os
try:
    import queue
except ImportError:
    import Queue as queue
import shutil
import warnings
import threading
import collections
import numpy as np
from time import time
import tensorflow as tf
import twodlearn as tdl
from twodlearn import monitoring
from tqdm import tqdm
try:
    from types import SimpleNamespace
except ImportError:
    from argparse import Namespace as SimpleNamespace


class DataFeeder:
    def __init__(self, feed_train, feed_valid=None):
        self.train_feeder = feed_train

        if feed_valid is None:
            self.valid_feeder = None
        else:
            self.valid_feeder = feed_valid

    def stop(self):
        # self.train_feeder.stop()
        # if self.valid_feeder is not None:
        #     self.valid_feeder.stop()
        return

    def __del__(self):
        self.stop()

    def feed_train(self):
        return self.train_feeder()

    def feed_valid(self):
        return self.valid_feeder()


class ConstantLr(object):
    def __init__(self, value):
        self.placeholder = tf.placeholder(tf.float32)
        self.value = value

    def __call__(self, step, global_step):
        return self.value


class OptimizationManager:
    ''' Performs a standard mini-batch training with validation evaluation '''

    def _init_options(self, options):
        default = {'progress/window_size': 50,
                   'progress/reset_multiplier': 10,
                   'progress/max_trials': 20}
        options = tdl.core.check_defaults(options, default)
        return options

    def __init__(self, session, optimizer=None, step_op=None, monitor_manager=None,
                 n_logging=100, saver=None, options=None, optimizer_op=None):
        self.session = session
        self.optimizer = optimizer
        self.step_op = step_op
        if optimizer_op is not None:
            warnings.warn('optimizer_op is deprecated, specify optimizer and '
                          'step_op instead')
            self.optimizer = None
            self.step_op = optimizer_op
        self.monitor_manager = monitor_manager
        self.n_logging = n_logging
        self.n_steps = 0
        self.saver = saver
        self.options = self._init_options(options)

    def check_progress(self, step, xp):
        """Check if progress was made in the last call to the optimizer
        Args:
            step (int): current optimizer step.
            xp (list): list of outputs from the training monitors.
        Returns:
            bool: variables were reset.
        """
        if (self.monitor_manager is None) or (self.saver is None):
            return False
        if len(self.monitor_manager.train.monitors) == 1:
            monitor = self.monitor_manager.train.monitors[0]
            xp = xp[0]
        else:
            # TODO: add a way to specify which monitor is measuring performance
            # of the optimization process
            return False
        if ((self.options['progress/window_size'] < step) and
                (monitor.min is not np.inf) and
                (len(self.saver.checkpoints) > 1)):
            mean = monitor.mean(self.options['progress/window_size'])
            if (self.options['progress/reset_multiplier']*(mean - monitor.min)
                    < (xp - monitor.min)):
                print('Optimizer seems to have diverged from previous '
                      'sub-optimal region ({}). Resetting...'
                      ''.format(xp))
                self.saver.restore()
                return True
        return False

    def check_nan(self, step, xp):
        """Check if the result from the optimizer includes Nan values.
        Args:
            step (int): current step of the optimizer.
            xp (list): list of outputs from the optimizer
        Returns:
            bool: True if variables were reset.
        """
        if any([np.isnan(oi).any() for oi in xp
                if oi is not None]):
            if self.saver is None:
                raise ValueError(
                    'Optimization returned NaN at step {}.'
                    'No checkpoint saver to restore state.'.format(step))
            else:
                print('Optimization returned NaN at step {}.'
                      'Restoring last checkpoint'.format(step))
                self.saver.restore()
                return True
        return False

    def run_step(self, step, ops, feed_dict):
        """Run a step of the optimizer.
        Args:
            step (type): Description of parameter `step`.
            ops (type): Description of parameter `ops`.
            feed_dict (type): Description of parameter `feed_dict`.
        Returns:
            type: Description of returned object.
        """
        step_op, train_ops, monitor_ops = ops
        n_trials = 0
        while True:
            out = self.session.run([step_op] + train_ops + monitor_ops,
                                   feed_dict=feed_dict)
            # check number of trials
            n_trials += 1
            if n_trials > self.options['progress/max_trials']:
                return out
            # Check for NaN
            if self.check_nan(step, xp=out):
                continue
            # check for progress
            if self.check_progress(step=step, xp=out[1:1 + len(train_ops)]):
                continue
            break
        return out

    def run(self,
            n_train_steps, feed_train=None,
            n_valid_steps=1, valid_eval_freq=1, feed_valid=None,
            monitor_training=True):

        if feed_train is None:
            def feed_train(): return None
        if feed_valid is None:
            def feed_valid(): return None
        data_feeder = DataFeeder(feed_train, feed_valid)

        if self.monitor_manager:
            train_monitors = self.monitor_manager.train.tf_monitors
            train_ops = [m.op for m in train_monitors]
            valid_monitors = self.monitor_manager.valid.tf_monitors
            valid_ops = [m.op for m in valid_monitors]
        else:
            train_monitors = []
            train_ops = []
            valid_monitors = []
            valid_ops = []

        if monitor_training and self.monitor_manager:
            monitor_monitors = self.monitor_manager.monitoring.tf_monitors
            monitor_ops = [m.op for m in monitor_monitors]
        else:
            monitor_monitors = []
            monitor_ops = []

        # safer function
        if self.saver is not None:
            self.saver.reset()

        # run optimizer
        try:
            for step in range(1, n_train_steps):
                # Run optimization step
                out = self.run_step(
                    step=step,
                    ops=(self.step_op, train_ops, monitor_ops),
                    feed_dict=data_feeder.feed_train())
                self.n_steps += 1

                # feed data to monitors
                if train_ops:
                    train_output = out[1:1 + len(train_ops)]
                    for i, monitor in enumerate(train_monitors):
                        monitor.feed(train_output[i], self.n_steps)
                if monitor_ops:
                    monitor_output = out[1 + len(train_ops):]
                    for i, monitor in enumerate(monitor_monitors):
                        monitor.feed(monitor_output[i], self.n_steps)

                # file loggers
                self.monitor_manager.train.write_data()
                self.monitor_manager.monitoring.write_data()

                # run validation evaluation
                if valid_ops and (step % valid_eval_freq == 0):
                    for step_valid in range(0, n_valid_steps):
                        valid_output = self.session.run(
                            valid_ops,
                            feed_dict=data_feeder.feed_valid())
                        for i, monitor in enumerate(valid_monitors):
                            monitor.feed(valid_output[i], self.n_steps)
                    # file loggers
                    self.monitor_manager.valid.write_data()
                # saver function
                if (self.saver is not None):
                    self.saver.add_checkpoint(step)

                # log
                if (step % self.n_logging == 0) and self.monitor_manager:
                    # print information
                    train_info = [(m.name, m.mean()) for m in train_monitors]
                    valid_info = [(m.name, m.mean()) for m in valid_monitors]
                    # log information in files
                    # self.monitor_manager.train.write_stats()
                    # self.monitor_manager.valid.write_stats()
                    # self.monitor_manager.monitoring.write_stats()
                    print("{} | {} | {}".format(step, train_info, valid_info))
        finally:
            # clean up
            data_feeder.stop()
            self.monitor_manager.flush()
            if self.saver is not None:
                self.saver.restore()
                self.saver.save()


class Optimizer(tdl.core.TdlModel):
    _submodels = ['learning_rate', 'monitor_manager', 'optimizer', 'saver']

    def _init_options(self, options):
        default = {'progress/window_size': 50,
                   'progress/reset_multiplier': 10,
                   'progress/max_trials': 20}
        options = tdl.core.check_defaults(options, default)
        return options

    @tdl.core.InputArgument
    def session(self, value):
        return (value if value is not None
                else tf.get_default_session()
                if tf.get_default_session() is not None
                else tf.InteractiveSession())

    @tdl.core.InputArgument
    def log_folder(self, value):
        if value is None:
            if tdl.core.is_property_set(self, 'monitor_manager'):
                value = self.monitor_manager.log_folder
            else:
                value = 'tmp/monitors/'
        return value

    def _monitor_from_dict(self, value):
        train = (value if 'train' not in value
                 else value['train'] if isinstance(value['train'], dict)
                 else {'train': value['train']})
        valid = (None if 'valid' not in value
                 else value['valid'] if isinstance(value['value'], dict)
                 else {'valid': value['valid']})
        monitor = (None if 'monitoring' not in value
                   else value['monitoring']
                   if isinstance(value['monitoring'], dict)
                   else {'monitoring': value['monitoring']})
        return monitoring.SimpleTrainingMonitor(
            train_vars=train, valid_vars=valid, monitoring_vars=monitor,
            log_folder=self.log_folder)

    @tdl.core.Submodel
    def monitor_manager(self, value):
        tdl.core.assert_initialized_if_available(
            self, 'monitor_manager', ['log_folder'])
        if value is None:
            value = {'train': {'train/loss': self.loss}}
        monitor_manager = (self._monitor_from_dict(value)
                           if isinstance(value, dict)
                           else value)
        loss_monitor = filter(
            lambda monitor: (tf.convert_to_tensor(monitor.op) ==
                             tf.convert_to_tensor(self.loss)),
            monitor_manager.train.monitors)
        if not loss_monitor:
            monitor_manager.train.add_monitor(
                monitoring.OpMonitor(self.loss, name=self.loss.name))
        return monitor_manager

    @tdl.core.LazzyProperty
    def loss_monitor(self):
        return list(filter(lambda monitor: monitor.op == self.loss,
                           self.monitor_manager.train.monitors))[0]

    @tdl.core.Submodel
    def learning_rate(self, value):
        if value is None:
            return ConstantLr(0.02)
        else:
            return value

    @tdl.core.Submodel
    def optimizer(self, value):
        if value is None:
            Optimizer = tf.train.AdamOptimizer
        elif callable(value):
            Optimizer = value
        else:
            return value
        if hasattr(self.learning_rate, 'placeholder'):
            optimizer = Optimizer(learning_rate=self.learning_rate.placeholder)
        else:
            optimizer = Optimizer(learning_rate=self.learning_rate)
        return optimizer

    @tdl.core.Submodel
    def step_op(self, _):
        step_op = self.optimizer.minimize(tf.convert_to_tensor(self.loss),
                                          var_list=self.var_list)
        self.reset()
        return step_op

    @property
    def var_optim(self):
        '''Variables created by the optimizer'''
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=self.scope)
        for var in self.var_list:
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                         scope=var.name.split(':')[0])
            vars += [vi for vi in var_list
                     if vi is not var]
        return vars

    def reset(self):
        '''Reset optimizer variables (var_optim)'''
        list(map(lambda x: x.initializer.run(), self.var_optim))

    @tdl.core.Submodel
    def saver(self, value):
        tdl.core.assert_initialized(self, 'saver', ['monitor_manager'])
        if value != 'default':
            return value
        if self.monitor_manager is None:
            return None
        monitor = (self.monitor_manager.valid.monitors[0]
                   if self.monitor_manager.valid.monitors
                   else self.monitor_manager.train.monitors[0])
        return EarlyStopping(
            monitor=monitor,
            var_list=self.var_list,
            logger_path=self.monitor_manager.log_folder,
            session=self.session)

    def __init__(self, loss, var_list, session=None, metrics=None,
                 n_logging=100, log_folder=None, options=None,
                 **kargs):
        self.loss = tf.convert_to_tensor(loss)
        self.var_list = (var_list if var_list is not None
                         else tf.trainable_variables())
        self.n_logging = n_logging
        self.n_steps = 0
        if 'saver' not in kargs:
            kargs['saver'] = 'default'
        if metrics is not None and 'monitor_manager' in kargs:
            raise ValueError('cannot specify metrics and monitor_manager '
                             'at the same time')
        metrics = (kargs['monitor_manager'] if 'monitor_manager' in kargs
                   else metrics)
        kargs = {key: val for key, val in kargs.items()
                 if key is not 'monitor_manager'}
        if log_folder is not None:
            kargs['log_folder'] = log_folder
        super(Optimizer, self).__init__(session=session,
                                        monitor_manager=metrics,
                                        options=options, **kargs)

    def feed_train(self):
        return dict()

    def check_progress(self, step, xp):
        """Check if progress was made in the last call to the optimizer
        Args:
            step (int): current optimizer step.
            xp (list): list of outputs from the training monitors.
        Returns:
            bool: variables were reset.
        """
        if (self.monitor_manager is None) or (self.saver is None):
            return False
        monitor = self.loss_monitor
        xp = xp[self.loss]
        if ((self.options['progress/window_size'] < step) and
                (monitor.min is not np.inf) and self.saver.checkpoints):
            mean = monitor.mean(self.options['progress/window_size'])
            if (self.options['progress/reset_multiplier']*(mean - monitor.min)
                    < (xp - monitor.min)):
                print('Optimizer seems to have diverged from previous '
                      'sub-optimal region ({}). Resetting...'
                      ''.format(xp))
                self.saver.restore()
                return True
        return False

    def check_nan(self, step, xp):
        """Check if the result from the optimizer includes Nan values.
        Args:
            step (int): current step of the optimizer.
            xp (list): list of outputs from the optimizer
        Returns:
            bool: True if variables were reset.
        """
        if any([np.isnan(oi).any() for oi in xp
                if oi is not None]):
            if self.saver is None:
                print('Optimization returned NaN at step {}.'
                      'Re-initializing variables'.format(step))
                self.session.run([v.initializer for v in self.var_list])
            else:
                print('Optimization returned NaN at step {}.'
                      'Restoring last checkpoint'.format(step))
                if self.saver.checkpoints:
                    self.saver.restore()
                else:
                    self.session.run([v.initializer for v in self.var_list])
                return True
        return False

    def run_step(self, step, ops, feed_dict):
        """Run a step of the optimizer.
        Args:
            step (type): Description of parameter `step`.
            ops (type): Description of parameter `ops`.
            feed_dict (type): Description of parameter `feed_dict`.
        Returns:
            type: Description of returned object.
        """
        if isinstance(self.learning_rate, ConstantLr):
            feed_dict[self.learning_rate.placeholder] = \
                self.learning_rate(step, self.n_steps)
        n_trials = 0
        while True:
            output = self.session.run(ops, feed_dict=feed_dict)
            output = {op: output[idx] for idx, op in enumerate(ops)}

            # check number of trials
            n_trials += 1
            if n_trials > self.options['progress/max_trials']:
                self.session.run([v.initializer for v in self.var_list])
                output = self.session.run(ops, feed_dict=feed_dict)
                output = {op: output[idx] for idx, op in enumerate(ops)}
                return output
            # Check for NaN
            if self.check_nan(step, xp=output.values()):
                continue
            # check for progress
            if self.check_progress(step=step, xp=output):
                continue
            break
        return output

    def run(self,
            n_train_steps, feed_train=None,
            n_valid_steps=1, valid_eval_freq=1, feed_valid=None,
            monitor_training=True):

        if feed_train is None:
            def feed_train(): return dict()
        if feed_valid is None:
            def feed_valid(): return dict()
        data_feeder = DataFeeder(feed_train, feed_valid)

        if self.monitor_manager:
            train_monitors = self.monitor_manager.train.tf_monitors
            train_ops = [m.op for m in train_monitors]
            valid_monitors = self.monitor_manager.valid.tf_monitors
            valid_ops = [m.op for m in valid_monitors]
        else:
            train_monitors = []
            train_ops = []
            valid_monitors = []
            valid_ops = []

        if monitor_training and self.monitor_manager:
            monitor_monitors = self.monitor_manager.monitoring.tf_monitors
            monitor_ops = [m.op for m in monitor_monitors]
        else:
            monitor_monitors = []
            monitor_ops = []

        # saver function
        if self.saver is not None:
            self.saver.reset()

        # run optimizer
        try:
            for step in tqdm(range(1, n_train_steps)):
                # Run optimization step
                xp = self.run_step(
                    step=step,
                    ops=[self.step_op] + train_ops + monitor_ops,
                    feed_dict=data_feeder.feed_train())
                self.n_steps += 1

                # feed data to monitors
                if train_ops:
                    for i, monitor in enumerate(train_monitors):
                        monitor.feed(xp[monitor.op], self.n_steps)
                if monitor_ops:
                    for i, monitor in enumerate(monitor_monitors):
                        monitor.feed(xp[monitor.op], self.n_steps)

                # file loggers
                self.monitor_manager.train.write_data()
                self.monitor_manager.monitoring.write_data()

                # run validation evaluation
                if valid_ops and (step % valid_eval_freq == 0):
                    for step_valid in range(0, n_valid_steps):
                        valid_output = self.session.run(
                            valid_ops,
                            feed_dict=data_feeder.feed_valid())
                        for i, monitor in enumerate(valid_monitors):
                            monitor.feed(valid_output[i], self.n_steps)
                    # file loggers
                    self.monitor_manager.valid.write_data()
                # saver function
                if (self.saver is not None):
                    self.saver.add_checkpoint(step)

                # log
                if (step % self.n_logging == 0) and self.monitor_manager:
                    # print information
                    train_info = [(m.name, m.mean()) for m in train_monitors]
                    valid_info = [(m.name, m.mean()) for m in valid_monitors]
                    # log information in files
                    print("{} | {} | {}".format(step, train_info, valid_info))
        finally:
            # clean up
            data_feeder.stop()
            self.monitor_manager.flush()
            if self.saver is not None:
                if self.saver.checkpoints:
                    self.saver.restore()
                self.saver.save()


class SimpleSaver(tdl.core.TdlObject):
    @property
    def checkpoints(self):
        return None

    def __init__(self, var_list, logger_path, session):
        self.session = session
        self.var_list = var_list
        super(SimpleSaver, self).__init__(save={'logger_path': logger_path})

    @tdl.core.EncapsulatedMethod
    def save(self, locals, value):
        self._saver = tf.train.Saver(var_list=self.var_list)
        self._saver_id = 0
        self._logger_path = os.path.join(value['logger_path'], 'optimizer')
        if os.path.exists(self._logger_path):
            shutil.rmtree(self._logger_path)
        os.makedirs(self._logger_path)

    @save.eval
    def save(self, locals):
        print('saving weights in {}'.format(self._logger_path))
        saver_path = os.path.join(self._logger_path, 'var_checkpoint')
        self._saver.save(
            sess=self.session,
            save_path=saver_path,
            global_step=self._saver_id)
        self._saver_id += 1

    def add_checkpoint(self, step):
        return

    def reset(self):
        return

    def restore_file(self):
        saver_path = os.path.join(self._logger_path,
                                  'var_checkpoint-{}'.format(self._saver_id-1))
        self._saver.restore(self.session, saver_path)


class EarlyStoppingV2(tdl.core.TdlObject):
    @property
    def checkpoints(self):
        return self._checkpoints

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def session(self):
        return self.optimizer.session

    @tdl.core.Submodel
    def objective(self, value):
        if isinstance(value, monitoring.TrainingMonitor):
            return value
        else:
            valid_match = filter(lambda x: x.op == value,
                                 self.optimizer.monitor_manager.valid.monitors)
            if valid_match:
                return valid_match[0]
            else:
                raise ValueError('{} not found in set of valid monitors.'
                                 ''.format(value))

    @tdl.core.EncapsulatedMethod
    def restore(self, local, value):
        local.placeholders = {var: tf.placeholder(tf.float32)
                              for var in self.optimizer.var_list}
        assign_vars = [var.assign(local.placeholders[var])
                       for var in self.optimizer.var_list]
        local.assign_vars = tf.group(assign_vars)

    @restore.eval
    def restore(self, local):
        ckpt = self.checkpoints[-1]
        feed_dict = {local.placeholders[var]: ckpt[var]
                     for var in self.optimizer.var_list}
        self.session.run(local.assign_vars, feed_dict=feed_dict)

    def reset(self):
        self.check_progress.local.best_value = np.nan

    def __init__(self, optimizer, objective, minimize=True):
        self._optimizer = optimizer
        super(type(self), self).__init__(objective=objective)

    def __bool__(self):
        return len(self.checkpoints) > 0


class EarlyStopping(tdl.core.TdlObject):
    @property
    def checkpoints(self):
        return self._ckpts

    def _init_options(self, options):
        default = {'start_steps': 300,
                   'ckpts_dt': 5.0,
                   'window_size': 50}
        options = tdl.core.check_defaults(options, default)
        return options

    def __init__(self, monitor, var_list, logger_path,
                 session, check_func=None, options=None):
        self.monitor = monitor
        self.session = session
        self.var_list = var_list
        self.options = self._init_options(options)
        super(EarlyStopping, self).__init__(save={'logger_path': logger_path})
        if check_func is not None:
            raise NotImplementedError('Custom check_func not yet implemented.'
                                      'Use None for the moment.')
        self.check_func = (check_func if check_func is not None
                           else self.check_lower)

    def check_progress(self, step, monitor):
        if ((self.options['window_size'] < step) and
                (monitor.min is not np.inf) and
                (len(self._ckpts) > 1)):
            mean = monitor.mean(self.options['window_size'])
            current_value = monitor.current_value
            if 10*(mean - monitor.min) < (current_value - monitor.min):
                # pdb.set_trace()
                print('Optimizer seems to have diverged from previous '
                      'sub-optimal region ({}). Resetting...'
                      ''.format(current_value))
                self.restore()

    @tdl.core.EncapsulatedMethod
    def check_lower(self, local, value):
        local.time_last_ckpt = time()
        local.best_value = np.nan

    @check_lower.eval
    def check_lower(self, local, step):
        current_value = self.monitor.mean(self.options['window_size'])
        save = (True if local.best_value is np.nan
                else current_value < local.best_value)
        save = ((time() - local.time_last_ckpt) > self.options['ckpts_dt']
                and (step > self.options['start_steps'])
                and save)
        if save:
            print(np.abs(local.best_value - current_value) /
                  (self.monitor.max - self.monitor.min))
            local.time_last_ckpt = time()
            local.best_value = current_value
        return save

    def check_greather(self, local):
        current_value = self.monitor.mean(self.options['window_size'])
        save = (True if local.best_value is np.nan
                else local.best_value < current_value)
        save = ((time() - local.time_last_ckpt) > self.options['ckpts_dt']
                and save)
        if save:
            print(np.abs(local.best_value - current_value) /
                  (self.monitor.max - self.monitor.min))
            local.time_last_ckpt = time()
            local.best_value = current_value
        return save

    @tdl.core.EncapsulatedMethod
    def add_checkpoint(self, local, value):
        self._ckpts = collections.deque(maxlen=10)

    @add_checkpoint.eval
    def add_checkpoint(self, local, step):
        if self.check_func(step):
            print('checkpoint created')
            values = self.session.run(self.var_list)
            vars = {var: value for var, value in zip(self.var_list, values)}
            self._ckpts.append(vars)

    @tdl.core.EncapsulatedMethod
    def restore(self, local, value):
        local.placeholders = {var: tf.placeholder(tf.float32)
                              for var in self.var_list}
        set_vars = [var.assign(local.placeholders[var])
                    for var in self.var_list]
        local.set_vars = tf.group(set_vars)

    @restore.eval
    def restore(self, local):
        ckpt = self._ckpts[-1]
        feed_dict = {local.placeholders[var]: ckpt[var]
                     for var in self.var_list}
        self.session.run(local.set_vars, feed_dict=feed_dict)

    def reset(self):
        self.check_lower.local.best_value = np.nan

    @tdl.core.EncapsulatedMethod
    def save(self, locals, value):
        self._saver = tf.train.Saver(var_list=self.var_list)
        self._saver_id = 0
        self._logger_path = os.path.join(value['logger_path'], 'optimizer')
        if os.path.exists(self._logger_path):
            shutil.rmtree(self._logger_path)
        os.makedirs(self._logger_path)
        self._save_time = time()

    @save.eval
    def save(self, locals):
        print('saving weights in {}'.format(self._logger_path))
        saver_path = os.path.join(self._logger_path, 'var_checkpoint')
        self._saver.save(
            sess=self.session,
            save_path=saver_path,
            global_step=self._saver_id)
        self._saver_id += 1

    def restore_file(self):
        saver_path = os.path.join(self._logger_path,
                                  'var_checkpoint-{}'.format(self._saver_id-1))
        self._saver.restore(self.session, saver_path)
