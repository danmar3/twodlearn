import os
import shutil
import collections
from time import time
import tensorflow as tf
import twodlearn as tdl
from twodlearn import monitoring
from tqdm import tqdm
import numpy as np


class ConstantLr(object):
    def __init__(self, value):
        self.placeholder = tf.placeholder(tf.float32)
        self.value = value

    def __call__(self, step, global_step):
        return self.value


@tdl.core.create_init_docstring
class BaseOptimizer(tdl.core.TdlModel):
    @classmethod
    def wrap(cls, other):
        ''' BaseOptimizer.wrap(Monitor).wrap(Logger)'''
        def run_step(self, **kwargs):
            return other.step_fn(self, wrapped_fn=cls.run_step, **kwargs)

        def get_ops(self):
            value = dict()
            if hasattr(cls, 'get_ops'):
                value.update(cls.get_ops(self))
            if hasattr(other, 'get_ops'):
                value.update(other.get_ops(self))
            return value

        def warmup(self, **kwargs):
            value = dict()
            if hasattr(cls, 'warmup'):
                cls.warmup(self, **kwargs)
            if hasattr(other, 'warmup'):
                other.warmup(self, **kwargs)
            return value

        def cleanup(self, **kwargs):
            value = dict()
            if hasattr(cls, 'cleanup'):
                cls.cleanup(self, **kwargs)
            if hasattr(other, 'cleanup'):
                other.cleanup(self, **kwargs)
            return value

        new_cls = type("Optimizer", (cls, other),
                       {'run_step': run_step,
                        'get_ops': get_ops,
                        'warmup': warmup,
                        'cleanup': cleanup})
        return tdl.core.create_init_docstring(new_cls)

    def get_ops(self):
        return {self.loss.name: self.loss,
                self.step_op.name: self.step_op,
                self.train_step.name: self.train_step}

    def run_step(self, **kwargs):
        return self.step_fn(**kwargs)

    @tdl.core.Submodel
    def feed_dict(self, _):
        def feed_fn():
            if callable(self.feed_dict.value):
                return self.feed_dict.value()
            else:
                return self.feed_dict.value
        return tdl.core.SimpleNamespace(value=None, fn=feed_fn)

    def step_fn(self, wrapped_fn=None, **kwargs):
        assert wrapped_fn is None,\
            'BaseOptimizer should be the first optimizer called'
        data = self.session.run(
            self.get_ops(),
            feed_dict=self.feed_dict.fn())
        return data

    @tdl.core.InputArgument
    def session(self, value):
        return (value if value is not None
                else tf.get_default_session()
                if tf.get_default_session() is not None
                else tf.InteractiveSession())

    @tdl.core.InputArgument
    def loss(self, value):
        if value is None:
            raise tdl.core.exceptions.ArgumentNotProvided(self, 'kernel_size')
        return tf.convert_to_tensor(value)

    @tdl.core.InputArgument
    def var_list(self, value):
        return value

    @tdl.core.Submodel
    def learning_rate(self, value):
        if value is None:
            value = 0.02
        return value

    @tdl.core.Submodel
    def optimizer(self, value):
        '''Optimizer used to perform the optimization. AdamOptimizer is used
           by default.
        '''
        if value is None:
            Optimizer = tf.train.AdamOptimizer
        elif callable(value):
            Optimizer = value
        else:
            return value
        optimizer = Optimizer(learning_rate=self.learning_rate)
        return optimizer

    @tdl.core.LazzyProperty
    def train_step(self):
        '''number of training steps performed.'''
        return tf.Variable(0, dtype=tf.int32, name='train_step')

    @tdl.core.Submodel
    def step_op(self, _):
        '''operation that performs the weight update.'''
        tdl.core.assert_initialized(self, 'step_op', ['loss', 'optimizer'])
        with tf.control_dependencies([self.train_step.assign_add(1)]):
            step_op = self.optimizer.minimize(
                tf.convert_to_tensor(self.loss),
                var_list=self.var_list)
        return step_op

    @tdl.core.MethodInit
    def restart(self, local):
        '''calls the initializer for all var_list and optimizer variables'''
        tdl.core.assert_initialized(
            self, 'restart', ['step_op', 'optimizer', 'train_step'])
        local.vars = set(self.optimizer.variables() + self.var_list +
                         [self.train_step])
        local.restart_op = tdl.core.variables_initializer(local.vars)

    @restart.eval
    def restart(self, local):
        '''calls the initializer for all var_list and optimizer variables'''
        self.session.run(local.restart_op)

    @tdl.core.MethodInit
    def _assert_initialized(self, local):
        '''assert variables have been initialized'''
        local.vars = set(self.optimizer.variables() + self.var_list +
                         [self.train_step])
        local.initialized = False

    @_assert_initialized.eval
    def _assert_initialized(self, local):
        if not local.initialized:
            print('initializing')
            tdl.core.initialize_variables(list(local.vars))
            local.initialized = True
        return

    def warmup(self, **kwargs):
        if (kwargs['feed_dict'] is not None) and ('feed_train' in kwargs):
            raise ValueError('cannot specify feed_dict and feed_train at the '
                             'same time')
        feed_dict = (kwargs['feed_train'] if 'feed_train' in kwargs
                     else kwargs['feed_dict'])
        if feed_dict is not None:
            self.feed_dict.value = feed_dict

    def cleanup(self, **kwargs):
        self.feed_dict.value = None

    def run(self, n_steps, feed_dict=None, **kargs):
        tdl.core.assert_initialized(
            self, 'run', ['step_op', 'restart', '_assert_initialized'])
        self._assert_initialized()
        self.warmup(n_steps=n_steps, feed_dict=feed_dict, **kargs)
        for step in range(1, n_steps):
            self.run_step(**kargs)
        self.cleanup(n_steps=n_steps, feed_dict=feed_dict, **kargs)


@tdl.core.create_init_docstring
class Monitor(tdl.core.TdlModel):
    @tdl.core.InputArgument
    def log_folder(self, value):
        if value is None:
            if tdl.core.is_property_set(self, 'monitor_manager'):
                value = self.monitor_manager.log_folder
            else:
                value = 'tdl_tmp/'
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

    @tdl.core.SubmodelInit
    def monitor_manager(self, metrics=None, valid_freq=10):
        tdl.core.assert_initialized_if_available(
            self, 'monitor_manager', ['log_folder'])
        if metrics is None:
            metrics = {'train': {'train/loss': self.loss}}
        monitor_manager = (self._monitor_from_dict(metrics)
                           if isinstance(metrics, dict)
                           else metrics)
        loss_monitor = filter(
            lambda monitor: (tf.convert_to_tensor(monitor.op) ==
                             tf.convert_to_tensor(self.loss)),
            monitor_manager.train.monitors)
        if not loss_monitor:
            monitor_manager.train.add_monitor(
                monitoring.OpMonitor(self.loss, name=self.loss.name))
        self._valid_freq = valid_freq
        return monitor_manager

    @tdl.core.LazzyProperty
    def loss_monitor(self):
        return list(filter(lambda monitor: monitor.op == self.loss,
                           self.monitor_manager.train.monitors))[0]

    def get_ops(self):
        train_monitors = self.monitor_manager.train.tf_monitors
        ops = {m.op.name: m.op for m in train_monitors}
        return ops

    @tdl.core.Submodel
    def feed_valid(self, _):
        def feed_fn():
            if callable(self.feed_valid.value):
                return self.feed_valid.value()
            else:
                return self.feed_valid.value
        return tdl.core.SimpleNamespace(value=None, fn=feed_fn)

    def step_fn(self, wrapped_fn=None, **kwargs):
        def feed_monitors(data, monitors, n_steps):
            for monitor in monitors:
                monitor.feed(data[monitor.op.name], n_steps)
        # run session
        data = wrapped_fn(self, **kwargs)
        n_steps = data[self.train_step.name]
        # feed data to monitors
        feed_monitors(data, self.monitor_manager.train.tf_monitors, n_steps)
        self.monitor_manager.train.write_data()      # file loggers
        # validation eval
        valid_monitors = self.monitor_manager.valid.tf_monitors
        if valid_monitors and (n_steps % self._valid_freq == 0):
            valid_ops = {m.op.name: m.op for m in valid_monitors}
            valid_data = self.session.run(
                valid_ops, feed_dict=self.feed_valid.fn())
            feed_monitors(valid_data, valid_monitors, n_steps)
            self.monitor_manager.valid.write_data()  # file loggers
            # merge data with valid_data
            data = {**data, **valid_data}
        return data

    def warmup(self, **kwargs):
        tdl.core.assert_initialized(self, 'warmup', ['monitor_manager'])
        if kwargs['feed_valid'] is not None:
            self.feed_valid.value = kwargs['feed_valid']

    def cleanup(self, **kwargs):
        self.feed_valid.value = None
        self.monitor_manager.flush()


@tdl.core.create_init_docstring
class StatusBar(tdl.core.TdlModel):
    @tdl.core.SubmodelInit
    def status_bar(self, update_freq=10):
        return tdl.core.SimpleNamespace(
            bar=None, update_freq=update_freq)

    def warmup(self, **kwargs):
        tdl.core.assert_initialized(self, 'warmup', ['status_bar'])
        self.status_bar.bar = tqdm(range(1, kwargs['n_steps']))

    def step_fn(self, wrapped_fn=None, **kwargs):
        data = wrapped_fn(self, **kwargs)
        if data[self.train_step.name] % self.status_bar.update_freq == 0:
            self.status_bar.bar.update(self.status_bar.update_freq)
            info = 'step {} | loss {:.4f} '.format(
                data[self.train_step.name],
                data[self.loss.name])
            if isinstance(self, Monitor):
                valid_monitors = self.monitor_manager.valid.tf_monitors
                for monitor in valid_monitors:
                    if monitor.op.name in data:
                        info += '| {} {:.4f}'.format(
                            monitor.name, monitor.mean())
            self.status_bar.bar.set_description(info)
        return data

    def cleanup(self, **kwargs):
        self.status_bar.bar.close()


@tdl.core.create_init_docstring
class Checkpointable(tdl.core.TdlModel):
    @tdl.core.SubmodelInit
    def checkpoints(self, buffer_size=10, update_dt=5.0):
        tdl.core.assert_initialized(self, 'checkpoints', ['var_list'])

        def restore(idx=-1):
            ckpt = self.checkpoints.buffer[-1]
            feed_dict = {self.checkpoints._placeholders[var]: ckpt[var]
                         for var in self.var_list}
            self.session.run(self.checkpoints._set_vars, feed_dict=feed_dict)

        def save():
            if isinstance(self, StatusBar):
                self.status_bar.bar.set_description('checkpoint added')
            values = self.session.run(self.var_list)
            vars = {var: value for var, value in zip(self.var_list, values)}
            self.checkpoints.buffer.append(vars)
            self.checkpoints.last_updated = time()

        placeholders = {
            var: tf.placeholder(var.dtype)
            for var in self.var_list}
        set_vars = tf.group([var.assign(placeholders[var])
                             for var in self.var_list])
        return tdl.core.SimpleNamespace(
            buffer=collections.deque(maxlen=buffer_size),
            update_dt=update_dt,
            restore=restore,
            save=save,
            last_updated=time(),
            _placeholders=placeholders,
            _set_vars=set_vars)

    def warmup(self, **kwargs):
        tdl.core.assert_initialized(self, 'warmup', ['checkpoints'])

    def step_fn(self, wrapped_fn=None, **kwargs):
        data = wrapped_fn(self, **kwargs)
        if ((time() - self.checkpoints.last_updated) >
                self.checkpoints.update_dt):
            self.checkpoints.save()
        return data


@tdl.core.create_init_docstring
class CheckProgress(tdl.core.TdlModel):
    '''reset to last checkpoint if progress deteriorates.'''

    @tdl.core.SubmodelInit
    def progress(self, filter_window=50, reset_multiplier=10.0):
        '''checks if progress has been made in last step.
        filter_window: size of the buffer used by the moving average filter.
        reset_multiplier: if current loss > reset_multiplier*filtered_loss,
            we restore the last checkpoint.
        '''
        def current_value():
            try:
                current_value = (sum(self.progress.buffer) /
                                 len(self.progress.buffer))
            except ZeroDivisionError:
                current_value = None
            return current_value

        def reset():
            self.progress.buffer.clear()
            self.progress.best_value = np.inf

        def append(value):
            self.progress.buffer.append(value)
            if value < self.progress.best_value:
                self.progress.best_value = value

        def check(value):
            if (len(self.checkpoints.buffer) == 0
                    or len(self.progress.buffer) < 2):
                return True
            current_dx = value - self.progress.best_value
            filtered_dx = max([(bi - self.progress.best_value)
                               for bi in self.progress.buffer])
            return current_dx < self.progress.reset_multiplier*filtered_dx

        assert isinstance(self, Checkpointable),\
            'CheckProgress requires a Checkpointable optimizer'
        if filter_window < 10:
            raise ValueError('filter_window should be larger than 2')
        return tdl.core.SimpleNamespace(
            buffer=collections.deque(maxlen=filter_window),
            reset_multiplier=reset_multiplier,
            best_value=np.inf,
            value=current_value,
            reset=reset,
            check=check,
            append=append,
            target=tf.convert_to_tensor(self.loss))

    def warmup(self, **kwargs):
        tdl.core.assert_initialized(
            self, 'warmup', ['checkpoints', 'progress'])

    def step_fn(self, wrapped_fn=None, **kwargs):
        data = wrapped_fn(self, **kwargs)
        target = data[self.progress.target.name]
        # check progress
        while not self.progress.check(target):
            if isinstance(self, StatusBar):
                self.status_bar.bar.set_description(
                    'optimization diverged... restoring checkpoint')
            self.checkpoints.restore()
            data = wrapped_fn(self, **kwargs)
            target = data[self.progress.target.name]
            self.progress.reset()
        # add loss to the filter buffer
        self.progress.append(target)
        return data


class CheckpointableProgress(Checkpointable):
    '''stores a checkpoint in internal buffer at a given frequency only if
    progress has been made.'''
    @tdl.core.SubmodelInit
    def filtered_target(self, window_size=30):
        def reset():
            self.filtered_target.buffer = collections.deque(maxlen=window_size)
            self.filtered_target.best = None

        return tdl.core.SimpleNamespace(
                buffer=collections.deque(maxlen=window_size),
                best=None,
                target=tf.convert_to_tensor(self.loss),
                reset=reset)

    def warmup(self, **kwargs):
        tdl.core.assert_initialized(
            self, 'warmup', ['checkpoints', 'filtered_target'])

    def step_fn(self, wrapped_fn=None, **kwargs):
        data = wrapped_fn(self, **kwargs)
        if self.filtered_target.target.name in data:
            target_value = data[self.filtered_target.target.name]
            target_buffer = self.filtered_target.buffer
            target_buffer.append(target_value)
            time_check = ((time() - self.checkpoints.last_updated) >
                          self.checkpoints.update_dt)
            if time_check:
                best_value = self.filtered_target.best
                filter_value = (sum(target_buffer) / len(target_buffer))
                if (best_value is None) or filter_value < best_value:
                    if len(target_buffer) > 0.1*target_buffer.maxlen:
                        self.checkpoints.save()
                        self.filtered_target.best = filter_value
        return data


@tdl.core.create_init_docstring
class Saver(tdl.core.TdlModel):
    @tdl.core.InputArgument
    def log_folder(self, value):
        if value is None:
            tdl.core.assert_initialized_if_available(
                self, 'log_folder', ['monitor_manager'])
            if tdl.core.is_property_set(self, 'monitor_manager'):
                value = self.monitor_manager.log_folder
            else:
                value = 'tdl_tmp/'
        return value

    @tdl.core.LazzyProperty
    def saver(self):
        tdl.core.assert_initialized(self, 'saver', ['log_folder'])

        def save():
            saver_path = os.path.join(self.saver.path, 'var_checkpoint')
            self.saver._saver.save(
                sess=self.session,
                save_path=saver_path,
                global_step=self.saver._save_id)
            self.saver._save_id += 1
            self.saver.last_saved = time()
            print('weights saved ({})'.format(saver_path))

        def restore():
            saver_path = os.path.join(
                self.saver.path,
                'var_checkpoint-{}'.format(self.saver._save_id-1))
            self.saver._saver.restore(self.session, saver_path)

        _saver = tf.train.Saver(var_list=self.var_list)
        path = os.path.join(self.log_folder, 'optimizer')
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        return tdl.core.SimpleNamespace(
            path=path, _saver=_saver, _save_id=0, last_saved=time(),
            save=save, restore=restore)

    def warmup(self, **kwargs):
        tdl.core.assert_initialized(self, 'warmup', ['saver'])

    def cleanup(self, **kwargs):
        self.saver.save()


@tdl.core.create_init_docstring
class CheckNan(tdl.core.TdlModel):
    @tdl.core.SubmodelInit
    def check_nan(self, n_trials=100):
        return tdl.core.SimpleNamespace(n_trials=n_trials)

    def warmup(self, **kwargs):
        tdl.core.assert_initialized(self, 'warmup', ['check_nan'])

    def step_fn(self, wrapped_fn=None, **kwargs):
        def is_any_nan(data):
            return any([np.isnan(value).any() for value in data.values()
                        if value is not None])

        def restore():
            print('Optimization resulted in Nan')
            for trial in range(self.check_nan.n_trials):
                self.checkpoints.restore()
                data = wrapped_fn(self, **kwargs)
                if not is_any_nan(data):
                    return data
            raise tdl.core.exceptions.NanResult()

        data = wrapped_fn(self, **kwargs)
        if is_any_nan(data):
            if (isinstance(self, Checkpointable) and
                    len(self.checkpoints.buffer) > 0):
                data = restore()
            else:
                raise tdl.core.exceptions.NanResult()
        return data


Optimizer = BaseOptimizer.wrap(CheckNan).wrap(CheckProgress)\
                         .wrap(Monitor)\
                         .wrap(CheckpointableProgress)\
                         .wrap(StatusBar)
