#  ***********************************************************************
#   Monitoring of training procedures
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
import io
import glob
import pickle
import shutil
import datetime
import itertools
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
import twodlearn.common
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import function

gradient_monitors_count = 0


class TrainingMonitor(object):
    ''' This class creates operations that logs information about the tensor
    x_tf during training
    '''

    def __init__(self, *x_tf, **kwargs):
        """Base initialization for a Training monitor.

        Args:
            *x_tf ([tf.tensor]): list of tf.tensors to be monitored (computed).
            **kwargs : type
                'name': each monitor has a name, if not provided, the name of
                    the first x_tf is used.
                'buffer_size': size of the buffer where values being monitored
                    are being stored
                'window_size': window_size for the moving average computations.
                'use_tensorboard': Use tensorboard to log the data.
                        TODO: so far, the only logging method is tensorboard,
                        hence this must be true

        Returns:
            TrainingMonitor: A TrainingMonitor that logs information while
                computing some basic statistics about the stream of data
                (moving average, min, max).
        """
        # x_tf: reference to a list of tensors on the comp. graph
        self.x_tf = x_tf
        # each monitor has a name, if not provided, the name of x_tf is used
        _given_name = (kwargs['name']
                       if 'name' in kwargs
                       else None)
        self.name = (_given_name
                     if _given_name is not None
                     else x_tf[0].name)
        # values being monitored are stored in a buffer of size buffer_size
        _given_buffer_size = (kwargs['buffer_size']
                              if 'buffer_size' in kwargs
                              else None)
        buffer_size = (_given_buffer_size
                       if _given_buffer_size is not None
                       else 1000)
        self.data_buffer = collections.deque(maxlen=buffer_size)
        self.step_buffer = collections.deque(maxlen=buffer_size)
        self._internal_step = 0
        # window size
        _given_window_size = (kwargs['window_size']
                              if 'window_size' in kwargs
                              else None)
        self.window_size = (_given_window_size
                            if _given_window_size is not None
                            else 100)
        # use tensorboard
        if 'use_tensorboard' in kwargs:
            self.use_tensorboard = kwargs['use_tensorboard']
        else:
            self.use_tensorboard = True
        # monitors that are not active should not be evaluated
        self.active = True
        # reference to the tf operation that returns the value being monitored
        self.op = None
        self._tf_summary = None
        # stats of the monitored quantity
        if 'compute_stats' in kwargs:
            self.compute_stats = kwargs['compute_stats']
        else:
            self.compute_stats = True
        if self.compute_stats:
            self.mean_buffer = collections.deque(maxlen=buffer_size)
            self.mean_step_buffer = collections.deque(maxlen=buffer_size)
            self._min = np.inf
            self._max = -np.inf
            self._mean_func = self._early_mean
            self._mean_min = np.inf
            self._mean_max = -np.inf
            self._mean = 0.0

    @property
    def current_value(self):
        if self.data_buffer:
            return self.data_buffer[-1]
        else:
            return np.nan

    @property
    def current_step(self):
        if self.step_buffer:
            return self.step_buffer[-1]
        else:
            return np.nan

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def mean_min(self):
        return self._mean_min

    @property
    def mean_max(self):
        return self._mean_max

    @property
    def data(self):
        ''' Creates a dataframe  with the data in the buffer
        @retval data_buffer as a dataframe
        '''
        return pd.DataFrame(np.array(self.data_buffer),
                            index=np.array(self.step_buffer),
                            columns=[self.name])

    @property
    def mean_data(self):
        ''' Creates a dataframe  with the data in mean_buffer
        @retval mean_buffer as a dataframe
        '''
        return pd.DataFrame(np.array(self.mean_buffer),
                            index=np.array(self.mean_step_buffer),
                            columns=[self.name])

    @property
    def tf_summary_op(self):
        return self._tf_summary_op

    def tf_summary_stats(self):
        return [tf.Summary.Value(tag=self.name + '/max',
                                 simple_value=self.max),
                tf.Summary.Value(tag=self.name + '/min',
                                 simple_value=self.min),
                tf.Summary.Value(tag=self.name + '/mean',
                                 simple_value=self.mean()),
                tf.Summary.Value(tag=self.name + '/mean_max',
                                 simple_value=self.mean_max),
                tf.Summary.Value(tag=self.name + '/mean_min',
                                 simple_value=self.mean_min)]

    def tf_summary_data(self):
        # if self.data_buffer:
        data = self.data_buffer[-1]
        # else:
        #    data = np.nan
        return tf.Summary.Value(tag=self.name, simple_value=data)

    def setup_op(self):
        if (self.op is not None) and self.use_tensorboard:
            self._tf_summary_op = tf.summary.scalar(self.name, self.op)

    def _early_mean(self, data):
        ''' calculates the moving average when the number of elements in the buffer
        is less than the window size
        '''
        n_elements = len(self.data_buffer)
        if n_elements < self.window_size:
            self._mean = ((self._mean * n_elements) + data) / (n_elements + 1)
        else:                   # if there are window_size elements in buffer
            self._mean_func = self._normal_mean
            self._normal_mean(data)

    def _normal_mean(self, data):
        ''' calculates the moving average when the number of elements in the buffer
        is equal or greather than the window size
        '''
        self._mean = ((self._mean * self.window_size)
                      - self.data_buffer[-self.window_size]
                      + data) / self.window_size
        if self._mean < self._mean_min:
            self._mean_min = self._mean
        if self._mean > self._mean_max:
            self._mean_max = self._mean

    def feed(self, data, step=None):
        """ feed data to the monitor
        @param data: data added into the buffer
        """
        if step is None:
            step = self._internal_step

        if self.compute_stats:
            # calculate maximum and minimum
            if data < self._min:  # TODO: check if data is numeric
                self._min = data
            if data > self._max:
                self._max = data
            # calculate moving average
            self._mean_func(data)
            if (self._internal_step % self.window_size == 0):
                self.mean_buffer.append(self.mean())
                self.mean_step_buffer.append(step)

        self.data_buffer.append(data)
        self.step_buffer.append(step)
        self._internal_step += 1

    def mean(self, window_size=None):
        """ Returns the mean for the last window_size elements
        @param n_elements
        """
        if window_size is None:
            return self._mean
        else:
            return sum(list(self.data_buffer)[-window_size:]) / window_size

    def get_stats(self):
        ''' get statistics
        @retval dictionary with the statistics of the monitor
        '''
        return {self.name + '/current': self.current_value,
                self.name + '/max': self.max,
                self.name + '/min': self.min,
                self.name + '/mean': self.mean(),
                self.name + '/mean_max': self.mean_max,
                self.name + '/mean_min': self.mean_min,
                self.name + '/steps': self.current_step}


class ScalarMonitor(TrainingMonitor):
    def __init__(self, x_tf=None, name=None, **kargs):
        if x_tf is not None:
            super(ScalarMonitor, self).__init__(x_tf, name=name, **kargs)
        else:
            super(ScalarMonitor, self).__init__(name=name, **kargs)


class ImageMonitor(TrainingMonitor):
    ''' Monitors provided image data '''
    @staticmethod
    def mplfig2tf(fig):
        ''' converts a matplotlib figure to a tensorflow tensor '''
        image_buf = io.BytesIO()
        fig.savefig(image_buf, format='png')
        image_buf.seek(0)
        image = tf.image.decode_png(image_buf.getvalue(), channels=4)
        return tf.expand_dims(image, 0)

    def mplfig2tfsummary(self, fig):
        tf_image = ImageMonitor.mplfig2tf(fig)
        summary_op = tf.summary.image(self.name, tf_image)
        summary = summary_op.eval()
        return summary

    def setup_op(self):
        ''' setup tf operation to be run, the output of this operation is
        the one saved by the monitor.
        The operation in this case is a tf summary
        '''
        if self.x_tf:
            op = self.x_tf[0]
            self._tf_summary_op = tf.summary.image(self.name, op)
            self.op = self._tf_summary_op
            return self.op

    def __init__(self, *x_tf, **kwargs):
        kwargs['compute_stats'] = False
        super(ImageMonitor, self).__init__(*x_tf, **kwargs)

    def tf_summary_data(self):
        ''' writes last feed value into a file to be read by tensorboard '''
        data = self.data_buffer[-1]
        # TODO: convert numpy array into tf.summary.image
        return data


class GradientL2Monitor(TrainingMonitor):
    ''' monitors the L2 norm of the gradient w.r.t x_tf'''

    def setup_op(self):
        # define tf operation
        global gradient_monitors_count
        gradient_monitors_count += 1

        def tf_op_grad(op, dl_dy):
            dl_dx = dl_dy

            grad_magnitude = tf.nn.l2_loss(dl_dy)
            self.op = grad_magnitude
            # print("grad for: {}".format(grad_magnitude))  # DEBUG

            return dl_dx

        @function.Defun(tf.float32,
                        func_name='grad_l2_monitor_{}'.format(
                            gradient_monitors_count),
                        python_grad_func=tf_op_grad)
        def tf_op(x):
            return x

        op_out = tf_op(*self.x_tf)
        super(GradientL2Monitor, self).setup_op()
        return op_out


class ResetGradientL2Monitor(TrainingMonitor):
    ''' monitors the gradients dl_dx. Gradients are set to zero if
    their norm is larger than clip_norm
    '''

    def __init__(self, x, clip_norm, **kwargs):
        if 'name' in kwargs:
            kwargs['name'] = kwargs['name'] + '/' + x.name
        else:
            kwargs['name'] = 'grad_magnitude/' + '/' + x.name
        super(ResetGradientL2Monitor, self).__init__(x, clip_norm, **kwargs)

    def setup_op(self):
        # define tf operation
        global gradient_monitors_count
        gradient_monitors_count += 1

        def tf_op_grad(op, dl_dy):
            x = op.inputs[0]
            clip_norm = op.inputs[1]

            # 1. evaluate gradient magnitude
            grad_magnitude = tf.nn.l2_loss(dl_dy)
            self.op = grad_magnitude
            # print("grad for: {}".format(grad_magnitude))  # DEBUG

            # 2. gradient clipping
            cond = grad_magnitude > clip_norm
            dl_dx = tf.cond(cond,
                            lambda: array_ops.zeros_like(x),
                            lambda: dl_dy)
            return dl_dx, array_ops.zeros_like(clip_norm)

        @function.Defun(tf.float32, tf.float32,
                        func_name='reset_grad_l2_monitor_{}'.format(
                            gradient_monitors_count),
                        python_grad_func=tf_op_grad)
        def tf_op(x, clip_norm):
            return x

        op_out = tf_op(*self.x_tf)
        super(ResetGradientL2Monitor, self).setup_op()
        return op_out


class OpMonitor(TrainingMonitor):
    ''' monitors the result of the given operation'''

    def setup_op(self):
        self.op = tf.convert_to_tensor(self.x_tf[0])
        super(OpMonitor, self).setup_op()
        return self.op


class MonitorManager(object):
    def __init__(self, log_folder='tmp/', tf_graph=None, use_tensorboard=True):
        self._monitors = dict()
        self._modified = True
        self._tf_summary = None
        self.use_tensorboard = True
        if self.use_tensorboard:
            shutil.rmtree(log_folder, ignore_errors=True)
            self._tf_summary_writer = \
                tf.summary.FileWriter(log_folder, tf_graph)
            self._tf_image_writer = \
                tf.summary.FileWriter(os.path.join(log_folder,
                                                   'images/'),
                                      tf_graph)
        else:
            self._tf_summary_writer = None

    def __getitem__(self, key):
        return self._monitors[key]

    @property
    def monitors(self):
        ''' Returns all monitors '''
        return [monitor for _, monitor in self._monitors.items()]

    @property
    def active_monitors(self):
        ''' Returns only active monitors '''
        return [monitor for _, monitor in self._monitors.items()
                if monitor.active]

    @property
    def tf_monitors(self):
        ''' Returns only active and valid monitors that have a
        tf op associated '''
        return [monitor for _, monitor in self._monitors.items()
                if monitor.active and (monitor.op is not None)]

    @property
    def monitors_data(self):
        ''' get data_buffer from the monitors
        @retval data_buffer from the list of monitors as a pandas dataframe
        '''
        return pd.concat([m.data for m in self.monitors], axis=1)

    @property
    def tf_summary(self):
        if (self._tf_summary is None) or (self._modified):
            assert self._monitors, "Monitor manager has no monitors attached"
            self._tf_summary = tf.summary.merge(
                [m.tf_summary for m in self.monitors])
            self._modified = False
        return self._tf_summary

    def write_stats(self, step=None):
        if not self._monitors:
            return
        if self.use_tensorboard:
            active_monitors = self.active_monitors
            scalar_monitors = [m for m in active_monitors
                               if not isinstance(m, ImageMonitor)]
            summary = tf.Summary(value=list(itertools.chain(
                *[m.tf_summary_stats() for m in scalar_monitors])))
            if step is None:
                # TODO: change step_buffer to be managed by the manager
                # instead of individual monitors
                step = max([m.step_buffer[-1] for m in self.active_monitors])
            # print('step: {}'.format(step)) # DEBUG
            self._tf_summary_writer.add_summary(summary, step)
        else:
            raise NotImplementedError(
                "Only tensorboard can be used at the moment")

    def write_data(self, step=None):
        ''' Writes last value feed into the monitor for being read by
        tensorboard '''
        if not self._monitors:
            return
        if self.use_tensorboard:
            active_monitors = self.active_monitors
            scalar_monitors = [m for m in active_monitors
                               if not isinstance(m, ImageMonitor)]
            image_monitors = [m for m in active_monitors
                              if isinstance(m, ImageMonitor)]
            # get summary
            scalars_summary = tf.Summary(value=[m.tf_summary_data()
                                                for m in scalar_monitors
                                                if m.step_buffer])
            images_summary = [m.tf_summary_data() for m in image_monitors]
            if len(images_summary) > 1:
                raise ValueError("Only one image monitor per manager supported"
                                 " at the moment")
            if step is None:
                # TODO: change step_buffer to be managed by the manager
                # instead of individual monitors
                step = max([m.step_buffer[-1] for m in self.active_monitors
                            if m.step_buffer])
            # Write into files
            self._tf_summary_writer.add_summary(scalars_summary, step)
            if images_summary:
                # TODO: INCLUDE ALL IMAGES
                print("----> image with step: {}".format(step))
                self._tf_image_writer.add_summary(
                    images_summary[0], step)
        else:
            raise NotImplementedError(
                "Only tensorboard can be used at the moment")

    def feed(self, data_in, step):
        if isinstance(data_in, dict):
            for name, data in data_in.items():
                self._monitors[name].feed(data, step)
        elif isinstance(data_in, list):
            assert len(self._monitors) == len(data_in), \
                'Provided data list does not match number of monitors'
            i = 0
            for name, monitor in self._monitors.items():
                monitor.feed(data_in[i])
                i += 1

    def add_monitor(self, monitor):
        assert monitor.name not in self._monitors
        self._modified = True
        self._monitors[monitor.name] = monitor
        return monitor.setup_op()

    def get_stats(self):
        ''' get statistics from loggers
        @retval dictionary with the statistics of each logger
        '''
        monitors_stats = [m.get_stats()
                          for m in self.monitors if m.compute_stats]
        monitors_stats = {k: v for d in monitors_stats for k, v in d.items()}
        return monitors_stats

    def flush(self):
        self._tf_summary_writer.flush()

    def close(self):
        self._tf_summary_writer.close()

    def reopen(self):
        self._tf_summary_writer.reopen()


class TrainingMonitorManager(object):
    def __init__(self, log_folder='tmp/monitors/', tf_graph=None):
        shutil.rmtree(log_folder, ignore_errors=True)
        self.log_folder = log_folder
        # os.rmdir(log_folder)
        self.train = MonitorManager(
            log_folder=os.path.join(log_folder, 'train'),
            tf_graph=tf_graph)
        self.valid = MonitorManager(
            log_folder=os.path.join(log_folder, 'valid'))
        self.monitoring = MonitorManager(
            log_folder=os.path.join(log_folder, 'monitoring'))

    def get_stats(self):
        train_stats = self.train.get_stats()
        valid_stats = self.train.get_stats()
        # return {**train_stats, **valid_stats}
        twodlearn.merge_dicts(train_stats, valid_stats)

    def flush(self):
        self.train.flush()
        self.valid.flush()
        self.monitoring.flush()

    def close(self):
        self.train.close()
        self.valid.close()
        self.monitoring.close()

    def reopen(self):
        self.train.reopen()
        self.valid.reopen()
        self.monitoring.reopen()


class SimpleTrainingMonitor(TrainingMonitorManager):
    def _init_monitor(self, vars, name, tf_graph=None):
        monitor = MonitorManager(
            log_folder=os.path.join(self.log_folder, name),
            tf_graph=tf_graph)
        for name, var in vars.items():
            monitor.add_monitor(OpMonitor(var, name=name))
        return monitor

    def __init__(self, train_vars, valid_vars=None, monitoring_vars=None,
                 log_folder='tmp/monitors/', tf_graph=None):
        shutil.rmtree(log_folder, ignore_errors=True)
        self.log_folder = log_folder
        # os.rmdir(log_folder)
        train_vars = (dict() if train_vars is None
                      else train_vars)
        valid_vars = (dict() if valid_vars is None
                      else valid_vars)
        monitoring_vars = (dict() if monitoring_vars is None
                           else monitoring_vars)
        self.train = self._init_monitor(train_vars, 'train', tf_graph=tf_graph)
        self.valid = self._init_monitor(valid_vars, 'valid')
        self.monitoring = self._init_monitor(monitoring_vars, 'monitoring')


class MonitoringData(object):
    ''' Class for loading tfevent files '''
    def read_tf_scalar_summary(file_path):
        ''' Returns the tfevent summary in file_path as a pandas
        dataframe '''

        data = dict()
        data['step'] = list()

        for event in tf.train.summary_iterator(file_path):
            if event.summary.value:
                data['step'].append(event.step)
            for v in event.summary.value:
                if v.tag not in data:
                    data[v.tag] = [v.simple_value]
                else:
                    data[v.tag].append(v.simple_value)

        return pd.DataFrame.from_dict(data)

    def search_monitor_data(self, search_path):
        ''' search for tfevents inside search_path recursively '''

        contents = os.listdir(search_path)
        print("searching in {}".format(search_path))
        if len(contents) == 0:
            return None

        out = dict()
        for item in contents:
            if 'events.out.tfevents' in item:
                event_file_path = os.path.join(search_path, item)
                try:
                    return MonitoringData.read_tf_scalar_summary(event_file_path)
                except:
                    return None
            elif 'image' not in item:
                folder_path = os.path.join(search_path, item)
                if os.path.isdir(folder_path):
                    out[item] = self.search_monitor_data(folder_path)
        return out

    def __init__(self, folder):
        performance_filename = glob.glob(folder + '*performance.pkl')
        print("loading {}".format(performance_filename[0]))

        performance_file = open(performance_filename[0], 'rb')
        self.performance = pickle.load(performance_file)
        performance_file.close()

        self.monitors = self.search_monitor_data(
            os.path.join(folder, 'monitors/'))


class MonitoringDataV2(object):
    ''' Class for loading tfevent files '''
    @staticmethod
    def read_tf_scalar_summary(file_path):
        ''' Returns the tfevent summary in file_path as a pandas
        dataframe '''

        data = dict()
        data['step'] = list()

        for event in tf.train.summary_iterator(file_path):
            if event.summary.value:
                data['step'].append(event.step)
            for v in event.summary.value:
                if v.tag not in data:
                    data[v.tag] = [v.simple_value]
                else:
                    data[v.tag].append(v.simple_value)

        return pd.DataFrame.from_dict(data)

    def search_monitor_data(self, search_path):
        ''' search for tfevents inside search_path recursively '''

        contents = os.listdir(search_path)
        print("searching in {}".format(search_path))
        if len(contents) == 0:
            return None

        out = dict()
        for item in contents:
            if 'events.out.tfevents' in item:
                event_file_path = os.path.join(search_path, item)
                #try:
                return MonitoringDataV2.read_tf_scalar_summary(event_file_path)
                #except:
                #    print('Error reading {}'.format(event_file_path))
                #    return None
            elif 'image' not in item:
                subfolder = os.path.join(search_path, item)
                if os.path.isdir(subfolder):
                    subfolder_data = self.search_monitor_data(subfolder)
                    if isinstance(subfolder_data, dict):
                        for name, data in subfolder_data.items():
                            out[os.path.join(item, name)] = data
                    elif isinstance(subfolder_data, pd.DataFrame):
                        out[item] = subfolder_data
        return (out if out
                else None)

    def __init__(self, folder):
        self.folder = folder
        options_file = glob.glob(folder + '*options.pkl')
        if options_file:
            print("loading {}".format(options_file[0]))
            with open(options_file[0], 'rb') as file:
                self.options = pickle.load(file)
        else:
            options_file = None

        self.monitors = self.search_monitor_data(folder)


class PerformanceSaver(object):
    def __init__(self, filename, tmp_path=None):
        # Create tmp folder if it does not exist
        if tmp_path is None:
            tmp_path = 'tmp/'

        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        self.filename = os.path.join(tmp_path, filename)

        try:
            self.datafile = open(self.filename, 'rb')
            self.data = pickle.load(self.datafile)
            self.datafile.close()
        except:
            self.data = pd.DataFrame()

    def log(self, log_dict):
        row_df = pd.DataFrame.from_records([log_dict])
        row_df['date'] = datetime.datetime.now()

        self.data = pd.concat([self.data, row_df], axis=0)

    def clear(self):
        self.data = pd.DataFrame()
        self.save()

    def save(self, log_dict=None):
        if log_dict is not None:
            self.log(log_dict)

        output_file = open(self.filename, 'wb')
        pickle.dump(self.data, output_file)
        output_file.close()
