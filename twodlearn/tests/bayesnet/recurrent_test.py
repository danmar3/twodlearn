# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pytest
docutils = pytest.importorskip("pyfmi")
try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import os
import pickle
import shutil
import unittest
import numpy as np
from time import time
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
import twodlearn as tdl
import twodlearn.debug
from twodlearn import optim
import twodlearn.templates.supervised
import twodlearn.bayesnet.recurrent
import twodlearn.monitoring as tdlm
from twodlearn.datasets import tsdataset
from twodlearn.reinforce.systems import (
    Cartpole, Acrobot, Cstr)
try:
    from twodlearn.reinforce.modelica.controllers.noisy_fo \
        import NoisyFO
    from twodlearn.reinforce.modelica.controllers.saturated_noisy_fo \
        import SaturatedNoisyFO
except ImportError:
    pass
try:
    from types import SimpleNamespace
except ImportError:
    from argparse import Namespace as SimpleNamespace


TESTS_PATH = os.path.dirname(os.path.abspath(__file__))
TMP_PATH = os.path.join(TESTS_PATH, 'tmp/')


# ███    ███  ██████  ██████  ███████ ██
# ████  ████ ██    ██ ██   ██ ██      ██
# ██ ████ ██ ██    ██ ██   ██ █████   ██
# ██  ██  ██ ██    ██ ██   ██ ██      ██
# ██      ██  ██████  ██████  ███████ ███████

class RnnBayesModel(twodlearn.templates.supervised.MlModel):
    @property
    def plant(self):
        return self._plant

    def _init_options(self, options):
        narx_options = {'cell/options': None}
        default = {'narx/options': narx_options,
                   'narx/window_size': 1,
                   'narx/cell/loc/n_hidden': [100, 100],
                   'narx/cell/loc/afunction': tdl.feedforward.selu01,
                   'narx/cell/loc/keep_prob': 0.9,
                   'narx/cell/scale/n_hidden': [10, 10],
                   'narx/cell/scale/afunction': tdl.feedforward.selu01,
                   'narx/train/batch_size': 500,
                   'narx/train/n_steps': 20,
                   'narx/test/n_steps': 30,
                   'narx/test/n_particles': 500,
                   'narx/reparameterization/type': 'local_gaussian',
                   'narx/cell/loc/prior/stddev': 100.0,
                   'narx/cell/scale/prior/stddev': 100.0,
                   'narx/noise_x/heteroscedastic': False}
        options = tdl.core.check_defaults(options, default)
        # if not options['narx/noise_x/heteroscedastic']:
        #    options = {k: v for k, v in options.items()
        #               if 'narx/cell/scale' not in k}
        options = super(RnnBayesModel, self)._init_options(options)
        return options

    def format_batch(self, batch, window_size, n_steps):
        _y = [batch['y'][i, :, :]
              for i in range(batch['y'].shape[0])]
        x0 = _y[:window_size]
        y = _y[window_size:]
        # dy = [_y[i] - _y[i - 1] for i in range(window_size, len(_y))]
        u = [batch['u'][i, :, :]
             for i in range(window_size - 1, batch['u'].shape[0] - 1)]

        assert (len(x0) == window_size and
                len(y) == n_steps and
                len(u) == n_steps), \
            'len of x0, y or u do not coincide with provided '\
            'window_size and n_steps'
        return x0, y, u

    def next_batch_in_format(self, dataset, window_size, n_steps, batch_size):
        batch = dataset.next_batch(window_size=window_size + n_steps,
                                   batch_size=batch_size)
        x0, y, u = self.format_batch(batch, window_size, n_steps)
        return x0, y, u

    def feed_train(self, dataset):
        x0, y, u = self.next_batch_in_format(
            dataset=dataset,
            window_size=self.options['narx/window_size'],
            n_steps=self.options['narx/train/n_steps'],
            batch_size=self.options['narx/train/batch_size'])
        feed_dict = dict()
        for i in range(self.options['narx/window_size']):
            feed_dict[self.train.x0[i].loc] = x0[i]
        for t in range(len(u)):
            feed_dict[self.train.loss.labels[t]] = y[t]
            feed_dict[self.train.inputs[t]] = u[t]
        feed_dict[self.train.loss.n_samples] = dataset.n_samples
        return feed_dict

    def feed_valid(self, dataset):
        return self.feed_train(dataset)

    def feed_test(self, x0, inputs):
        """Creates a feed dictionary with the values in x0 an inputs.

        Args:
            x0 ([]np.Array): list of previous window_size observed states.
            inputs ([]np.Array): list with the exogenous inputs.

        Returns:
            dict: tf feed dictionary with the values of x0 and inputs for the
                test model
        """
        window_size = self.test.window_size
        n_steps = len(inputs)
        assert n_steps <= len(self.test.unrolled),\
            'number of provided inputs exceeds the number of unrolled '\
            'networks in the test model'
        feed_dict = dict()
        for i in range(window_size):
            feed_dict[self.test.x0[i].loc.base] = x0[i]
        for t in range(n_steps):
            feed_dict[self.test.inputs[t].base] = inputs[t]
        return feed_dict

    @tdl.core.EagerMethod
    def confidence_eval(self, value):
        mc_loc = [out.samples.mean for out in self.test.y]
        mc_scale = [out.samples.stddev for out in self.test.y]
        target = [tf.placeholder(tf.float32, shape=mc_loc[0].shape)
                  for t in range(len(mc_loc))]
        with tf.name_scope('confidence_eval'):
            with tf.name_scope('stddev_deviation'):
                std_deviation = [tf.abs(target[t] - mc_loc[t]) / mc_scale[t]
                                 for t in range(len(mc_loc))]
                std_deviation = tf.add_n(std_deviation) / len(mc_loc)
            with tf.name_scope('stddev_norm'):
                std_norm = tf.add_n([tf.reduce_sum(scale**2)
                                     for scale in mc_scale]) / len(mc_loc)
            with tf.name_scope('average_stddev'):
                average_std = (tf.add_n([scale for scale in mc_scale]) /
                               len(mc_loc))
            with tf.name_scope('average_l2loss'):
                losses = [tdl.losses.L2Loss(mc_loc[t], labels=target[t])
                          for t in range(len(mc_loc))]
                average_l2loss = tf.convert_to_tensor(
                    tdl.losses.AddNLosses(losses))/np.float32(len(mc_loc))

        self._confidence_eval = SimpleNamespace()
        self._confidence_eval.stddev_deviation = std_deviation
        self._confidence_eval.stddev_norm = std_norm
        self._confidence_eval.average_stddev = average_std
        self._confidence_eval.target = target
        self._confidence_eval.average_l2loss = average_l2loss

    @confidence_eval.eval
    def confidence_eval(self, x0, inputs, target):
        feed_dict = self.feed_test(x0, inputs)
        for yi, yip in zip(target, self._confidence_eval.target):
            feed_dict[yip] = yi
        stddev_deviation, stddev_norm, average_l2loss, average_stddev = \
            self.session.run(
                [self._confidence_eval.stddev_deviation,
                 self._confidence_eval.stddev_norm,
                 self._confidence_eval.average_l2loss,
                 self._confidence_eval.average_stddev], feed_dict=feed_dict)
        return stddev_deviation, stddev_norm, average_l2loss, average_stddev

    @tdl.core.EagerMethod
    def cr_zscore(self, value):
        mc_loc = [out.samples.mean for out in self.test.y]
        mc_scale = [out.samples.stddev for out in self.test.y]
        samples = [tf.placeholder(tf.float32, shape=mc_loc[0].shape)
                   for t in range(len(mc_loc))]
        with tf.name_scope('zscore'):
            z_score = [(samples[t] - mc_loc[t]) / mc_scale[t]
                       for t in range(len(mc_loc))]
            z_score = tf.stack(z_score, axis=0)
        with tf.name_scope('containing_ratio'):
            # cr1 = tf.reduce_mean(tf.cast(tf.reduce_all(tf.abs(z_score) < 1.0,
            #                                           axis=1),
            #                             tf.float32))
            # cr2 = tf.reduce_mean(tf.cast(tf.reduce_all(tf.abs(z_score) < 2.0,
            #                                           axis=1),
            #                             tf.float32))
            # cr3 = tf.reduce_mean(tf.cast(tf.reduce_all(tf.abs(z_score) < 3.0,
            #                                           axis=1),
            #                             tf.float32))
            _zscore = tf.sqrt(tf.reduce_sum(z_score**2.0, axis=1))
            cr1 = tf.reduce_mean(tf.cast(_zscore < 1.0, tf.float32))
            cr2 = tf.reduce_mean(tf.cast(_zscore < 2.0, tf.float32))
            cr3 = tf.reduce_mean(tf.cast(_zscore < 3.0, tf.float32))

        self._cr_zscore = SimpleNamespace()
        self._cr_zscore.cr = [cr1, cr2, cr3]
        self._cr_zscore.zscore = z_score
        self._cr_zscore.samples = samples

    @cr_zscore.eval
    def cr_zscore(self, x0, inputs, target):
        feed_dict = self.feed_test(x0, inputs)
        for yi, yip in zip(target, self._cr_zscore.samples):
            feed_dict[yip] = yi
        cr1, cr2, cr3, zscore = self.session.run(
            [self._cr_zscore.cr[0],
             self._cr_zscore.cr[1],
             self._cr_zscore.cr[2],
             self._cr_zscore.zscore], feed_dict=feed_dict)
        return (cr1, cr2, cr3), zscore

    def cr_zscore_eval(self, dataset):
        x0, target, inputs = self.next_batch_in_format(
            dataset=dataset,
            window_size=self.test.window_size,
            n_steps=len(self.test.y),
            batch_size=1)
        target = [np.squeeze(y) for y in target]
        cr, zscore = self.cr_zscore(x0, inputs, target)
        return cr, zscore

    def fit(self, dataset):
        def confidence_eval():
            x0, target, inputs = self.next_batch_in_format(
                dataset=dataset.test,
                window_size=self.test.window_size,
                n_steps=len(self.test.y),
                batch_size=1)
            target = [np.squeeze(y) for y in target]
            stddev_deviation, stddev_norm, average_l2loss, average_stddev = \
                self.confidence_eval(x0, inputs, target)
            return stddev_deviation, stddev_norm, average_l2loss, average_stddev

        def feed_train():
            return self.feed_train(dataset.train)

        def feed_valid():
            # ------
            deviation, norm, l2loss, average = confidence_eval()
            data = {'test/confidence/stddev_deviation': deviation.max(),
                    'test/confidence/stddev_norm': norm,
                    'test/confidence/l2loss': l2loss}
            for i in range(self.model.n_outputs):
                monitor_i = 'test/confidence/average_stddev_{}'.format(i)
                data[monitor_i] = average[i]
            self.test_monitor.feed(data, step=None)
            self.test_monitor.write_data()
            # ------
            return self.feed_train(dataset.valid)

        t0 = time()
        self.optimizer.run(
            n_train_steps=self.options['optim/train/max_iter'],
            feed_train=feed_train,
            valid_eval_freq=self.options['optim/valid/eval_freq'],
            feed_valid=feed_valid)
        t1 = time()
        self.test_monitor.flush()
        print('Training took: ', (t1 - t0), 's')

    def visualize_predictions(self, dataset, n_steps=None, ax=None):
        ''' Plots the model's predictions for data extracted from the
        training/validation/testing datasets.
        '''
        n_outputs = self.model.n_outputs
        if n_steps is None:
            n_steps = len(self.test.unrolled)
        else:
            assert n_steps <= len(self.test.unrolled),\
                'provided number of steps is bigger than the number '\
                'of steps allocated for simulation'
        # Get data
        x0, target, u = self.next_batch_in_format(
            dataset=dataset, window_size=self.options['narx/window_size'],
            n_steps=n_steps, batch_size=1)
        feed_dict = self.feed_test(x0, u)

        # Get predictions
        _y = [out.samples.mean
              for out in self.test.y[:n_steps]]
        _y += [out.samples.stddev
               for out in self.test.y[:n_steps]]
        yp = self.session.run(_y, feed_dict=feed_dict)
        yp_mean = yp[:n_steps]
        yp_stddev = yp[n_steps:]
        yp_mean = np.stack(yp_mean, axis=0)
        yp_stddev = np.stack(yp_stddev, axis=0)
        target = np.concatenate(target, axis=0)
        # Plot
        fig, ax = plt.subplots(n_outputs, 1,
                               figsize=(10, 3 * n_outputs),
                               squeeze=False)

        for var_id in range(n_outputs):
            self._plot_mu_std(yp_mean[:, var_id],
                              yp_stddev[:, var_id],
                              ax[var_id, 0],
                              y=target[:, var_id],
                              title='x' + str(var_id))
        return target, yp

    def _plot_mu_std(self, yp_mu, yp_std, ax, x=None, y=None, title=''):
        ''' Creates a plot of the given 1D yp_mu points, with standard deviation
        given by yp_std
        '''

        if x is None:
            x = np.arange(yp_mu.shape[0])
        if y is not None:
            ax.plot(np.squeeze(x), np.squeeze(y), 'k+')
        ax.plot(np.squeeze(x),
                np.squeeze(yp_mu),
                'b')
        ax.plot(np.squeeze(x),
                np.squeeze(yp_mu) + np.squeeze(yp_std),
                'r')
        ax.plot(np.squeeze(x),
                np.squeeze(yp_mu) - np.squeeze(yp_std),
                'r')
        ax.set_title(title, fontsize=15)

    def _init_model(self):
        n_inputs = self.plant.n_actuators
        n_outputs = self.plant.n_sensors
        window_size = self.options['narx/window_size']
        with tf.name_scope('narx') as scope:
            if self.options['narx/reparameterization/type'] == 'local_gaussian':
                loc = tdl.bayesnet.BayesianMlp(
                    n_inputs=n_inputs + n_outputs * window_size,
                    n_outputs=n_outputs,
                    n_hidden=self.options['narx/cell/loc/n_hidden'],
                    afunction=self.options['narx/cell/loc/afunction'])
                if self.options['narx/noise_x/heteroscedastic'] is False:
                    scale = None
                else:
                    scale = tdl.bayesnet.BoundedBayesianMlp(
                        n_inputs=n_inputs + n_outputs * window_size,
                        n_outputs=n_outputs,
                        n_hidden=self.options['narx/cell/scale/n_hidden'],
                        afunction=self.options['narx/cell/scale/afunction'],
                        lower=self.options['narx/cell/scale/lower_bound'],
                        upper=self.options['narx/cell/scale/upper_bound'])
            elif self.options['narx/reparameterization/type'] == 'dropout':
                loc = tdl.bayesnet.BernoulliBayesianMlp(
                    n_inputs=n_inputs + n_outputs * window_size,
                    n_outputs=n_outputs,
                    n_hidden=self.options['narx/cell/loc/n_hidden'],
                    keep_prob=self.options['narx/cell/loc/keep_prob'],
                    afunction=self.options['narx/cell/loc/afunction'])
                if self.options['narx/noise_x/heteroscedastic'] is False:
                    scale = None
                else:
                    scale = tdl.bayesnet.BoundedBernoulliBayesianMlp(
                        n_inputs=n_inputs + n_outputs * window_size,
                        n_outputs=n_outputs,
                        keep_prob=1.0,
                        lower=self.options['narx/cell/scale/lower_bound'],
                        upper=self.options['narx/cell/scale/upper_bound'],
                        n_hidden=self.options['narx/cell/scale/n_hidden'],
                        afunction=self.options['narx/cell/scale/afunction'])

            fz = twodlearn.bayesnet.Normal(loc=loc, scale=scale,
                                           shape=[None, n_outputs])
            cell = tdl.bayesnet.recurrent.NormalNarxCell(fz=fz)
            regularizer = fz.regularizer.init(
                loc_scale=self.options['narx/cell/loc/prior/stddev'],
                scale_scale=self.options['narx/cell/scale/prior/stddev'])

        model = tdl.bayesnet.recurrent.BayesNarx(
            cell=cell, window_size=window_size, name=scope)
        model.regularizer = regularizer

        if self.options['narx/normalize']:
            normalizer = dict()
            train_stats = self.dataset.train.get_stats(['y', 'u'])
            normalizer['y'] = tdl.normalizer.Normalizer(
                loc=train_stats.mean['y'],
                scale=train_stats.stddev['y'])
            normalizer['u'] = tdl.normalizer.Normalizer(
                loc=train_stats.mean['u'],
                scale=train_stats.stddev['u'])
            model.cell.normalizer = normalizer
        model.n_inputs = n_inputs
        model.n_outputs = n_outputs
        return model

    def _init_train_model(self):
        options = {
            'regularizer/prior/stddev':
            self.options['narx/cell/loc/prior/stddev']}
        train = self.model.evaluate(
            n_unrollings=self.options['narx/train/n_steps'],
            options=options,
            name='train')
        n_samples = tf.placeholder(tf.float32)
        train.loss.init(reg_alpha=1.0 / n_samples)
        train.loss.n_samples = n_samples
        return train

    def _init_valid_model(self):
        return self.train

    def _init_test_model(self):
        test = self.model.mc_evaluate(
            n_particles=self.options['narx/test/n_particles'],
            n_unrollings=self.options['narx/test/n_steps'],
            name='mc_test')
        return test

    def _init_training_monitor(self, logger_path):
        ml_monitor = tdlm.TrainingMonitorManager(
            log_folder=logger_path,
            tf_graph=self.session.graph)

        ml_monitor.train.add_monitor(
            tdlm.OpMonitor(self.train.loss.empirical,
                           name="train/loss"))
        ml_monitor.train.add_monitor(
            tdlm.OpMonitor(self.train.loss.value,
                           name="train/losswithreg"))

        ml_monitor.valid.add_monitor(
            tdlm.OpMonitor(self.valid.loss.empirical,
                           name="valid/loss"))
        return ml_monitor

    def _init_test_monitor(self, logger_path):
        ml_monitor = tdlm.MonitorManager(
            log_folder=os.path.join(logger_path, 'test'))
        ml_monitor.add_monitor(
            tdlm.TrainingMonitor(name="test/confidence/stddev_deviation"))
        ml_monitor.add_monitor(
            tdlm.TrainingMonitor(name="test/confidence/stddev_norm"))
        ml_monitor.add_monitor(
            tdlm.TrainingMonitor(name="test/confidence/l2loss"))
        for i in range(self.test.y[0].value.shape[1].value):
            ml_monitor.add_monitor(
                tdlm.TrainingMonitor(
                    name="test/confidence/average_stddev_{}".format(i)))
        return ml_monitor

    def _init_optimizer(self, loss=None):
        if loss is None and isinstance(self.train.loss, twodlearn.losses.Loss):
            loss = self.train.loss.value
        else:
            loss = self.train.loss

        optimizer = optim.Optimizer(
            loss=loss,
            var_list=tdl.core.get_trainable(self.model),
            session=self.session,
            monitor_manager=self.monitor,
            n_logging=self.options['optim/n_logging'],
            learning_rate=self.options['optim/train/learning_rate'],
            saver=optim.SimpleSaver(
                var_list=tdl.core.get_trainable(self.model),
                logger_path=self.monitor.log_folder,
                session=self.session))

        return optimizer

    @tdl.core.EagerMethod
    def save(self, value):
        self.saver = tf.train.Saver(
            var_list=tdl.core.get_trainable(self.model),
            max_to_keep=30)
        self._saver_folder = os.path.join(self.logger_path,
                                          'model_checkpoints')

        if os.path.exists(self._saver_folder):
            shutil.rmtree(self._saver_folder)
        os.makedirs(self._saver_folder)

    @save.eval
    def save(self, global_step):
        self.saver.save(
            sess=self.session,
            save_path=os.path.join(self._saver_folder, 'model_saved_vars'),
            global_step=global_step)

    def load(self, ckpt_name):
        '''Load trainable variables parameters from ckpt_name'''
        self.saver.restore(self.session, save_path=ckpt_name)

    def __init__(self, plant, dataset=None,
                 options=None, logger_path='tmp', session=None):
        self._plant = plant
        self.dataset = dataset
        self.logger_path = logger_path
        super(RnnBayesModel, self).__init__(options=options,
                                            logger_path=logger_path,
                                            session=session)
        self.test_monitor = self._init_test_monitor(logger_path)


# ███████ ██    ██ ██████  ███████ ██████  ██    ██ ██ ███████ ███████ ██████
# ██      ██    ██ ██   ██ ██      ██   ██ ██    ██ ██ ██      ██      ██   ██
# ███████ ██    ██ ██████  █████   ██████  ██    ██ ██ ███████ █████   ██   ██
#      ██ ██    ██ ██      ██      ██   ██  ██  ██  ██      ██ ██      ██   ██
# ███████  ██████  ██      ███████ ██   ██   ████   ██ ███████ ███████ ██████

class Supervised(twodlearn.templates.supervised.Supervised):
    @property
    def plant(self):
        return self._plant

    def _init_options(self, options):
        default = {'plant/name': 'cartpole',
                   'dataset/init/n_trajectories': 100,
                   'dataset/init/exploration_std': 0.3,
                   'dataset/init/n_steps': 1000,
                   'dataset/normalize': False,
                   \
                   'optim/train/max_iter': 2000,
                   'optim/valid/eval_freq': 5,
                   'optim/train/learning_rate': 0.001,
                   \
                   'narx/normalize': True,
                   'narx/window_size': 1,
                   'narx/cell/n_hidden': [100, 100],
                   'narx/train/batch_size': 1000,
                   'narx/train/n_steps': 20,
                   'narx/test/n_steps': 30,
                   'narx/test/n_particles': 500,
                   'narx/reparameterization/type': 'local_gaussian',
                   'narx/cell/loc/prior/stddev': 100.0,
                   'narx/cell/scale/prior/stddev': 100.0,
                   'narx/noise_x/heteroscedastic': False}
        options = tdl.core.check_defaults(options, default)
        options = super(Supervised, self)._init_options(options)
        return options

    @tdl.core.EagerMethod
    def save_dataset(self, value):
        self._dataset_folder = os.path.join(self.tmp_path, 'dataset/')
        if os.path.exists(self._dataset_folder):
            shutil.rmtree(self._dataset_folder)
        os.makedirs(self._dataset_folder)

    @save_dataset.eval
    def save_dataset(self, filename):
        self.dataset.save(os.path.join(self._dataset_folder, filename))

    def load_dataset(self, filename):
        dataset = tsdataset.TSDatasets.from_saved_file(filename)
        self._dataset = dataset

    def _init_dataset(self):
        ''' Record some data with random actions '''
        def uniform_init(x, k):
            return (np.random.normal(
                loc=0.0,
                scale=self.options['dataset/init/exploration_std'],
                size=[self.plant.n_actuators]))

        if self.options['plant/name'] == 'cstr':
            init_controller = SaturatedNoisyFO(1.0)
            init_controller.set_parameters(
                y_min=200, y_max=350, nominal=250,
                tau=3.0, x_stddev=50.0, y_stddev=20.0)
        else:
            init_controller = uniform_init

        print('Initializing Dataset using random inputs')
        records = list()
        for i in tqdm(range(self.options['dataset/init/n_trajectories'])):
            sim_data = self.plant.simulate(
                policy=init_controller,
                steps=self.options['dataset/init/n_steps'],
                render=False)

            records.append(tsdataset.Record(sim_data))

        # Create initial dataset
        n_train_samples = int(
            np.ceil(self.options['dataset/init/n_trajectories'] * 0.5))
        dataset = tsdataset.TSDatasets(
            train=records[:n_train_samples],
            valid=records[n_train_samples:],
            test=[r.copy()
                  for r in records[n_train_samples:]])
        # dataset.set_groups({'x': '^x.*', 'u': ['u'], 'diff': '^diff_x.*'})
        dataset.set_groups({'y': '^x[0-9]',
                            'u': '^u[0-9]'})

        # Normalize the dataset
        if self.options['dataset/normalize']:
            dataset.normalize(['y', 'u'])
        print('Dataset generated. std:{}'
              ''.format(dataset.train.normalizer.std))

        return dataset

    def _init_ml_model(self, logger_path):
        model = RnnBayesModel(plant=self.plant,
                              dataset=self.dataset,
                              options=self.options,
                              logger_path=logger_path)
        return model

    def run_training(self):
        self.ml_model.fit(self.dataset)

    def _init_plant(self):
        plant_options = {'cartpole': Cartpole,
                         'acrobot': Acrobot,
                         'cstr': lambda: Cstr(self.options['plant/dt'])}
        if self.options['plant/name'] in plant_options:
            SystemClass = plant_options[self.options['plant/name']]
            return SystemClass()
        else:
            raise ValueError("Specified plant is not available, options are:"
                             " {}".format(list(plant_options.keys())))

    def __init__(self, options=None, tmp_path='tmp'):
        self._tmp_path = tmp_path
        self._options = self._init_options(options)
        self._plant = self._init_plant()
        self._dataset = self._init_dataset()
        self._ml_model = self._init_ml_model(self.tmp_path)


# ███████ ██   ██  ██████   ██████  ████████ ██ ███    ██  ██████
# ██      ██   ██ ██    ██ ██    ██    ██    ██ ████   ██ ██
# ███████ ███████ ██    ██ ██    ██    ██    ██ ██ ██  ██ ██   ███
#      ██ ██   ██ ██    ██ ██    ██    ██    ██ ██  ██ ██ ██    ██
# ███████ ██   ██  ██████   ██████     ██    ██ ██   ████  ██████

class ShootingOptim(tdl.core.TdlProgram):
    def _init_test_model(self):
        """Define the operation that computes the monte-calo estimation of the
        system trajectory.
        """
        test = self.model.mc_evaluate(
            n_unrollings=self.options['shooting/mc_estimate/n_steps'],
            n_particles=self.options['shooting/mc_estimate/n_particles'],
            name='shooting_test')

        with tf.name_scope('shooting_test/loss'):
            target = tf.placeholder(tf.float32,
                                    shape=[1, self.model.n_outputs])
            loss_y = self._define_loss_y(test.y, target)
            test.loss_y = loss_y
            test.target = target
        return test

    def feed_test(self, x0, u):
        feed_dict = dict()
        feed_dict[self.test.x0[0].loc.base] = (x0[0] if isinstance(x0, list)
                                               else x0)
        for k in range(len(u)):
            feed_dict[self.test.inputs[k].base] = u[k]
        return feed_dict

    def particles_trajectory(self, x0, u):
        """Use the test model to estimate the trajectory of the system

        Args:
            x0 (np.Array): initial state of the system.
            u (list(np.Array)): list with the control signals indexed by time
                step.

        Returns:
            (np.Array): estimated mean for the trajectory obserbable states.
            (np.Array): estimated stddev for the trajectory obserbable states.
        """
        feed_dict = self.feed_test(x0, u)
        particles = [net.dz.output.samples.value for net in self.test.unrolled]
        particles = particles[:len(u)]
        output = self.session.run(particles, feed_dict=feed_dict)
        return output

    def estimate_trajectory(self, x0, u):
        """Use the test model to estimate the trajectory of the system

        Args:
            x0 (np.Array): initial state of the system.
            u (list(np.Array)): list with the control signals indexed by time
                step.

        Returns:
            (np.Array): estimated mean for the trajectory obserbable states.
            (np.Array): estimated stddev for the trajectory obserbable states.
        """
        feed_dict = self.feed_test(x0, u)
        loc = [self.test.y[t].samples.mean for t in range(len(u))]
        scale = [self.test.y[t].samples.stddev for t in range(len(u))]
        output = self.session.run(loc + scale, feed_dict=feed_dict)

        out_mu = np.stack(output[:len(u)],
                          axis=0)
        out_std = np.stack(output[len(u):2 * len(u)],
                           axis=0)
        return out_mu, out_std

    def estimate_loss_y(self, x0, u, target):
        """Use the test model to estimate the loss of the system trajectory

        Args:
            x0 (np.Array): initial state of the system.
            u (list(np.Array)): list with the control signals indexed by time
                step.

        Returns:
            (float32): estimated loss
        """
        feed_dict = dict()
        feed_dict[self.test.x0[0].loc.base] = x0
        for k in range(len(self.test.y)):
            feed_dict[self.test.inputs[k].base] = u[k]
        feed_dict[self.test.target] = target

        loss_op = self.test.loss_y.value
        loss = self.session.run(loss_op, feed_dict=feed_dict)
        return loss

    def visualize_predictions(self, x0, u, real=None, ax=None):
        mu, std = self.estimate_trajectory(x0=x0, u=u)
        n_states = mu.shape[1]
        if ax is None:
            fig, ax = plt.subplots(n_states, 1)
        for i in range(n_states):
            ax[i].plot(mu[:, i])
            ax[i].plot(real[:, i], 'k+')
            ax[i].plot(mu[:, i] + std[:, i], 'r')
            ax[i].plot(mu[:, i] - std[:, i], 'r')
        return mu, std

    @tdl.core.EagerMethod
    def cr_zscore(self, value):
        mc_loc = [out.samples.mean for out in self.test.y[:-1]]
        mc_scale = [out.samples.stddev for out in self.test.y[:-1]]
        samples = [tf.placeholder(tf.float32, shape=mc_loc[0].shape)
                   for t in range(len(mc_loc))]
        with tf.name_scope('zscore'):
            z_score = [(samples[t] - mc_loc[t]) / mc_scale[t]
                       for t in range(len(mc_loc))]
            z_score = tf.stack(z_score, axis=0)
        with tf.name_scope('containing_ratio'):
            _zscore = tf.sqrt(tf.reduce_sum(z_score**2.0, axis=1))
            cr1 = tf.reduce_mean(tf.cast(_zscore < 1.0, tf.float32))
            cr2 = tf.reduce_mean(tf.cast(_zscore < 2.0, tf.float32))
            cr3 = tf.reduce_mean(tf.cast(_zscore < 3.0, tf.float32))

        self._cr_zscore = SimpleNamespace()
        self._cr_zscore.cr = [cr1, cr2, cr3]
        self._cr_zscore.zscore = z_score
        self._cr_zscore.samples = samples

    @cr_zscore.eval
    def cr_zscore(self, dataset):
        x0, target, inputs = self.model_interface.next_batch_in_format(
            dataset=dataset,
            window_size=self.test.window_size,
            n_steps=len(self.test.y) - 1,
            batch_size=1)
        target = [np.squeeze(y) for y in target]
        feed_dict = self.feed_test(x0, inputs)
        for yi, yip in zip(target, self._cr_zscore.samples):
            feed_dict[yip] = yi
        cr1, cr2, cr3, zscore = self.session.run(
            [self._cr_zscore.cr[0],
             self._cr_zscore.cr[1],
             self._cr_zscore.cr[2],
             self._cr_zscore.zscore], feed_dict=feed_dict)
        return (cr1, cr2, cr3), zscore

    def _define_loss_y(self, y_list, target):
        """ defines the section of the loss depending on the obserbable
        variables (sensors). The entire loss is assumend to take the format:
        loss = sum(y_loss(yt)) + sum(u_loss(ut))

        Args:
            yt ([tf.tensor]): list with the sensor readings.

        Returns:
            tdl.Loss: Loss corresponding to the given yt.
        """
        def step_loss(yt, t, T):
            losses = list()
            qy = self.options['shooting/loss/y_t/gain']
            # deviation
            if t >= T - 10:
                deviation = tdl.losses.QuadraticLoss(
                    x=yt.samples.value, q=qy, target=target)
                losses.append(deviation.mean())
            # bounds
            y0_min = self.options['shooting/loss/less_than/y0']
            if y0_min is not None:
                bound = tdl.losses.LessThan(
                    x=yt.samples.value, reference=y0_min,
                    mask=np.array([[1.0, 0.0]]))
                losses.append(bound.mean())
            if losses:
                return (tdl.losses.AddNLosses(losses) if len(losses) > 1
                        else losses[0])
            else:
                return None

        loss_yt = [step_loss(yt, t, len(y_list))
                   for t, yt in enumerate(y_list)]
        loss_yt = [lt for lt in loss_yt if lt is not None]
        loss_y = (tdl.losses.AddNLosses(loss_yt) if len(loss_yt) > 1
                  else loss_yt[0])
        return loss_y

    @tdl.core.EagerMethod
    def eval_loss_y(self, value):
        n_outputs = self.mc_estimate.loss.target.shape[1]
        y = [SimpleNamespace(samples=tdl.bayesnet.McEstimate(
            value=tf.placeholder(tf.float32, shape=(None, n_outputs))))
            for t in range(self.options['shooting/mc_estimate/n_steps'])]
        self._eval_loss_y = SimpleNamespace()
        self._eval_loss_y.target = tf.placeholder(tf.float32,
                                                  shape=(1, n_outputs))
        self._eval_loss_y.loss = self._define_loss_y(
            y_list=y, target=self._eval_loss_y.target)
        self._eval_loss_y.inputs = [yt.samples.value for yt in y]

    @eval_loss_y.eval
    def eval_loss_y(self, y, target):
        """evaluate the loss_y function for the specified trajectory y.

        Args:
            y (np.Array): trajectory of the sensor output.
            target (np.Array): target of the loss.

        Returns:
            float: value for loss_y
        """
        y = [np.expand_dims(y[t, :], axis=0) for t in range(y.shape[0])]
        feed_dict = {self._eval_loss_y.inputs[t]: y[t]
                     for t in range(len(self._eval_loss_y.inputs))}
        feed_dict[self._eval_loss_y.target] = target
        loss = self.session.run(self._eval_loss_y.loss.value,
                                feed_dict=feed_dict)
        return loss

    def current_loss(self, x0, target):
        feed_dict = {self.mc_estimate.x0[0].loc.base: x0,
                     self.mc_estimate.loss.target: target}
        loss = self.session.run(tf.convert_to_tensor(self.mc_estimate.loss),
                                feed_dict=feed_dict)
        return loss

    def get_inputs_value(self):
        u = self.mc_estimate.inputs
        return self.session.run(u)

    def _init_shooting(self):
        ''' Define the model that performs the trajectory optimization '''
        if self.options['shooting/constraints/u_t'] is not None:
            u_bounds = self.options['shooting/constraints/u_t']
            u_in = list()
            u_mean = self.options['shooting/init/u_t/mean']
            u_stddev = self.options['shooting/init/u_t/stddev']
            for k in range(self.options['shooting/mc_estimate/n_steps']):
                # inputs_k = tdl.BoundedVariable(
                #    min=u_bounds[0], max=u_bounds[1],
                #    initializer=lambda mu: tf.random_normal(
                #        shape=(1, self.model.n_inputs),
                #        mean=mu, stddev=u_stddev),
                #    initial_value=u_mean,
                #    name='inputs_{}'.format(k))
                initial_value = tf.random_normal(
                    shape=(1, self.model.n_inputs),
                    mean=u_mean, stddev=u_stddev)
                inputs_k = tdl.core.ConstrainedVariable(
                    initial_value=initial_value, min=u_bounds[0], max=u_bounds[1])
                u_in.append(inputs_k)
            reset_op = tdl.variables_initializer(u_in)

            mc_shooting = self.model.mc_evaluate(
                inputs=u_in,
                n_unrollings=self.options['shooting/mc_estimate/n_steps'],
                n_particles=self.options['shooting/mc_estimate/n_particles'],
                name='mc_test')
            mc_shooting.reset_inputs = reset_op
        else:
            mc_shooting = self.model.mc_evaluate(
                n_unrollings=self.options['shooting/mc_estimate/n_steps'],
                n_particles=self.options['shooting/mc_estimate/n_particles'],
                options={'inputs/type': 'variable',
                         'inputs/mean': self.options['shooting/init/u_t/mean'],
                         'inputs/std':
                         self.options['shooting/init/u_t/stddev'],
                         'inputs/shape': [1, self.model.n_inputs]},
                name='mc_test')

        # Compute quadratic loss
        qu = self.options['shooting/loss/u_t/gain']
        with tf.name_scope(mc_shooting.scope):
            with tf.name_scope('loss'):
                y_T = mc_shooting.y[-1].samples.value
                target = tf.placeholder(tf.float32, shape=[1, y_T.shape[1]])
                loss_y = self._define_loss_y(mc_shooting.y, target)
                loss_ut = [tdl.losses.QuadraticLoss(u_t.base, q=qu).mean()
                           for u_t in mc_shooting.inputs]
                loss_u = tdl.losses.AddNLosses(loss_ut)
                mc_shooting.loss_y = loss_y
                mc_shooting.loss = loss_y + loss_u
                mc_shooting.loss.target = target
        return mc_shooting

    def _init_monitor(self, logger_path):
        monitor = tdlm.TrainingMonitorManager(
            log_folder=logger_path,
            tf_graph=self.session.graph)
        monitor.train.add_monitor(
            tdlm.OpMonitor(
                self.mc_estimate.loss.value,
                buffer_size=self.options['monitor/loggers/buffer_size'],
                name="shooting/loss"))
        return monitor

    def _init_optimizer(self):
        learning_rate = self.options['shooting/learning_rate']
        var_list = [tdl.get_trainable(u_t) for u_t in self.mc_estimate.inputs]
        optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = optim.OptimizationManager(
            session=self.session,
            optimizer=optimizer,
            step_op=optimizer.minimize(
                loss=tf.convert_to_tensor(self.mc_estimate.loss),
                var_list=var_list),
            monitor_manager=self.monitor,
            n_logging=self.options['shooting/loggers/n_logging'],
            saver=optim.EarlyStopping(
                monitor=self.monitor.train['shooting/loss'],
                var_list=tdl.core.get_trainable(self.mc_estimate.inputs),
                logger_path=self.logger_path,
                session=self.session))
        return optimizer

    def __init__(self, model_interface, options, logger_path, session):
        self.model_interface = model_interface
        self.model = model_interface.model
        self.logger_path = logger_path
        self._options = options
        self.session = session

        self.mc_estimate = self._init_shooting()
        self.test = self._init_test_model()
        self.monitor = self._init_monitor(logger_path)
        self.optimizer = self._init_optimizer()
        # TODO: Initialize only local variables
        tf.global_variables_initializer().run()
        print('TF Variables Initialized')
        super(ShootingOptim, self).__init__()


#  ██████  ███    ██ ██      ██ ███    ██ ███████
# ██    ██ ████   ██ ██      ██ ████   ██ ██
# ██    ██ ██ ██  ██ ██      ██ ██ ██  ██ █████
# ██    ██ ██  ██ ██ ██      ██ ██  ██ ██ ██
#  ██████  ██   ████ ███████ ██ ██   ████ ███████

class OnlineTrajOptim(object):
    @property
    def shooting(self):
        return self.main.shooting

    @property
    def ml_model(self):
        return self.main.ml_model

    @property
    def plant(self):
        return self.main.plant

    @property
    def dataset(self):
        return self.main.dataset

    def _init_options(self, options):
        default = {'online/n_iter': 10,
                   'online/train/init/n_iter': 2,
                   'online/train/n_iter': 2,
                   'online/shooting/n_iter': 1,
                   'online/train/exploration/n_trajectories': 10,
                   'online/valid/exploration/n_trajectories': 10,
                   'online/train/exploration/std': 9,
                   'online/valid/exploration/std': 9,
                   'online/trajectory/filename': None}
        options = tdl.core.check_defaults(options, default)
        return options

    def _init_monitor(self, logger_path):
        monitor = tdlm.MonitorManager(
            log_folder=os.path.join(logger_path))

        list(map(lambda name: monitor.add_monitor(
            tdlm.TrainingMonitor(name=name)),
            ["online/loss/expected",
             "online/loss/real",
             "online/loss/meandev",
             "online/x0/expected",
             "online/x1/expected",
             "online/x2/expected",
             "online/x3/expected",
             "online/x0/stddev/T",
             "online/x1/stddev/T",
             "online/x2/stddev/T",
             "online/x3/stddev/T",
             "online/x0/stddev/mean",
             "online/x1/stddev/mean",
             "online/x2/stddev/mean",
             "online/x3/stddev/mean",
             "online/x0/real",
             "online/x1/real",
             "online/x2/real",
             "online/x3/real",
             "online/deviation/mean",
             "online/deviation/max",
             "online/stddev/mean",
             "online/stddev/max"]))

        monitor.add_monitor(
            tdlm.ImageMonitor(name="online/trajectory"))
        return monitor

    def reject_trajectory(self, step):
        x0 = np.array([self.plant.initial_state])
        target = self.options['shooting/loss/target']
        loss = self.shooting.current_loss(x0, target)
        print('current loss:', loss)
        if np.isnan(loss):
            return True

        if step < 5:
            return loss > 2500.0
        else:
            return loss > 1000.0

    def run(self, max_iter=None):
        plt_monitor = self.monitor['online/trajectory']
        for i in range(self.options['online/train/init/n_iter']):
            self.main.run_training()
            # self.main.ml_model.fit()

        max_iter = (max_iter if max_iter is not None
                    else self.options['online/n_iter'])
        for step in range(max_iter):
            # 1. train the ml model
            for j in range(self.options['online/train/n_iter']):
                self.main.run_training()
            # save trainable
            self.main.ml_model.save(step)
            # save dataset
            self.main.save_dataset('dataset_{}.dat'.format(step))

            # 3. trajectory optimization
            for j in range(self.options['online/shooting/n_iter']):
                x0 = np.array([self.plant.initial_state])
                suboptimal_u = self.main.run_shooting()

            _count = 0
            while (self.reject_trajectory(step) and (_count < 3)):
                print('-->Trajectory rejected')
                self.main.shooting.mc_estimate.reset_inputs.run()
                for j in range(self.options['online/shooting/n_iter']):
                    x0 = np.array([self.plant.initial_state])
                    suboptimal_u = self.main.run_shooting()
                _count += 1

            # 4. simulate and visualize trajectory
            sim_data = self.plant.simulate(
                policy=lambda x, k: suboptimal_u[k],
                steps=len(suboptimal_u))
            x_names = [x for x in sim_data.keys() if 'x' in x]
            real_x = sim_data[x_names].values
            real_x0 = real_x[-1, 0]
            real_x1 = real_x[-1, 1]
            real_x2 = real_x[-1, 2]
            real_x3 = real_x[-1, 3]
            real_cost = self.main.shooting.eval_loss_y(
                y=real_x,
                target=self.options['shooting/loss/target'])
            print('final x: {}, real cost: {}'
                  ''.format(real_x[-1, :], real_cost))

            # get estimations
            fig, ax = plt.subplots(
                nrows=self.main.ml_model.model.n_outputs,
                ncols=1,
                figsize=(10, 3 * self.main.ml_model.model.n_outputs))
            mc_estimate_mean, mc_estimate_stddev = \
                self.shooting.visualize_predictions(
                    x0=x0, u=suboptimal_u, real=real_x[1:, :], ax=ax)

            deviation = np.mean((np.fabs(real_x[1:, :] - mc_estimate_mean[:-1, :])) /
                                mc_estimate_stddev[:-1, :],
                                axis=1)
            deviation_mean = np.mean(deviation)
            deviation_max = np.max(deviation)
            stddev_mean = np.mean(mc_estimate_stddev)
            stddev_max = np.max(mc_estimate_stddev)

            estimated_x0 = mc_estimate_mean[-2, 0]
            estimated_x1 = mc_estimate_mean[-2, 1]
            estimated_x2 = mc_estimate_mean[-2, 2]
            estimated_x3 = mc_estimate_mean[-2, 3]
            stddev_x0 = mc_estimate_stddev[-2, 0]
            stddev_x1 = mc_estimate_stddev[-2, 1]
            stddev_x2 = mc_estimate_stddev[-2, 2]
            stddev_x3 = mc_estimate_stddev[-2, 3]
            stddev_mean_x0 = np.mean(mc_estimate_stddev[:, 0])
            stddev_mean_x1 = np.mean(mc_estimate_stddev[:, 1])
            stddev_mean_x2 = np.mean(mc_estimate_stddev[:, 2])
            stddev_mean_x3 = np.mean(mc_estimate_stddev[:, 3])
            estimated_cost = self.main.shooting.estimate_loss_y(
                x0=x0, u=suboptimal_u,
                target=self.options['shooting/loss/target'])

            l2loss = np.sum((real_x[1:, :] - mc_estimate_mean[:-1, :])**2,
                            axis=1)
            l2loss = np.mean(l2loss)
            # l2loss = np.mean(np.sqrt(l2loss))
            # log online control learning results
            self.monitor.feed(
                {"online/loss/expected": estimated_cost,
                 "online/loss/real": real_cost,
                 "online/loss/meandev": l2loss,
                 "online/x0/expected": estimated_x0,
                 "online/x1/expected": estimated_x1,
                 "online/x2/expected": estimated_x2,
                 "online/x3/expected": estimated_x3,
                 "online/x0/stddev/T": stddev_x0,
                 "online/x1/stddev/T": stddev_x1,
                 "online/x2/stddev/T": stddev_x2,
                 "online/x3/stddev/T": stddev_x3,
                 "online/x0/stddev/mean": stddev_mean_x0,
                 "online/x1/stddev/mean": stddev_mean_x1,
                 "online/x2/stddev/mean": stddev_mean_x2,
                 "online/x3/stddev/mean": stddev_mean_x3,
                 "online/x0/real": real_x0,
                 "online/x1/real": real_x1,
                 "online/x2/real": real_x2,
                 "online/x3/real": real_x3,
                 "online/deviation/mean": deviation_mean,
                 "online/deviation/max": deviation_max,
                 "online/stddev/mean": stddev_mean,
                 "online/stddev/max": stddev_max,
                 "online/trajectory": plt_monitor.mplfig2tfsummary(fig)},
                step)
            self.monitor.write_data()

            # save trajectory if is the one with minimum loss
            def best_controls(current_loss):
                best_loss = self.monitor['online/loss/real'].min
                return abs(current_loss - best_loss) < 1e-7
            if best_controls(real_cost):
                np.save(os.path.join(self.logger_path, 'optimal_u'),
                        suboptimal_u)
            np.save(os.path.join(self.logger_path,
                                 'optimal_u_step{}'.format(step)),
                    suboptimal_u)

            # 5. add new runs to the dataset
            sim_data = self.plant.simulate(lambda x, k: suboptimal_u[k],
                                           steps=len(suboptimal_u))
            self.dataset.train.add_record(tsdataset.Record(sim_data))

            # Sample trajectories and add them to the dataset
            def explore_controller(x, k, dataset):
                assert dataset in ('train', 'valid'),\
                    'invalid dataset argument, options are: '\
                    '{\'train\', \'valid\'}'
                scale = (self.options['online/train/exploration/std']
                         if dataset == 'train'
                         else self.options['online/valid/exploration/std'])
                noise = np.random.normal(
                    loc=0.0, scale=scale, size=1)
                return suboptimal_u[k] + noise

            for t in range(self.options['online/train/exploration'
                                        '/n_trajectories']):
                sim_data = self.plant.simulate(
                    policy=lambda x, k: explore_controller(x, k, 'train'),
                    steps=len(suboptimal_u))
                self.dataset.train.add_record(tsdataset.Record(sim_data))

            for t in range(self.options['online/valid/exploration'
                                        '/n_trajectories']):
                sim_data = self.plant.simulate(
                    policy=lambda x, k: explore_controller(x, k, 'valid'),
                    steps=len(suboptimal_u))
                self.dataset.valid.add_record(tsdataset.Record(sim_data))
                self.dataset.test.add_record(tsdataset.Record(sim_data))
            # reset controls
            # if (real_x < 100.0).any():  # TODO: delete this for cartpole
            #    print('Resetting controls')
            #    self.main.shooting.mc_estimate.reset_inputs.run()
            # reset dataset
            print('Training samples:', self.dataset.train.n_samples)
            print('Valid samples:', self.dataset.valid.n_samples)
        # Clean up
        self.monitor.flush()

    def __init__(self, main, options, logger_path, session):
        self.main = main
        self.logger_path = logger_path
        self.options = self._init_options(options)
        self.monitor = self._init_monitor(logger_path)


# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████

class TrajectoryOptimization(Supervised):

    def _init_options(self, options):
        default = {'plant/name': 'cartpole',
                   'dataset/init/n_trajectories': 100,
                   'dataset/init/exploration_std': 0.3,
                   'dataset/init/n_steps': 1000,
                   'dataset/normalize': False,
                   'saver/options/file': 'main_options.pkl',
                   \
                   'optim/train/max_iter': 2000,
                   'optim/valid/eval_freq': 5,
                   'optim/train/learning_rate': 0.001,
                   \
                   'shooting/optim/max_iter': 1000,
                   'shooting/loss/less_than/y0': 100,
                   'shooting/loss/y_t/gain': np.array([1.0, 1.0, 1.0, 1.0]),
                   'shooting/loss/u_t/gain': np.array([0.001]),
                   'shooting/loss/target': np.array(0),
                   'shooting/init/u_t/mean': 0.0,
                   'shooting/init/u_t/stddev': 0.1,
                   'shooting/learning_rate': 0.001,
                   'shooting/loggers/n_logging': 100,
                   'shooting/mc_estimate/n_steps': 100,
                   'shooting/mc_estimate/n_particles': 500,
                   'shooting/constraints/u_t': [200, 350],
                   \
                   'narx/normalize': True,
                   'narx/window_size': 1,
                   'narx/cell/n_hidden': [100, 100],
                   'narx/train/batch_size': 1000,
                   'narx/train/n_steps': 20,
                   'narx/test/n_steps': 30,
                   'narx/test/n_particles': 500,
                   'narx/reparameterization/type': 'local_gaussian',
                   'narx/cell/loc/prior/stddev': 100.0,
                   'narx/cell/scale/prior/stddev': 100.0,
                   'narx/noise_x/heteroscedastic': False}
        options = tdl.core.check_defaults(options, default)
        options = super(TrajectoryOptimization, self)._init_options(options)
        return options

    def run_shooting(self, x0=None, target=None, n_optim_steps=None):
        ''' Performs stochastic trajectory optimization by optimizing
        the inputs 'u' on the network described by self.ml_model
        '''
        x0 = (x0 if x0 is not None
              else np.array([self.plant.initial_state]))
        target = (target if target is not None
                  else self.options['shooting/loss/target'])
        n_optim_steps = (n_optim_steps if n_optim_steps is not None
                         else self.options['shooting/optim/max_iter'])

        def feed_train():
            model = self.shooting.mc_estimate
            feed_dict = {model.x0[0].loc.base: x0,
                         model.loss.target: target}
            return feed_dict

        # Optimization loop
        print("Performing trajectory optimization using control model")
        t0 = time()
        self.shooting.optimizer.run(n_optim_steps,
                                    feed_train=feed_train)
        t1 = time()
        print('Trajectory optimizer took: ', (t1 - t0), 's')

        # return the optimized inputs
        # TODO: THIS LOOKS LIKE AN ERROR
        u_p = self.shooting.session.run(
            [tdl.get_trainable(u_t)[0].value()
             for u_t in self.shooting.mc_estimate.inputs])
        # u_p = np.squeeze(np.concatenate(u_p, axis=0))
        return u_p

    def run_online(self, max_iter=None):
        return self.online.run(max_iter)

    def __init__(self, options=None, tmp_path='tmp'):
        self._tmp_path = tmp_path
        self._options = self._init_options(options)
        self._plant = self._init_plant()
        self._dataset = self._init_dataset()
        self._ml_model = self._init_ml_model(os.path.join(self.tmp_path,
                                                          'ml_model'))
        self.shooting = ShootingOptim(model_interface=self.ml_model,
                                      options=self.options,
                                      logger_path=os.path.join(self.tmp_path,
                                                               'shooting'),
                                      session=self.ml_model.session)
        self.online = OnlineTrajOptim(main=self,
                                      options=self.options,
                                      logger_path=os.path.join(self.tmp_path,
                                                               'online'),
                                      session=self.ml_model.session)
        tdl.core.TdlProgram.__init__(self)
        # monitors
        self.monitors = [self.ml_model.monitor,
                         self.ml_model.test_monitor,
                         self.shooting.monitor,
                         self.online.monitor]
        # save options
        _options_path = os.path.join(self.tmp_path,
                                     self.options['saver/options/file'])
        with open(_options_path, 'wb') as handle:
            pickle.dump(self.options, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_default_options(model='cstr', heteroscedastic=False,
                        reparameterization='local_gaussian'):
    test_options = {
        'plant/name': 'cstr',
        'plant/dt': 0.5,
        'dataset/init/n_trajectories': 100,
        'dataset/init/exploration_std': None,
        'dataset/init/n_steps': 45,
        \
        'optim/train/learning_rate': 0.0005,
        \
        'online/n_iter': 1,
        'online/train/init/n_iter': 1,
        'online/train/n_iter': 1,
        'online/shooting/n_iter': 1,
        'online/train/exploration/n_trajectories': 10,
        'online/valid/exploration/n_trajectories': 10,
        'online/train/exploration/std': 9,
        'online/valid/exploration/std': 9,
        \
        'shooting/learning_rate': 0.1,
        'shooting/loss/target': np.array([[400, 280]]),
        'shooting/loss/less_than/y0': 100,
        'shooting/loss/y_t/gain': np.array([10.0, 10.0]),
        'shooting/loss/u_t/gain': np.array([0.01]),
        'shooting/init/u_t/mean': 250.0,
        'shooting/init/u_t/stddev': 1.0,
        'shooting/mc_estimate/n_steps': 200,
        'shooting/mc_estimate/n_particles': 500,
        'shooting/constraints/u_t': [200, 350],
        \
        'narx/normalize': True,
        'narx/train/n_steps': 10,
        'narx/cell/loc/n_hidden': [50, 50, 50],
        'narx/cell/loc/prior/stddev': 100.0,
        'narx/cell/scale/prior/stddev': 100.0,
        'narx/cell/scale/lower_bound': 1e-7,
        'narx/cell/scale/upper_bound': None,
        'narx/noise_x/heteroscedastic': heteroscedastic
    }
    cstr_options = {
        'plant/name': 'cstr',
        'plant/dt': 0.5,
        'dataset/init/n_trajectories': 100,
        'dataset/init/exploration_std': None,
        'dataset/init/n_steps': 45,
        \
        'optim/train/learning_rate': 0.0005,
        \
        'online/n_iter': 10,
        'online/train/init/n_iter': 1,
        'online/train/n_iter': 2,
        'online/shooting/n_iter': 2,
        'online/train/exploration/n_trajectories': 10,
        'online/valid/exploration/n_trajectories': 10,
        'online/train/exploration/std': 9,
        'online/valid/exploration/std': 9,
        \
        'shooting/learning_rate': 0.1,
        'shooting/loss/target': np.array([[400, 280]]),
        'shooting/loss/less_than/y0': 100,
        'shooting/loss/y_t/gain': np.array([10.0, 10.0]),
        'shooting/loss/u_t/gain': np.array([0.01]),
        'shooting/init/u_t/mean': 250.0,
        'shooting/init/u_t/stddev': 1.0,
        'shooting/mc_estimate/n_steps': 200,
        'shooting/mc_estimate/n_particles': 500,
        'shooting/constraints/u_t': [200, 350],
        \
        'narx/normalize': True,
        'narx/train/n_steps': 10,
        'narx/cell/loc/n_hidden': [50, 50, 50],
        'narx/cell/loc/prior/stddev': 100.0,
        'narx/cell/scale/prior/stddev': 100.0,
        'narx/cell/scale/lower_bound': 1e-7,
        'narx/cell/scale/upper_bound': None,
        'narx/reparameterization/type': reparameterization,
        'narx/noise_x/heteroscedastic': heteroscedastic
    }
    cartpole_options = {
        'plant/name': 'cartpole',
        'plant/dt': 0.5,
        'dataset/init/n_trajectories': 50,
        'dataset/init/exploration_std': 0.35,
        'dataset/init/n_steps': 50,
        \
        'optim/train/learning_rate': 0.002,
        'optim/train/max_iter': 3001,
        'optim/valid/eval_freq': 5,
        \
        'online/n_iter': 10,
        'online/train/init/n_iter': 0,
        'online/train/n_iter': 1,
        'online/shooting/n_iter': 1,
        'online/train/exploration/n_trajectories': 15,
        'online/valid/exploration/n_trajectories': 15,
        'online/train/exploration/std': 0.1,
        'online/valid/exploration/std': 0.1,
        \
        'shooting/loss/target': np.array([[0.0, 0.0, 0.0, 0.0]]),
        'shooting/loss/less_than/y0': None,
        'shooting/loss/y_t/gain': np.array([1.0, 20.0, 0.1, 0.1]),
        'shooting/loss/u_t/gain': np.array([0.1]),
        'shooting/init/u_t/mean': 0.0,
        'shooting/init/u_t/stddev': 0.01,
        'shooting/learning_rate': 0.01,
        'shooting/optim/max_iter': 3000,
        'shooting/mc_estimate/n_steps': 50,
        'shooting/mc_estimate/n_particles': 1000,
        'shooting/constraints/u_t': [-1.0, 1.0],
        \
        'narx/normalize': True,
        'narx/train/n_steps': 10,
        'narx/cell/loc/n_hidden': [100, 100],
        'narx/cell/loc/afunction': tf.nn.softplus,
        'narx/cell/loc/keep_prob': [1.0, 0.9, 0.9],
        'narx/cell/scale/n_hidden': [50],
        'narx/cell/scale/keep_prob': [1.0, 1.0],
        'narx/cell/scale/afunction': tf.nn.softplus,
        'narx/cell/scale/lower_bound': 1e-7,
        'narx/cell/scale/upper_bound': 1.0,
        'narx/reparameterization/type': reparameterization,
        'narx/cell/loc/prior/stddev': 1.0/np.sqrt(0.5),
        'narx/cell/scale/prior/stddev': 1.0/np.sqrt(0.5),
        'narx/noise_x/heteroscedastic': heteroscedastic
    }
    acrobot_options = {
        'plant/name': 'acrobot',
        'plant/dt': 0.5,
        'dataset/init/n_trajectories': 50,
        'dataset/init/exploration_std': 0.35,
        'dataset/init/n_steps': 100,
        \
        'optim/train/learning_rate': 0.002,
        'optim/train/max_iter': 3001,
        'optim/valid/eval_freq': 4,
        \
        'online/n_iter': 10,
        'online/train/init/n_iter': 0,
        'online/train/n_iter': 1,
        'online/shooting/n_iter': 1,
        'online/train/exploration/n_trajectories': 15,
        'online/valid/exploration/n_trajectories': 15,
        'online/train/exploration/std': 0.05,
        'online/valid/exploration/std': 0.05,
        \
        'shooting/loss/target': np.array([[0.0, 0.0, 0.0, 0.0]]),
        'shooting/loss/less_than/y0': None,
        'shooting/loss/y_t/gain': np.array([25.0, 2.0, 0.05, 0.05]),
        'shooting/loss/u_t/gain': np.array([0.1]),
        'shooting/init/u_t/mean': 0.0,
        'shooting/init/u_t/stddev': 0.3,
        'shooting/learning_rate': 0.002,
        'shooting/optim/max_iter': 3000,
        'shooting/mc_estimate/n_steps': 110,
        'shooting/mc_estimate/n_particles': 1000,
        'shooting/constraints/u_t': [-1.0, 1.0],
        \
        'narx/normalize': True,
        'narx/train/n_steps': 10,
        'narx/cell/loc/n_hidden': [350, 350],
        'narx/cell/loc/afunction': tf.nn.relu,
        'narx/cell/loc/keep_prob': [1.0, 0.9, 0.9],
        'narx/cell/scale/n_hidden': [50],
        'narx/cell/scale/keep_prob': [1.0, 1.0],
        'narx/cell/scale/afunction': tf.nn.relu,
        'narx/cell/scale/lower_bound': 1e-7,
        'narx/cell/scale/upper_bound': 1,
        'narx/reparameterization/type': reparameterization,
        'narx/cell/loc/prior/stddev': 1.0/np.sqrt(0.5),
        'narx/cell/scale/prior/stddev': 1.0/np.sqrt(0.5),
        'narx/noise_x/heteroscedastic': heteroscedastic
    }

    if reparameterization == 'local_gaussian':
        cartpole_options['online/train/exploration/std'] = 0.05,
        cartpole_options['online/valid/exploration/std'] = 0.05,
        cartpole_options['online/train/init/n_iter'] = 1
        cartpole_options['online/train/n_iter'] = 2
    elif reparameterization == 'dropout':
        # cartpole_options['narx/cell/loc/afunction'] = tf.nn.softplus
        # cartpole_options['narx/cell/scale/afunction'] = tf.nn.softplus
        cartpole_options['narx/cell/loc/afunction'] = tf.nn.relu
        cartpole_options['narx/cell/scale/afunction'] = tf.nn.relu
    else:
        raise AttributeError('reparameterization option {} not valid'
                             ''.format(reparameterization))

    model_dict = {
        'test': test_options,
        'cstr': cstr_options,
        'cartpole': cartpole_options,
        'acrobot': acrobot_options
    }
    if model not in model_dict:
        raise AttributeError('invalid model {}. Available options are:{}'
                             ''.format(model, model_dict.keys()))
    return model_dict[model]


class RnnBayesnetTest(unittest.TestCase):
    def test_options(self):
        options = {
            'narx/normalize': False,
            'optim/train/learning_rate': 0.001005
        }
        main = Supervised(options=options,
                          tmp_path=TMP_PATH)
        assert (main.ml_model.options['optim/train/learning_rate'] ==
                options['optim/train/learning_rate']), \
            'Options provided to the main class did not correctly '\
            'propagate'
        main.run_training()
        loss = main.ml_model.monitor.train['train/loss'].current_value
        assert np.isfinite(loss), 'training loss is not finite'
        loss = main.ml_model.monitor.train['train/loss'].mean()
        assert (loss < -1.9) and (loss > -3.1),\
            'loss value is outside the expected range'

        vars1 = set(tdl.core.get_trainable(main.ml_model.model))
        vars2 = set(tf.trainable_variables(main.ml_model.model.scope))
        assert vars1 == vars2, \
            'variables found using get_trainable are different from '\
            'the ones found with tf.trainable_variables'

    def test_dropout(self):
        options = {
            'narx/normalize': False,
            'optim/train/learning_rate': 0.001005,
            'narx/reparameterization/type': 'dropout'
        }
        main = Supervised(options=options,
                          tmp_path=TMP_PATH)
        assert isinstance(main.ml_model.model.cell,
                          twodlearn.bayesnet.recurrent.NormalNarxCell),\
            'problem with setting up dropout model as an option'
        assert (main.ml_model.options['optim/train/learning_rate'] ==
                options['optim/train/learning_rate']), \
            'Options provided to the main class did not correctly '\
            'propagate'
        main.run_training()
        loss = main.ml_model.monitor\
                            .train['train/loss'].current_value
        assert np.isfinite(loss), 'training loss is not finite'

    def test_loss(self):
        options = {
            'narx/normalize': False,
            'optim/train/learning_rate': 0.001005,
            'narx/reparameterization/type': 'dropout'
        }
        main = Supervised(options=options,
                          tmp_path=TMP_PATH)

        tf.convert_to_tensor(main.ml_model.train.loss)


if __name__ == "__main__":
    unittest.main()
