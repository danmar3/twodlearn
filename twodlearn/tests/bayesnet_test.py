from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest
import numpy as np
import twodlearn.debug
import tensorflow as tf
import matplotlib.pyplot as plt
import twodlearn.templates.supervised
import twodlearn.bayesnet as tdlb
import twodlearn.feedforward as tdlf
import twodlearn.monitoring as tdlm
from twodlearn.datasets.base import Datasets, Dataset
from twodlearn.optim import OptimizationManager

TESTS_PATH = os.path.dirname(os.path.abspath(__file__))
TMP_PATH = os.path.join(TESTS_PATH, 'tmp/')


def create_heteroscedastic_dataset(n_samples=1000):
    sigma_r = 1.0 * np.exp(-0.005 * np.linspace(0, 1000, n_samples))
    # get linspace for independent variable
    aux_x = np.linspace(0, 6, n_samples)
    # sample output, given x
    aux_y = np.linspace(0, 6, n_samples)
    for i in range(n_samples):
        aux_y[i] = np.random.normal(np.sin(aux_x[i]), sigma_r[i])
    # build the dataset
    data = Datasets(Dataset(aux_x, aux_y))
    data = data.normalize()
    print('Data set created, shape: ', data.train.x.shape)
    return data


def create_homoscedastic_dataset(n_samples=1000):
    sigma_r = 0.3
    # get linspace for independent variable
    aux_x = np.linspace(0, 6, n_samples)
    # sample output, given x
    aux_y = np.linspace(0, 6, n_samples)
    for i in range(n_samples):
        aux_y[i] = np.random.normal(np.sin(aux_x[i]), sigma_r)
    # build the dataset
    data = Datasets(Dataset(aux_x, 10 * aux_y))
    data = data.normalize()
    print('Data set created, shape: ', data.train.x.shape)
    return data


def plot_yp(x, yp_mu, yp_std, ax, dataset=None, title=''):
    if dataset is not None:
        ax.plot(np.squeeze(dataset.x),
                np.squeeze(dataset.y), 'k+', label='samples')
    downsampling = 2
    # mean
    ax.plot(np.squeeze(x[::downsampling]),
            np.squeeze(yp_mu[::downsampling]),
            'b', label='mean')
    # std
    ax.plot(np.squeeze(x[::downsampling]),
            np.squeeze(yp_mu[::downsampling]) +
            np.squeeze(yp_std[::downsampling]),
            'r', label='std')
    ax.plot(np.squeeze(x[::downsampling]),
            np.squeeze(yp_mu[::downsampling]) -
            np.squeeze(yp_std[::downsampling]),
            'r')
    ax.set_title(title, fontsize=18)
    ax.legend(loc='upper right')
    return ax


class BayesModel(twodlearn.templates.supervised.MlModel):
    def _init_options(self, options):
        default = {'n_inputs': 1,
                   'n_outputs': 1,
                   'loc/n_hidden': [10],
                   'scale/n_hidden': [5],
                   'test/n_particles': 1000,
                   'heteroscedastic': False}
        assert 'train/n_samples' in options,\
            'train/n_samples must be specified in the options'
        options = super(BayesModel, self)._init_options(options, default)
        return options

    def _init_model(self):
        if self.options['heteroscedastic'] is True:
            model = tdlb.HeteroscedasticNormalMlp(
                loc_args={'n_inputs': self.options['n_inputs'],
                          'n_outputs': self.options['n_outputs'],
                          'n_hidden': self.options['loc/n_hidden'],
                          'afunction': tdlf.selu01},
                scale_args={'n_hidden': self.options['scale/n_hidden'],
                            'afunction': tdlf.selu01,
                            'lower': 0.05,
                            'upper': None}
            )
        else:
            model = tdlb.NormalMlp(
                loc_args={'n_inputs': self.options['n_inputs'],
                          'n_outputs': self.options['n_outputs'],
                          'n_hidden': self.options['loc/n_hidden'],
                          'afunction': tdlf.selu01},
                LocClass=tdlb.BayesianMlp)
        return model

    def _init_train_model(self):
        inputs = tf.placeholder(tf.float32)
        train = self.model(inputs)
        with tf.name_scope('loss'):
            fit_loss = tdlb.GaussianNegLogLikelihood(train)
            train.labels = fit_loss.labels
            train.fit_loss = fit_loss.value
            reg_loc = tdlb.GaussianKL(
                p=train.loc.weights,
                q=tf.distributions.Normal(loc=0.0, scale=1000.0))

            if self.options['heteroscedastic'] is True:
                reg_scale = tdlb.GaussianKL(
                    p=train.scale.weights,
                    q=tf.distributions.Normal(loc=0.0, scale=100.0))
                train.reg_scale = reg_scale
                train.reg_loss = reg_loc.value + reg_scale.value
            else:
                train.reg_loss = reg_loc.value

            with tf.name_scope('train_loss'):
                train.loss = train.fit_loss + \
                    (1.0/self.options['train/n_samples'])*train.reg_loss
        return train

    def _init_test_model(self):
        test = self.model.mc_evaluate(n_particles=self.options['test/n_particles'],
                                      name='mc_test')
        return test

    def _init_training_monitor(self, logger_path):
        # monitoring
        ml_monitor = tdlm.TrainingMonitorManager(
            log_folder=logger_path,
            tf_graph=self.session.graph)

        ml_monitor.train.add_monitor(
            tdlm.OpMonitor(self.train.loss,
                           name="train/loss"))
        if self.options['heteroscedastic']:
            ml_monitor.train.add_monitor(
                tdlm.OpMonitor(tf.reduce_max(self.train.reg_scale.p.scale),
                               name="weights/max_scale"))
        return ml_monitor


class BayesModel2(BayesModel):
    def _init_model(self):
        loc_options = {'layers/options': {'w/prior/stddev': 1000.0}}
        scale_options = {'layers/options': {'w/prior/stddev': 100.0}}
        if self.options['heteroscedastic'] is True:
            model = tdlb.HeteroscedasticNormalMlp(
                loc_args={'n_inputs': self.options['n_inputs'],
                          'n_outputs': self.options['n_outputs'],
                          'n_hidden': self.options['loc/n_hidden'],
                          'afunction': tdlf.selu01,
                          'options': loc_options},
                scale_args={'n_hidden': self.options['scale/n_hidden'],
                            'afunction': tdlf.selu01,
                            'lower': 0.05,
                            'upper': None,
                            'options': scale_options}
            )
        else:
            model = tdlb.NormalMlp(
                loc_args={'n_inputs': self.options['n_inputs'],
                          'n_outputs': self.options['n_outputs'],
                          'n_hidden': self.options['loc/n_hidden'],
                          'afunction': tdlf.selu01,
                          'options': loc_options},
                LocClass=tdlb.BayesianMlp)
        return model

    def _init_train_model(self):
        inputs = tf.placeholder(tf.float32)
        train = self.model(inputs)
        with tf.name_scope('loss'):
            train.fit_loss = tdlb.GaussianNegLogLikelihood(train)
            train.labels = train.fit_loss.labels
            train.reg_loss = self.model.regularizer.init()

            with tf.name_scope('train_loss'):
                train.loss = train.fit_loss.value + \
                    (1.0/self.options['train/n_samples'])*train.reg_loss.value
        return train

    def _init_training_monitor(self, logger_path):
        # monitoring
        ml_monitor = tdlm.TrainingMonitorManager(
            log_folder=logger_path,
            tf_graph=self.session.graph)

        ml_monitor.train.add_monitor(
            tdlm.OpMonitor(self.train.loss,
                           name="train/loss"))
        if self.options['heteroscedastic']:
            ml_monitor.train.add_monitor(
                tdlm.OpMonitor(tf.reduce_max(self.train.reg_loss.loss2.losses[0].p.scale),
                               name="weights/max_scale"))
        return ml_monitor


class Supervised(twodlearn.templates.supervised.Supervised):
    MlModel = BayesModel

    def _init_dataset(self):
        if self.options['heteroscedastic'] is True:
            dataset = create_heteroscedastic_dataset(
                n_samples=self.options['train/n_samples'])
        else:
            dataset = create_homoscedastic_dataset(
                n_samples=self.options['train/n_samples'])
        return dataset

    def _init_ml_model(self, logger_path):
        model = self.MlModel(options=self.options,
                             logger_path=self.tmp_path)
        return model

    def feed_train(self):
        feed_dict = {self.ml_model.train.inputs:
                     np.expand_dims(self.dataset.train.x, 1),
                     self.ml_model.train.labels:
                     np.expand_dims(self.dataset.train.y, 1)}
        return feed_dict

    def visualize(self):
        test_x = np.linspace((1+0.01)*self.dataset.train.x.min(),
                             (1+0.01)*self.dataset.train.x.max(),
                             num=1000)
        test_x = np.expand_dims(test_x, axis=1)

        y_mean = list()
        y_std = list()
        for i in range(test_x.shape[0]):
            m_i, std_i = self.ml_model.session.run(
                [self.ml_model.test.samples.mean,
                 self.ml_model.test.samples.stddev],
                feed_dict={self.ml_model.test.inputs: np.expand_dims(test_x[i],
                                                                     axis=1)})
            y_mean.append(m_i)
            y_std.append(std_i)

        y_mean = np.concatenate(y_mean, 0)
        y_std = np.concatenate(y_std, 0)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        plot_yp(test_x, y_mean, y_std, ax, self.dataset.train)


class Supervised2(Supervised):
    MlModel = BayesModel2


class BayesnetTest(unittest.TestCase):
    def test_options(self):
        layer_opt = {
            'w/stddev/alpha': 1.0
        }
        model = tdlb.NormalMlp(
            loc_args={'n_inputs': 50,
                      'n_outputs': 10,
                      'n_hidden': [5, 5],
                      'options': {'layers/options': layer_opt}})

        assert (model.loc.layers[0].options['w/stddev/alpha'] ==
                layer_opt['w/stddev/alpha']),\
            'Error on setting up options'

    def test_homoscedastic(self):
        options = {'train/n_samples': 1000,
                   'optim/train/max_steps': 1000,
                   'heteroscedastic': False}
        main = Supervised(options=options,
                          tmp_path=TMP_PATH)
        main.run_training()
        loss = main.ml_model.monitor\
                            .train['train/loss'].current_value
        assert np.isfinite(loss), 'training resulted in nan for bayesnet'
        loss = main.ml_model.monitor\
                            .train['train/loss'].mean()
        assert (loss < 2.25) and (loss > 1.0),\
            'loss value is outside the expected range'

    def test_heteroscedastic(self):
        options = {'train/n_samples': 1000,
                   'optim/train/max_steps': 1000,
                   'heteroscedastic': True}
        main = Supervised(options=options,
                          tmp_path=TMP_PATH)
        main.run_training()
        loss = main.ml_model.monitor\
                            .train['train/loss'].current_value
        assert np.isfinite(loss), 'training resulted in nan for bayesnet'
        loss = main.ml_model.monitor\
                            .train['train/loss'].mean()
        assert (loss < -1.1) and (loss > -1.5),\
            'loss value is outside the expected range'

    def test_regularizer(self):
        options = {'train/n_samples': 1000,
                   'optim/train/max_steps': 1000,
                   'heteroscedastic': True}
        main = Supervised(options=options,
                          tmp_path=TMP_PATH)
        scale = 15.3
        reg_loc1 = main.ml_model.model.loc.regularizer.init(prior_stddev=scale)
        reg_loc2 = tdlb.GaussianKL(p=main.ml_model.train.loc.weights,
                                   q=tf.distributions.Normal(loc=0.0, scale=scale))
        r1, r2 = main.ml_model.session.run([reg_loc1.value, reg_loc2.value])
        np.testing.assert_almost_equal(r1, r2, decimal=5),\
            'KL regularizer do not coincide ({}, {})'.format(r1, r2)

    def test_homoscedastic2(self):
        options = {'train/n_samples': 1000,
                   'optim/train/max_steps': 1000,
                   'heteroscedastic': False}
        main = Supervised2(options=options,
                           tmp_path=TMP_PATH)
        main.run_training()
        loss = main.ml_model.monitor\
                            .train['train/loss'].current_value
        assert np.isfinite(loss), 'training resulted in nan for bayesnet'
        loss = main.ml_model.monitor\
                            .train['train/loss'].mean()
        assert (loss < 2.25) and (loss > 1.0),\
            'loss value is outside the expected range'

    def test_heteroscedastic2(self):
        options = {'train/n_samples': 1000,
                   'optim/train/max_steps': 1000,
                   'heteroscedastic': True}
        main = Supervised2(options=options,
                           tmp_path=TMP_PATH)
        main.run_training()
        loss = main.ml_model.monitor\
                            .train['train/loss'].current_value
        assert np.isfinite(loss), 'training resulted in nan for bayesnet'
        loss = main.ml_model.monitor\
                            .train['train/loss'].mean()
        assert (loss < -1.1) and (loss > -1.5),\
            'loss value is outside the expected range'

    def test_normal(self):
        main = tdlb.Normal(shape=[100, 300])
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        test = main.evaluate()

        x, mean1 = sess.run([test.samples.value,
                             test.samples.mean])
        np.testing.assert_almost_equal(np.mean(x), np.mean(mean1))


if __name__ == "__main__":
    unittest.main()
