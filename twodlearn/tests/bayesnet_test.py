from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest
import numpy as np
import tensorflow_probability as tfp
import twodlearn.debug
import tensorflow as tf
import matplotlib.pyplot as plt
import twodlearn as tdl
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


def get_model(dataset, uncertainty):
    if uncertainty == 'homoscedastic':
        model = tdl.bayesnet.NormalModel(
            batch_shape=[None, 1],
            loc=tdl.stacked.StackedLayers(
                layers=[tdl.bayesnet.AffineNormalLayer(
                            units=10, tolerance=1e-5),
                        tf.keras.layers.Activation(tf.nn.softplus),
                        tdl.bayesnet.AffineNormalLayer(
                            units=1, tolerance=1e-5)]))
    elif uncertainty == 'heteroscedastic':
        model = tdl.bayesnet.NormalModel(
            batch_shape=[None, 1],
            loc=tdl.stacked.StackedLayers(
                layers=[tdl.bayesnet.AffineNormalLayer(
                            units=5, tolerance=1e-5),
                        tf.keras.layers.Activation(tf.nn.softplus),
                        tdl.bayesnet.AffineNormalLayer(
                            units=1, tolerance=1e-5)]),
            scale=tdl.stacked.StackedLayers(
                layers=[tdl.bayesnet.AffineNormalLayer(
                            units=3, tolerance=1e-3),
                        tf.keras.layers.Activation(tf.nn.softplus),
                        tdl.bayesnet.AffineNormalLayer(
                            units=1, tolerance=1e-3),
                        tf.keras.layers.Activation(tf.nn.softplus),
                        lambda x: x+1e-3]))
    labels = tf.placeholder(tf.float32, (None, 1))
    inputs = tf.placeholder(tf.float32, (None, 1))
    outputs = model(inputs)
    # loss
    kernel_prior = tfp.distributions.Normal(loc=0.0, scale=1000.0)
    kl = [tfp.distributions.kl_divergence(kernel, kernel_prior)
          for kernel in [model.loc.layers[0].kernel,
                         model.loc.layers[2].kernel]]
    try:
        kernel_prior = tfp.distributions.Normal(loc=0.0, scale=100.0)
        kl.update([tfp.distributions.kl_divergence(kernel, kernel_prior)
                   for kernel in [model.scale.layers[0].kernel,
                                  model.scale.layers[2].kernel]])
    except AttributeError:
        pass
    log_prob = tf.reduce_mean(tf.reduce_sum(outputs.log_prob(labels), -1),
                              axis=0)
    kl = tf.add_n([tf.reduce_sum(kl_i) for kl_i in kl])
    train = tdl.core.SimpleNamespace(
        inputs=inputs,
        outputs=outputs,
        loss=-log_prob + (1/dataset.train.n_samples)*kl,
        labels=labels)

    # Test model
    test_model = tdl.bayesnet.SampleLayer(distribution=model)
    test = tdl.core.SimpleNamespace(
        inputs=inputs,
        outputs=test_model(inputs)
        )

    return tdl.core.SimpleNamespace(model=model, train=train, test=test)


class BayesnetTest(unittest.TestCase):
    def test_homoscedastic(self):
        dataset = create_homoscedastic_dataset()
        estimator = get_model(dataset, 'homoscedastic')
        optim = tdl.optim.Optimizer(
            loss=estimator.train.loss,
            var_list=tdl.core.get_trainable(estimator.model),
            log_folder='tmp/',
            learning_rate=0.2)
        tdl.core.initialize_variables(estimator.model)
        optim.run(
            feed_train=lambda:
            {estimator.train.labels: dataset.train.y[..., np.newaxis],
             estimator.train.inputs: dataset.train.x[..., np.newaxis]},
            n_train_steps=1000)

        loss_value = estimator.train.loss.eval(
            feed_dict={
                estimator.train.labels: dataset.train.y[..., np.newaxis],
                estimator.train.inputs: dataset.train.x[..., np.newaxis]})
        # loss = main.ml_model.monitor\
        #            .train['train/loss'].current_value
        assert np.isfinite(loss_value), 'training resulted in nan for bayesnet'
        loss_avg = optim.monitor_manager.train['train/loss'].mean()
        assert (loss_avg < 2.85) and (loss_avg > 1.0),\
            'loss value is outside the expected range'

    def test_heteroscedastic(self):
        dataset = create_heteroscedastic_dataset()
        estimator = get_model(dataset, 'heteroscedastic')
        optim = tdl.optim.Optimizer(
            loss=estimator.train.loss,
            var_list=tdl.core.get_trainable(estimator.model),
            log_folder='tmp/',
            learning_rate=0.005)
        tdl.core.initialize_variables(estimator.model)
        optim.run(
            feed_train=lambda:
            {estimator.train.labels: dataset.train.y[..., np.newaxis],
             estimator.train.inputs: dataset.train.x[..., np.newaxis]},
            n_train_steps=3000)

        loss_value = estimator.train.loss.eval(
            feed_dict={
                estimator.train.labels: dataset.train.y[..., np.newaxis],
                estimator.train.inputs: dataset.train.x[..., np.newaxis]})
        # loss = main.ml_model.monitor\
        #            .train['train/loss'].current_value
        assert np.isfinite(loss_value), 'training resulted in nan for bayesnet'
        loss_avg = optim.monitor_manager.train['train/loss'].mean()
        assert (loss_avg < -0.7) and (loss_avg > -1.0),\
            'loss value is outside the expected range'

    def test_normal(self):
        with tf.Session().as_default() as sess:
            main = tdlb.Normal(shape=[100, 300])
            tf.global_variables_initializer().run()
            test = main.evaluate()

            x, mean1 = sess.run([test.samples.value,
                                 test.samples.mean])
            np.testing.assert_almost_equal(np.mean(x), np.mean(mean1))

    def test_normal_model(self):
        attr_names = tdl.core.common._find_tdl_attrs(
            tdl.bayesnet.NormalModel,
            tdl.core.TDL_INIT_DESCRIPTORS)
        assert all(name in {'batch_shape', 'input_shape', 'loc', 'scale'}
                   for name in attr_names)


if __name__ == "__main__":
    unittest.main()
