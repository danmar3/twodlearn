from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pytest
import unittest
import numpy as np
import tensorflow as tf
import twodlearn as tdl
import matplotlib.pyplot as plt
from twodlearn.bayesnet.gaussian_process import (
    GaussianProcess, ExplicitVGP)

TESTS_PATH = os.path.dirname(os.path.abspath(__file__))
TMP_PATH = os.path.join(TESTS_PATH, 'tmp/')


def get_data():
    train_x = np.concatenate([np.expand_dims(np.linspace(-10, -7, 10, dtype=np.float32), 1),
                              np.expand_dims(np.linspace(7, 10, 10, dtype=np.float32), 1)])
    train_y = np.sin(train_x)
    test_x = np.expand_dims(np.linspace(-15, 15, 100, dtype=np.float32), 1)
    return train_x, train_y, test_x


def gaussian_kernel(X1, X2, l_scale, f_scale, y_scale=None):
    # gaussian_kernel: gaussian kernel calculation between datasets X1 and X2
    # X1 is a matrix, whose rows represent samples
    #   X2 is a matrix, whose rows represent samples
    #   K(i,j) = (f_scale^2) exp(-0.5 (x1(i)-x2(j))' (l^-2)I (x1(i)-x2(j)))
    # Calculate kernel matrix
    X1 = np.expand_dims(X1, -2)
    X2 = np.expand_dims(X2, -3)
    K = np.sum((X1-X2)**2, -1)
    K = (f_scale**2) * np.exp(-K / (2 * (l_scale**2)))
    if y_scale is not None:
        K = K + (y_scale**2) * np.eye(K.shape[0])
    return K


def gaussian_process(x_train, y_train, x_test, l_scale, f_scale, y_scale,
                     tolerance):
    k11 = gaussian_kernel(x_train, x_train, l_scale=l_scale, f_scale=f_scale)
    k11 = k11 + (tolerance + y_scale**2)*np.eye(k11.shape[0])
    k11inv = np.linalg.inv(k11)
    k12 = gaussian_kernel(x_train, x_test, l_scale=l_scale, f_scale=f_scale)
    k22 = gaussian_kernel(x_test, x_test, l_scale=l_scale, f_scale=f_scale)
    # l11 = np.linalg.cholesky(k11)
    mean = np.matmul(np.matmul(k12.T, k11inv), y_train)
    cov = k22 - np.matmul(np.matmul(k12.T, k11inv), k12)
    return mean, cov


class GpmTest(unittest.TestCase):
    def test_kernel(self):
        with tf.Session().as_default():
            train_x, train_y, test_x = get_data()
            l_scale = np.random.rand()
            f_scale = np.random.rand()
            model = tdl.kernels.GaussianKernel(l_scale=l_scale,
                                               f_scale=f_scale)
            kernel_tdl = model.evaluate(train_x, train_x)
            kernel_np = gaussian_kernel(train_x, train_x,
                                        l_scale=l_scale,
                                        f_scale=f_scale)
            tf.global_variables_initializer().run()
            np.testing.assert_almost_equal(kernel_tdl.value.eval(), kernel_np,
                                           decimal=5)

    def test_gp_model(self):
        with tf.Session().as_default() as session:
            train_x, train_y, test_x = get_data()
            kernel = tdl.kernels.GaussianKernel(l_scale=0.3)
            model = GaussianProcess(train_x, np.squeeze(train_y),
                                    kernel=kernel,
                                    y_scale=tf.Variable(0.1, trainable=False))
            train = model.predict(train_x)
            test = model.predict(test_x)
            loss = model.marginal_likelihood().loss
            optimizer = tdl.optim.Optimizer(
                loss, var_list=tdl.core.get_trainable(model),
                log_folder='gpm_test_tmp/monitors')
            tdl.core.initialize_variables(model)
            optimizer.run(500)

            assert np.isfinite(loss.eval()),\
                'optimization failed, loss resulted in non finite number'
            assert (loss.eval() > -9.5 and
                    loss.eval() < -7.5),\
                'value for the gp loss outside of expected bounds'
            [mean, stddev] = session.run([
                tf.convert_to_tensor(test.loc),
                tf.convert_to_tensor(test.scale)])
            assert (np.isfinite(mean)).all(),\
                'optimization failed, mean resulted in non finite number'
            assert (np.isfinite(stddev)).all(),\
                'optimization failed, stddev resulted in non finite number'

    def test_mean_cov(self):
        with tf.Session().as_default():
            train_x, train_y, test_x = get_data()
            l_scale = 0.1
            f_scale = 0.1
            y_scale = 0.01
            kernel = tdl.kernels.GaussianKernel(
                l_scale=l_scale, f_scale=f_scale)
            gp_model = GaussianProcess(train_x, train_y.transpose(),
                                       y_scale=y_scale, kernel=kernel)
            test = gp_model.predict(test_x)
            mean, cov = gaussian_process(train_x, train_y, test_x,
                                         l_scale, f_scale, y_scale,
                                         gp_model.tolerance)
            tf.global_variables_initializer().run()
            np.testing.assert_almost_equal(np.squeeze(test.loc.eval()),
                                           np.squeeze(mean),
                                           decimal=4)
            np.testing.assert_almost_equal(
                np.squeeze(test.covariance.eval()),
                np.squeeze(cov), decimal=4)

    def test_shapes(self):
        '''test shapes of vgp.'''
        m = 20
        n_inputs = 1
        n_states = 4
        n_outputs = 4
        model = ExplicitVGP(
            m=m, input_shape=[None, n_inputs + n_states],
            batch_shape=[n_outputs],
            basis=tdl.AutoInit(),
            xm={'independent': True},
            kernel={'l_scale': tdl.constrained.PositiveVariableExp(
                np.power(1.0/m, 1.0/n_outputs))
            })

        batch_shape = 51
        inputs = tf.zeros([batch_shape, n_inputs+n_states])
        labels = tf.zeros([n_outputs, batch_shape])
        posterior = model.predict(inputs)
        loss = posterior.neg_elbo(labels=labels)
        loss = tf.convert_to_tensor(loss)
        assert posterior.loc.shape.as_list() == [4, 51]
        assert posterior.covariance.shape.as_list() == [4, 51, 51]
        assert loss.shape.as_list() == [4]

    def test_shapes_independent(self):
        '''test shapes of vgp using (batched) independent inputs.'''
        m = 20
        n_inputs = 1
        n_states = 4
        n_outputs = 4
        model = ExplicitVGP(
            m=m, input_shape=[None, n_inputs + n_states],
            batch_shape=[n_outputs],
            basis=tdl.AutoInit(),
            xm={'independent': True},
            kernel={'l_scale': tdl.constrained.PositiveVariableExp(
                np.power(1.0/m, 1.0/n_outputs))
            })

        batch_shape = 51
        inputs = tf.zeros([batch_shape, n_inputs+n_states])
        labels = tf.zeros([batch_shape, n_outputs, 1])
        posterior = model.predict(inputs[:, tf.newaxis, tf.newaxis, ...])
        loss = posterior.neg_elbo(labels=labels)
        loss = tf.convert_to_tensor(loss)
        assert posterior.loc.shape.as_list() == [51, 4, 1]
        assert posterior.covariance.shape.as_list() == [51, 4, 1, 1]
        assert loss.shape.as_list() == [51, 4]


if __name__ == "__main__":
    unittest.main()
