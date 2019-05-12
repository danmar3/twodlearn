from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest
import numpy as np
import tensorflow as tf
import twodlearn as tdl
import twodlearn.bayesnet.distributions
import tensorflow_probability as tfp

TESTS_PATH = os.path.dirname(os.path.abspath(__file__))
TMP_PATH = os.path.join(TESTS_PATH, 'tmp/')


class DistributionsTests(unittest.TestCase):
    def test_pdmatrix_shapes(self):
        m1 = tdl.bayesnet.distributions.PDMatrix(
            shape=[3, 4, 5, 5]
        )
        m2 = tdl.bayesnet.distributions.PDMatrix(
            batch_shape=[3, 4],
            domain_dimension=5
        )
        m3 = tdl.bayesnet.distributions.PDMatrix(
            batch_shape=(),
            domain_dimension=5
        )
        m4 = tdl.bayesnet.distributions.PDMatrix(
            domain_dimension=5
        )
        m5 = tdl.bayesnet.distributions.PDMatrix(
            shape=5
        )
        assert m1.shape == m2.shape,\
            'shape of matrices shoud be the same'
        assert m3.shape == tf.TensorShape([5, 5])
        assert m4.shape == tf.TensorShape([5, 5])
        assert m5.shape == tf.TensorShape([5, 5])

    def test_log_prob(self):
        with tf.Session().as_default():
            mvn_tdl = tdl.bayesnet.distributions.MVN(
                shape=2, covariance=tf.constant([[2.0, 0.0], [0.0, 1.0]]))
            mvn_tfp = tfp.distributions.MultivariateNormalFullCovariance(
                covariance_matrix=mvn_tdl.covariance)

            test = mvn_tdl.sample(2000)
            tf.global_variables_initializer().run()
            samp = test.eval()
            logprob_1 = mvn_tdl.log_prob(samp)
            logprob_2 = mvn_tfp.log_prob(samp)
            np.testing.assert_almost_equal(logprob_1.eval(), logprob_2.eval(),
                                           decimal=5)

    def test_log_prob2(self):
        with tf.Session().as_default():
            scale = 5.0*np.random.normal(size=[5, 5]).astype(np.float32)
            mvn_tdl = tdl.bayesnet.distributions.MVN(scale=scale)
            mvn_tfp = tfp.distributions.MultivariateNormalFullCovariance(
                covariance_matrix=mvn_tdl.covariance)

            test = mvn_tdl.sample(2000)
            tf.global_variables_initializer().run()
            samp = test.eval()
            logprob_1 = mvn_tdl.log_prob(samp)
            logprob_2 = mvn_tfp.log_prob(samp)
            np.testing.assert_almost_equal(logprob_1.eval(), logprob_2.eval(),
                                           decimal=4)

    def test_mvn_kl(self):
        with tf.Session().as_default():
            scalea = np.random.normal(size=[5, 5]).astype(np.float32)
            scaleb = 0.3*np.random.normal(size=[5, 5]).astype(np.float32)
            scalec = 0.2*np.random.normal(size=[5, 5]).astype(np.float32)
            mvna_tdl = tdl.bayesnet.distributions.MVN(scale=scalea)
            mvnb_tdl = tdl.bayesnet.distributions.MVN(scale=scaleb)
            mvnc_tdl = tdl.bayesnet.distributions.MVN(scale=scalec)
            mvna_tfp = tfp.distributions.MultivariateNormalFullCovariance(
                covariance_matrix=mvna_tdl.covariance)
            mvnb_tfp = tfp.distributions.MultivariateNormalFullCovariance(
                covariance_matrix=mvnb_tdl.covariance)

            test = mvna_tdl.sample(2000)
            test = mvnb_tdl.sample(2000)
            test = mvnc_tdl.sample(2000)
            tf.global_variables_initializer().run()
            kl1 = tdl.bayesnet.losses.KLDivergence(mvna_tdl, mvnb_tdl)
            kl2 = tdl.bayesnet.losses.KLDivergence(mvna_tfp, mvnb_tfp)
            np.testing.assert_approx_equal(kl1.value.eval(), kl2.value.eval())
            with self.assertRaises(AssertionError):
                kl3 = tdl.bayesnet.losses.KLDivergence(mvna_tdl, mvnc_tdl)
                np.testing.assert_approx_equal(
                    kl1.value.eval(), kl3.value.eval())

    def test_mvn_kl2(self):
        with tf.Session().as_default():
            scalea = np.random.normal(size=[5, 5]).astype(np.float32)
            scaleb = 0.5*np.random.normal(size=[5, 5]).astype(np.float32)
            mvna_tdl = tdl.bayesnet.distributions.MVN(scale=scalea)
            mvnb_tdl = tdl.bayesnet.distributions.MVN(scale=scaleb)
            mvna_tfp = tfp.distributions.MultivariateNormalFullCovariance(
                covariance_matrix=mvna_tdl.covariance)
            mvnb_tfp = tfp.distributions.MultivariateNormalFullCovariance(
                covariance_matrix=mvnb_tdl.covariance)

            test = mvna_tdl.sample(2000)
            test = mvnb_tdl.sample(2000)
            tf.global_variables_initializer().run()
            kl1 = tdl.bayesnet.losses.KLDivergence(mvna_tdl, mvnb_tfp)
            kl2 = tdl.bayesnet.losses.KLDivergence(mvna_tfp, mvnb_tdl)
            np.testing.assert_approx_equal(kl1.value.eval(), kl2.value.eval())

    def test_diag_1(self):
        with tf.Session().as_default():
            diag = tdl.bayesnet.distributions.PDMatrixDiag(shape=[2, 1, 4, 4],
                                                           raw=0.5)
            tdl.core.is_property_set(diag.cholesky, 'value')
            tf.global_variables_initializer().run()
            diag.cholesky.linop.to_dense().eval()
            assert tdl.core.is_property_set(diag.cholesky, 'value') is False

    def test_mvn_diag_1(self):
        with tf.Session().as_default():
            diag = tdl.bayesnet.distributions.PDMatrixDiag(shape=[2, 1, 4, 4],
                                                           raw=0.5)
            mvn = tdl.bayesnet.distributions.MVNDiag(covariance=diag)
            mvn.scale.linop  # init
            tf.global_variables_initializer().run()
            assert tdl.core.is_property_set(mvn.scale, 'value') is False
            mvn.scale.linop.diag_part().eval().mean() == 0.5

    def test_kl_diag_1(self):
        with tf.Session().as_default():
            diag1 = tdl.bayesnet.distributions.PDMatrixDiag(
                shape=[2, 1, 4, 4], raw=0.5)
            diag2 = tdl.bayesnet.distributions.PDMatrixDiag(
                shape=[2, 1, 4, 4], raw=0.89)
            mvn1 = tdl.bayesnet.distributions.MVNDiag(covariance=diag1)
            mvn1_tf = tfp.distributions.MultivariateNormalDiag(
                scale_diag=mvn1.scale.linop.diag_part()
            )
            mvn2 = tdl.bayesnet.distributions.MVNDiag(covariance=diag2)
            mvn2_tf = tfp.distributions.MultivariateNormalDiag(
                scale_diag=tf.linalg.diag_part(mvn2.scale.value)
            )

            kl1 = tdl.bayesnet.losses.KLDivergence(mvn1, mvn2).value
            kl2 = tdl.bayesnet.losses.KLDivergence(mvn1_tf, mvn2_tf).value
            tf.global_variables_initializer().run()
            np.testing.assert_almost_equal(kl1.eval(), kl2.eval())

    def test_kl_diag_2(self):
        with tf.Session().as_default():
            scale1 = 0.1*np.random.normal(size=[5, 5]).astype(np.float32)
            mvn1 = tdl.bayesnet.distributions.MVN(scale=scale1)
            mvn1_tf = tfp.distributions.MultivariateNormalFullCovariance(
                covariance_matrix=mvn1.covariance
            )

            diag2 = tdl.bayesnet.distributions.PDMatrixDiag(
                shape=[5, 5], raw=0.89)
            mvn2 = tdl.bayesnet.distributions.MVNDiag(covariance=diag2)
            mvn2_tf = tfp.distributions.MultivariateNormalDiag(
                scale_diag=tf.linalg.diag_part(mvn2.scale.value)
            )

            kl1 = tdl.bayesnet.losses.KLDivergence(mvn1, mvn2).value
            kl2 = tdl.bayesnet.losses.KLDivergence(mvn1_tf, mvn2_tf).value
            tf.global_variables_initializer().run()
            np.testing.assert_almost_equal(kl1.eval(), kl2.eval())

    def test_mvn_diag_log_prob(self):
        with tf.Session().as_default():
            mvn_tdl = tdl.bayesnet.distributions.MVNDiag(
                covariance=tdl.bayesnet.distributions
                              .PDMatrixDiag(raw=tf.constant([2.0, 1.0])))
            mvn_tfp = tfp.distributions.MultivariateNormalDiag(
                scale_diag=tf.linalg.diag_part(mvn_tdl.scale.value)
            )

            test = mvn_tdl.sample(2000)
            tf.global_variables_initializer().run()
            samp = test.eval()
            logprob_1 = mvn_tdl.log_prob(samp)
            logprob_2 = mvn_tfp.log_prob(samp)
            np.testing.assert_almost_equal(logprob_1.eval(), logprob_2.eval(),
                                           decimal=5)


if __name__ == "__main__":
    unittest.main()
