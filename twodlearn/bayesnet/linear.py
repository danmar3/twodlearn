from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import twodlearn as tdl
import twodlearn.bayesnet as tdlb
import twodlearn.bayesnet.distributions
import twodlearn.linalg
import twodlearn.debug


def _frob_squared(M):
    ''' squared frobenius norm '''
    return tf.reduce_sum(tf.square(M), axis=[-2, -1])


class VLinear(tdl.core.TdlModel):
    ''' variational linear model '''
    @tdl.core.InputModel
    def basis(self, value):
        if value is None:
            value = tdl.kernels.ConcatOnes()
        assert (callable(value)), 'explicit basis should be callable'
        return value

    @tdl.core.InputArgument
    def basis_dim(self, value):
        ''' number of dimensions of the explicit basis '''
        tdl.core.assert_initialized(self, 'basis_dim', ['basis'])
        if value is None:
            tdl.core.assert_any_available(
                self, 'basis_dim', ['weights', 'weights_prior'])
            if tdl.core.is_property_initialized(self, 'weights_prior'):
                value = self.beta_prior.event_shape[-1].value
            if tdl.core.is_property_initialized(self, 'weights'):
                value = self.beta.event_shape[-1].value
        return value

    @tdl.core.Submodel
    def weights_prior(self, value, AutoType=None):
        ''' prior for the linear parameters '''
        if AutoType is None:
            AutoType = tdl.AutoTensor()
        if value is None:
            value = 100.0
        if isinstance(value, (int, float)):
            tdl.core.assert_initialized(self, 'beta_prior',
                                        ['basis_dim', 'batch_shape'])
            value = tdlb.distributions.MVNScaledIdentity(
                shape=self.batch_shape.concatenate(self.basis_dim),
                scale=(value, AutoType),
                loc=AutoType)
        return value

    @tdl.core.Submodel
    def weights(self, value):
        ''' linear weights '''
        if value is None:
            tdl.core.assert_initialized(self, 'weights', ['weights_prior'])
            value = tdlb.distributions.MVNDiag(
                shape=self.batch_shape.concatenate(self.basis_dim),
                scale=tdl.AutoVariable(),
                # loc=tdl.AutoVariable(),
                loc=(tf.convert_to_tensor(self.beta_prior.loc),
                     tdl.AutoVariable()),
                tolerance=self.tolerance)
        return value

    @tdl.core.InputArgument
    def batch_shape(self, value):
        ''' number of output dimensions '''
        if value is None:
            value = 1
        if not isinstance(value, tf.TensorShape):
            value = tf.TensorShape(value)
        return value

    @tdl.core.PropertyShortcuts({'model': ['weights', 'weights_prior']})
    class VLinearTransform(tdlb.distributions.MVN):
        ''' distribution after performing a linear transformation '''
        @tdl.core.InputModel
        def model(self, value):
            ''' VLinear model '''
            return value

        @tdl.core.InputArgument
        def inputs(self, value):
            return value

        @tdl.core.Submodel
        def y_scale(self, _):
            ''' linear operation for y_scale '''
            scale = (self.model.y_scale(self.inputs)
                     if self.model.y_scale is not None
                     else None)
            return scale

        @tdl.core.LazzyProperty
        def basis_x(self):
            return self.model.basis(self.inputs)

        @tdl.core.LazzyProperty
        def loc(self):
            ''' H(X) @ weights '''
            return self.basis_x.linop.matvec(self.weights.loc)

        @tdl.core.LazzyProperty
        def covariance(self):
            lt_xt = self.weights.scale.matmul(
                tf.convert_to_tensor(self.basis_x),
                adjoint=True,
                adjoint_arg=True
            )
            return tdl.linalg.Mt_times_M(lt_xt)

        def with_noise(self):
            '''return the posterior with noise'''
            covariance = tf.convert_to_tensor(self.covariance)
            noise_cov = tdl.linalg.M_times_Mt(self.y_scale)
            # add the shift
            if isinstance(noise_cov, tf.linalg.LinearOperator):
                covariance = noise_cov.add_to_tensor(covariance)
            else:
                covariance = covariance + noise_cov
            return tdlb.distributions.MVN(
                loc=tf.convert_to_tensor(self.loc),
                covariance=covariance)

        def neg_melbo(self, labels, dataset_size=None):
            ''' return the negative mean elbo'''
            return VLinear.VLinearELBO(
                labels=labels, dataset_size=dataset_size)

    @tdl.core.PropertyShortcuts(
        {'posterior': ['inputs', 'y_scale', 'basis_x', 'weights',
                       'weights_prior']})
    class VLinearELBO(tdl.core.TdlModel):
        @tdl.core.InputArgument
        def labels(self, value):
            return value

        @tdl.core.InputArgument
        def dataset_size(self, value):
            ''' number of samples in the entire dataset '''
            if value is None:
                value = self.batch_size
            dtype = tf.convert_to_tensor(self.inputs).dtype
            return tf.cast(value, dtype)

        @tdl.core.InputModel
        def posterior(self, value):
            if value is None:
                raise ValueError('VLinearELBO has not been provided with the '
                                 'posterior distribution')
            if not isinstance(value, VLinear.VLinearTransform):
                raise TypeError('The posterior distrubution for VLinearELBO '
                                'needs to be an instance of VLinearTransform')
            return value

        @tdl.core.LazzyProperty
        def expected_log_likelihood(self):
            if tdl.linalg.is_scaled_identity(self.y_scale):
                dist = tdlb.distributions.MVNScaledIdentity(
                    loc=self.basis_x.linop.matvec(self.weights.loc),
                    scale=self.y_scale.multiplier)
            elif tdl.linalg.is_diagonal(self.y_scale):
                dist = tdlb.distributions.MVNDiag(
                    loc=self.basis_x.linop.matvec(self.weights.loc),
                    scale=self.y_scale.diag_part())
            else:
                raise NotImplementedError('y_scale must be a scaled identity '
                                          'or diagonal')

            tr = _frob_squared(self.y_scale.solve(
                self.weights.scale.matmul(
                    self.base_x, adjoint=True, adjoint_arg=True),
                adjoint_arg=True))
            return dist.log_prob(self.labels) - 0.5 * tr

        @tdl.core.LazzyProperty
        def weights_kl(self):
            return tdlb.losses.KLDivergence(self.weights, self.weights_prior)

        @tdl.core.LazzyProperty
        def batch_size(self):
            ''' number of samples in a single mini-batch '''
            return tf.cast(tf.shape(self.inputs)[0],
                           tf.convert_to_tensor(self.inputs).dtype)

        @tdl.core.LazzyProperty
        def melbo(self):
            return (self.expected_log_likelihood/self.batch_size
                    - self.weights_kl/self.dataset_size)

        @tdl.core.LazzyProperty
        def value(self):
            return -self.melbo
