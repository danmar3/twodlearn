from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import twodlearn as tdl
import twodlearn.bayesnet as tdlb
import twodlearn.bayesnet.distributions
import twodlearn.linalg
from twodlearn.bayesnet.distributions import Cholesky
import twodlearn.debug


def _frob_squared(M):
    ''' squared frobenius norm '''
    return tf.reduce_sum(tf.square(M), axis=[-2, -1])


class GPNegLogEvidence(tdl.losses.EmpiricalLoss):
    ''' Negative log evidence for a Gaussian Process with training covariance
    cov = K(train, train).
    The loss takes the following form:
        loss = 0.5(y' inv(cov) y + log(|cov|) + n log(2 pi))

    if conv_inv can be provided when instiating the loss to prevent computing
    the inverse multiple times
    '''
    _default_name = 'GPNegLogEvidence'
    _parameters = ['labels']

    @tdl.SimpleParameter
    def labels(self, value):
        if value is None:
            assert (self.cov.shape[0] == self.cov.shape[1]),\
                'GPNegLogEvidence requires covariance to be a square matrix '\
                '(m, m), where m are the samples. Provided covariance has '\
                'a shape of {}'.format(self.cov.shape)
            value = tf.placeholder(tf.float32,
                                   shape=(self.cov.shape[0], 1),
                                   name='labels')
        return value

    def define_fit_loss(self, cov, cov_inv, labels, loc):
        if loc is None:
            y = labels
        else:
            y = labels - loc

        Ky = cov
        Ky_inv = cov_inv
        if Ky_inv is None:
            Ky_inv = tf.matrix_inverse(Ky)

        n = tf.cast(Ky.shape[0], tf.float32)
        with tf.name_scope('loss'):
            loss = 0.5*(tf.matmul(tf.transpose(y), tf.matmul(Ky_inv, y))
                        + tf.log(tf.matrix_determinant(Ky))
                        + n * tf.log(2*np.pi))
        return loss

    def __init__(self, cov, cov_inv=None, labels=None, loc=None, name=None):
        self.loc = loc
        self.cov = cov
        super(GPNegLogEvidence, self).__init__(labels=labels, name=name)
        with tf.name_scope(self.scope):
            self._value = self.define_fit_loss(cov, cov_inv, self.labels, loc)


class DynamicScaledIdentity(tdl.core.TdlModel):
    '''scaled identity [batched] matrix whose size depends on the
    shape of the input'''
    @property
    def batch_shape(self):
        return tf.convert_to_tensor(self.multipliers).shape

    @tdl.core.InputArgument
    def tolerance(self, value):
        return value

    @tdl.core.InputParameter
    def multipliers(self, value):
        """tensor whose elements represent the multiplier factors for the
        identity matrices.
        """
        return value

    def __call__(self, inputs):
        '''returns a linear scaled identity operator.
        The number of rows in the identity matrices is equal to
        inputs.shape[-2]
        '''
        inputs = tf.convert_to_tensor(inputs)
        if inputs.shape.is_fully_defined():
            n_rows = inputs.shape[-2]
        else:
            n_rows = tf.shape(inputs)[-2]
        if self.tolerance is None:
            multipliers = self.multipliers
        else:
            multipliers = self.multipliers + self.tolerance
        return tf.linalg.LinearOperatorScaledIdentity(
            num_rows=n_rows, multiplier=multipliers
        )


class GaussianProcess(tdl.core.TdlModel):
    '''Gaussian process with zero mean.
    The covariance kernel takes by default the following form:
       X1 is a matrix, whose rows represent samples
       X2 is a matrix, whose rows represent samples
       K(i,j) = (f_scale**2) exp(-0.5 (x1(i)-x2(j))' (l**-2)I (x1(i)-x2(j)) )

    By default, when float values of l_scale, f_scale and y_scale are provided,
    a trainable variable is created. If want them fixed, you can
    provide a kernel with parameters as tf.Variable(value, trainable=False)
    '''
    @tdl.core.InputParameter
    def xm(self, value):
        if isinstance(value, np.ndarray):
            value = tf.convert_to_tensor(value)
        return value

    @tdl.core.InputParameter
    def ym(self, value):
        if isinstance(value, np.ndarray):
            value = tf.convert_to_tensor(value)
        return value

    @tdl.core.LazzyProperty
    def m(self):
        ''' number of observed points x.shape[-2]'''
        m = tf.convert_to_tensor(self.xm).shape[-2].value
        if m is None:
            m = tf.shape(self.xm)[-2]
        return m

    @tdl.core.InputParameter
    def tolerance(self, value):
        if value is None or isinstance(value, float):
            value = (value if value is not None
                     else tdl.core.global_options.tolerance)
        return value

    @tdl.core.InputParameter
    def y_scale(self, value, AutoType=None):
        ''' Measurement scale: y \sim N(f, y_scale**2 I) '''
        if AutoType is None:
            AutoType = tdl.core.variable
        if isinstance(value, (int, float, np.ndarray)):
            ym = tf.convert_to_tensor(self.ym)
            if ym.shape.is_fully_defined():
                multiplier_shape = ym.shape[:-1]
            else:
                multiplier_shape = tf.shape(ym)[:-1]
            value = AutoType(tf.constant(value, shape=multiplier_shape),
                             name='y_scale')
        if value is not None:
            value = DynamicScaledIdentity(multipliers=value)
        return value

    @tdl.core.Submodel
    def kernel(self, value):
        if value is None:
            value = tdl.kernels.GaussianKernel(
                l_scale=1/self.m,
                f_scale=1.0, name='kernel')
        return value

    def __init__(self, xm, ym, y_scale=0.1, kernel=None,
                 options=None, name=None, **kargs):
        super(GaussianProcess, self)\
            .__init__(xm=xm, ym=ym, y_scale=y_scale,
                      kernel=kernel, options=options, name=name, **kargs)

    @tdl.core.ModelMethod(['distribution', 'loss', 'k11'], [])
    def marginal_likelihood(self, object):
        '''marginal likelihood p(y|X) = N(0, K(X,X) + (y_scale**2) I)'''
        k11 = self.kernel.evaluate(self.xm, self.xm)
        k11 = tf.convert_to_tensor(k11)
        if self.y_scale is not None:
            y_scale = self.y_scale(self.xm)
            shift = tdl.linalg.M_times_Mt(y_scale)
            # expand dims if input is not batched
            extra_dims = shift.shape.ndims - k11.shape.ndims
            k11 = tdl.core.array.expand_dims_left(k11, ndims=extra_dims)
            # compute sum
            if isinstance(shift, tf.linalg.LinearOperator):
                k11 = shift.add_to_tensor(k11)
            else:
                k11 = k11 + shift
        k11 = tdl.core.array.add_diagonal_shift(k11, self.tolerance)
        dist = tdlb.distributions.MVN(
            loc=tf.zeros_like(self.ym),
            covariance=k11)
        loss = -dist.log_prob(self.ym)
        return dist, loss, k11

    @tdl.core.PropertyShortcuts({'model': ['xm', 'ym']})
    class GpOutput(tdl.core.OutputModel):
        @tdl.core.InputModel
        def model(self, value):
            ''' GP model '''
            return value

        @tdl.core.InferenceInput
        def inputs(self, value):
            return value

        @tdl.core.LazzyProperty
        def y_scale(self):
            return (self.model.y_scale(self.inputs)
                    if self.model.y_scale is not None
                    else None)

        @tdl.Submodel
        def k11(self, _):
            tolerance = self.model.tolerance
            k11 = self.model.kernel.evaluate(self.xm, self.xm)
            k11 = tf.convert_to_tensor(k11)
            if self.y_scale is not None:
                shift = tdl.linalg.M_times_Mt(self.y_scale)
                # expand dims if input is not batched
                extra_dims = shift.shape.ndims - k11.shape.ndims
                k11 = tdl.core.array.expand_dims_left(k11, ndims=extra_dims)
                # add the shift
                if isinstance(shift, tf.linalg.LinearOperator):
                    k11 = shift.add_to_tensor(k11)
                else:
                    k11 = k11 + shift
            k11 = tdl.core.array.add_diagonal_shift(k11, tolerance)
            cholesky = Cholesky(k11)
            return tdl.core.SimpleNamespace(value=k11, cholesky=cholesky)

        @tdl.Submodel
        def k12(self, _):
            k12 = self.model.kernel.evaluate(self.xm, self.inputs)
            k12.linop = tf.linalg.LinearOperatorFullMatrix(
                tf.convert_to_tensor(k12))
            return k12

        @tdl.Submodel
        def k22(self, _):
            return self.model.kernel.evaluate(self.inputs, self.inputs)

    class GPPosterior(GpOutput, tdlb.distributions.MVN):
        """Compute the posterior distribution for p(f*| x*, ym, Xm)
        where {ym, Xm} is the training dataset of the GP model.
        """
        @tdl.core.LazzyProperty
        def loc(self):
            ''' loc = k21 @ k11 \ ym '''
            loc = self.k12.linop.matvec(
                tdl.linalg.solvevec(self.k11.cholesky, self.ym),
                adjoint=True)
            return loc

        @tdl.core.LazzyProperty
        def covariance(self):
            tdl.core.assert_initialized(self, 'covariance', ['shape'])
            k12 = tf.convert_to_tensor(self.k12)
            k22 = tf.convert_to_tensor(self.k22)
            k22 = tdl.core.array.add_diagonal_shift(k22, self.model.tolerance)
            # covariance = k22 - k21 @ k11inv @ k12
            v = self.k11.cholesky.linop.solve(k12)
            vop = tf.linalg.LinearOperatorFullMatrix(v)
            covariance = k22 - vop.matmul(v, adjoint=True)
            # expand dims if covariance does not include batch dimensions
            if covariance.shape.ndims == 2:
                if covariance.shape.is_fully_defined():
                    new_shape = tf.TensorShape([1]*self.batch_shape.ndims)\
                                  .concatenate(covariance.shape)
                else:
                    cov_shape = tf.shape(covariance)
                    new_shape = tf.concat(
                        [tf.ones(self.batch_shape.ndims, cov_shape.dtype),
                         cov_shape], axis=0)
                covariance = tf.reshape(covariance, shape=new_shape)
            return covariance

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

    def predict(self, inputs):
        """Compute the posterior distribution for p(f*| x*, y, X)
        where {y, X} is the training dataset of the GP model.
        Args:
            inputs (tf.Tensor, TdlModel): matrix with test inputs x*.
        Returns:
            GpOutput: TdlModel with the value of p(f*| x*, y, X).
                loc: mean for f*
                scale: scale for f*
                posterior: distributions for y* and f*
        """
        return GaussianProcess.GPPosterior(model=self, inputs=inputs)


class GpWithExplicitMean(tdl.core.TdlModel):
    @tdl.core.InputModel
    def gp_model(self, value):
        return value

    @tdl.core.InputModel
    def explicit_basis(self, value):
        if value is None:
            # value = tdl.core.Identity()
            value = tdl.kernels.ConcatOnes()
        assert (callable(value)), 'explicit basis should be callable'
        return value

    @tdl.core.InputArgument
    def prior_scale(self, value):
        ''' scale of the prior '''
        # valid = (type(None), tf.distributions.Normal,
        #         tfp.distributions.MultivariateNormalFullCovariance)
        return value

    def __init__(self, gp_model, prior_scale=None,
                 explicit_basis=None, **kargs):
        if explicit_basis is not None:
            kargs['explicit_basis'] = explicit_basis
        super(GpWithExplicitMean, self).__init__(
            gp_model=gp_model, prior_scale=prior_scale,
            **kargs)

    class MarginalOutput(tdl.core.OutputModel):
        @property
        def gp_model(self):
            return self.model.gp_model

        @property
        def prior_scale(self):
            return self.model.prior_scale

        @tdl.core.LazzyProperty
        def gp_marginal(self):
            return self.model.gp_model.marginal_likelihood()

        @tdl.core.LazzyProperty
        def k11(self):
            value = tf.convert_to_tensor(self.gp_marginal.k11)
            return tdl.core.SimpleNamespace(value=value,
                                            cholesky=Cholesky(value))

        @tdl.core.LazzyProperty
        def basis(self):
            h = self.model.explicit_basis(self.gp_model.xm)
            value = tf.convert_to_tensor(h)
            linop = tf.linalg.LinearOperatorFullMatrix(h)
            return tdl.core.SimpleNamespace(value=value, linop=linop)

        @tdl.core.LazzyProperty
        def A(self):
            # A = basis^T @ k11inv @ basis
            basis = tf.convert_to_tensor(self.basis)
            A = self.basis.linop.matmul(
                tdl.linalg.solvemat(self.k11.cholesky, basis),
                adjoint=True)
            if self.prior_scale:
                shift = self.gp_model.tolerance + 1.0/(self.prior_scale**2)
            else:
                shift = self.gp_model.tolerance
            A = tdl.core.array.add_diagonal_shift(A, shift)
            A_cholesky = Cholesky(A)
            return tdl.core.SimpleNamespace(
                value=A, cholesky=A_cholesky,
                log_det=A_cholesky.log_abs_determinant())

        @tdl.core.LazzyProperty
        def yCy(self):
            ''' y^T @ Kinv @ H @ Ainv @ H^t @ Kinv @ y'''
            y = tf.squeeze(self.gp_model.ym)
            # ky = Kinv @ y
            ky = tdl.linalg.solvevec(self.k11.cholesky, y)
            # Ainv @ H^t @ ky
            hky = self.basis.linop.matvec(ky, adjoint=True)
            ahky = tdl.linalg.solvevec(self.A.cholesky, hky)
            return tf.reduce_sum(hky * ahky)

    @tdl.core.ModelMethod(['loss'], [], MarginalOutput)
    def marginal_likelihood(self, object):
        yCy = object.yCy
        log_prob = 0.5*yCy - 0.5*object.A.log_det
        loss = object.gp_marginal.loss - log_prob
        return loss

    class GpOutput(tdl.core.TdlModel):
        @property
        def gp_model(self):
            return self.model.gp_model

        @property
        def tolerance(self):
            return self.model.gp_model.tolerance

        @tdl.core.InputArgument
        def model(self, value):
            return value

        @tdl.core.InputArgument
        def inputs(self, value):
            return value

        @tdl.core.LazzyProperty
        def gp_output(self):
            return self.gp_model.predict(self.inputs)

        @property
        def k11(self):
            return self.gp_output.k11

        @property
        def k12(self):
            return self.gp_output.k12

        @tdl.core.LazzyProperty
        def k21_k11inv(self):
            k11_inv = tf.convert_to_tensor(self.gp_output.k11_inv)
            k21 = tf.convert_to_tensor(self.gp_output.k21)
            return tf.matmul(k21, k11_inv)

        @tdl.core.LazzyProperty
        def basis(self):
            train = self.model.explicit_basis(self.gp_model.xm)
            train = tdl.core.SimpleNamespace(
                value=tf.convert_to_tensor(train), output=train,
                linop=tf.linalg.LinearOperatorFullMatrix(train))
            test = self.model.explicit_basis(self.inputs)
            test = tdl.core.SimpleNamespace(
                value=tf.convert_to_tensor(test), output=test,
                linop=tf.linalg.LinearOperatorFullMatrix(test))
            return tdl.core.SimpleNamespace(train=train, test=test)

        @tdl.core.LazzyProperty
        def beta_r(self):
            h_test = tf.convert_to_tensor(self.basis.test)
            h_train = tf.convert_to_tensor(self.basis.train)
            beta_r = self.k12.linop.matmul(
                adjoint=True,
                x=tdl.linalg.solvemat(self.k11.cholesky, h_train))
            value = h_test - beta_r
            linop = tf.linalg.LinearOperatorFullMatrix(value)
            return tdl.core.SimpleNamespace(value=value, linop=linop)

        @tdl.core.LazzyProperty
        def ht_k11inv_h(self):
            # H^t @ K11^-1 @ H = (L11 \ H)^t (L11 \ H)
            H = tf.convert_to_tensor(self.basis.train)
            v = self.k11.cholesky.solve(H)
            # value = v^t @ v
            vop = tf.linalg.LinearOperatorFullMatrix(v)
            value = vop.matmul(v, adjoint=True)
            if self.model.prior_scale:
                shift = self.tolerance + 1.0/(self.model.prior_scale**2)
            else:
                shift = self.tolerance
            value = tdl.core.array.add_diagonal_shift(value, shift)
            return tdl.core.SimpleNamespace(
                value=value, cholesky=Cholesky(value))

        @tdl.core.LazzyProperty
        def beta_mean(self):
            H = self.basis.train
            y = tf.squeeze(self.gp_model.ym)
            y_hk = H.linop.matvec(
                tdl.linalg.solvevec(self.k11.cholesky, y),
                adjoint=True)
            beta = tdl.linalg.solvevec(
                self.ht_k11inv_h.cholesky, y_hk)
            return beta

    class EGPPosterior(GpOutput, tdlb.distributions.MVN):
        @tdl.core.LazzyProperty
        def loc(self):
            return tf.squeeze(self.gp_output.loc) \
                  + self.beta_r.linop.matvec(self.beta_mean)

        @tdl.core.LazzyProperty
        def covariance(self):
            ''' cov = gp_cov + R @ (H @ Ky^-1 @ H^t)^-1 @ R^T '''
            R = tf.convert_to_tensor(self.beta_r)
            v = self.ht_k11inv_h.cholesky.solve(
                tdl.core.array.transpose_rightmost(R))
            vop = tf.linalg.LinearOperatorFullMatrix(v)
            covariance = (self.gp_output.covariance
                          + vop.matmul(v, adjoint=True))
            if self.tolerance is None:
                tolerance = self.model.tolerance
            else:
                tolerance = self.tolerance
            return tdl.core.array.add_diagonal_shift(
                covariance, tolerance)

    def predict(self, inputs):
        """Compute the posterior distribution for p(f*| x*, y, X)
        where {y, X} is the training dataset of the GP model.
        Args:
            inputs (tf.Tensor, TdlModel): matrix with test inputs x*.
        Returns:
            GpOutput: TdlModel with the value of p(f*| x*, y, X).
                loc: mean for f*
                scale: scale for f*
        """
        return GpWithExplicitMean.EGPPosterior(model=self, inputs=inputs)
        # if self.gp_model.y_scale is not None:
        #     y_cov = tdl.linalg.M_times_Mt(gp.y_scale)\
        #                .add_to_tensor(covariance)
        #     y = tdlb.distributions.MVN(loc=loc, covariance=y_cov)
        # else:
        #     y = None
        # posterior = tdl.core.SimpleNamespace(y=y, f=dist)
        # return loc, scale, posterior


class VariationalGP(tdl.core.TdlModel):
    @tdl.core.InputParameter
    def tolerance(self, value):
        if value is None or isinstance(value, float):
            value = (value if value is not None
                     else tdl.core.global_options.tolerance)
        return value

    @tdl.core.InputParameter
    def y_scale(self, value, AutoType=None):
        if AutoType is None:
            AutoType = tdl.constrained.PositiveVariableExp
        if value is None:
            value = 0.3
        if isinstance(value, (int, float, np.ndarray)):
            tdl.core.assert_initialized_if_available(
                self, 'y_scale', ['batch_shape'])
            if tdl.core.is_property_set(self, 'batch_shape'):
                if isinstance(value, (int, float)):
                    value = np.full(shape=self.batch_shape, fill_value=value)
                assert value.shape == tuple(self.batch_shape.as_list())
            value = AutoType(value)
        if value is not None:
            value = DynamicScaledIdentity(multipliers=value,
                                          tolerance=self.tolerance)
        return value

    @tdl.core.InputArgument
    def m(self, value):
        ''' number of inducing points '''
        if value is None:
            value = self.fm.loc.shape[0].value
        return value

    @tdl.core.InputArgument
    def input_dim(self, value):
        if value is None:
            value = self.xm.shape[1:]
        return value

    @tdl.core.InputArgument
    def batch_shape(self, value):
        ''' number of batched distributions '''
        if value is None:
            tdl.core.assert_any_available(
                self, 'batch_shape', ['fm', 'y_scale'])
            if tdl.core.is_property_set(self, 'fm'):
                return self.fm.shape[:-1]
            else:
                return self.y_scale.batch_shape
        elif isinstance(value, tf.TensorShape):
            return value
        else:
            return tf.TensorShape(value)

    @tdl.core.SimpleParameter
    def fm(self, value):
        ''' variational parameters for the GP model '''
        return tdlb.distributions.MVN(
            shape=self.batch_shape.concatenate(self.m),
            scale=tdl.AutoVariable(),
            loc=tdl.AutoVariable(),
            tolerance=2*self.tolerance)

    @tdl.core.SimpleParameter
    def xm(self, value):
        ''' inducing points for the GP model '''
        if value is not None:
            return value
        if isinstance(self.input_dim, int):
            xinit = tf.truncated_normal(shape=[self.m, self.input_dim])
        else:
            xinit = tf.truncated_normal(
                shape=tf.concat([[self.m], self.input_dim], axis=0))
        return tdl.core.variable(xinit)

    @tdl.core.Submodel
    def kernel(self, value):
        if value is None:
            tdl.core.assert_initialized(self, 'kernel', ['m'])
            value = tdl.kernels.GaussianKernel(
                l_scale=np.power(1.0/self.m, 1.0/np.prod(self.input_dim)),
                f_scale=1.0, name='kernel')
        return value

    @tdl.core.PropertyShortcuts({'model': ['fm', 'xm', 'kernel']})
    class VGPInference(tdl.core.TdlModel):
        @tdl.core.LazzyProperty
        def y_scale(self):
            scale = (self.model.y_scale(self.inputs)
                     if self.model.y_scale is not None
                     else None)
            return scale

        @tdl.core.InputModel
        def model(self, value):
            return value

        @tdl.core.InputArgument
        def inputs(self, value):
            return value

        @tdl.core.LazzyProperty
        def kmm(self):
            kmm = self.kernel.evaluate(self.xm, self.xm)
            kmm = tdl.core.array.add_diagonal_shift(kmm, self.model.tolerance)
            return tdl.core.SimpleNamespace(value=kmm,
                                            cholesky=Cholesky(kmm))

        @tdl.core.LazzyProperty
        def kmx(self):
            kmx = self.kernel.evaluate(self.xm, self.inputs)
            linop = tf.linalg.LinearOperatorFullMatrix(
                tf.convert_to_tensor(kmx))
            return tdl.core.SimpleNamespace(value=kmx, linop=linop)

    class VGPEstimate(VGPInference, tdlb.distributions.MVN):
        @tdl.core.LazzyProperty
        def loc(self):
            ''' Kxm @ inv(Kmm) @ fm.loc '''
            return self.kmx.linop.matvec(
                tdl.linalg.solvevec(self.kmm.cholesky, self.fm.loc),
                adjoint=True)

        @tdl.core.LazzyProperty
        def covariance(self):
            # Kxm @ inv(Kmm) @ u
            kxx = self.kernel.evaluate(self.inputs, self.inputs)
            # Kxm @ inv(Kmm) @ Kmx
            kxmx = self.kmx.linop.matmul(
                tdl.linalg.solvemat(self.kmm.cholesky, self.kmx),
                adjoint=True)
            # Kxm @ inv(Kmm) @ A @ inv(Kmm) @ Kmx
            kxmamx = tdl.linalg.M_times_Mt(
                self.kmx.linop.matmul(
                    tdl.linalg.solvemat(self.kmm.cholesky, self.fm.scale),
                    adjoint=True))
            covariance = kxx - kxmx + kxmamx
            if self.tolerance is None:
                tolerance = self.model.tolerance
            else:
                tolerance = self.tolerance
            return tdl.core.array.add_diagonal_shift(
                covariance, tolerance)

        def neg_elbo(self, labels, dataset_size):
            return VariationalGP.VGPElbo(posterior=self, labels=labels,
                                         dataset_size=dataset_size)

    @tdl.core.PropertyShortcuts(
        {'posterior': ['model', 'y_scale', 'inputs', 'kmm', 'kmx',
                       'fm', 'xm', 'kernel']})
    class VGPElbo(tdl.core.TdlModel):
        @tdl.core.InputModel
        def posterior(self, value):
            return value

        @tdl.core.InputArgument
        def labels(self, value):
            ''' output samples '''
            return value

        @tdl.core.InputArgument
        def dataset_size(self, value):
            ''' number of samples in the entire dataset '''
            return value

        @tdl.core.LazzyProperty
        def batch_size(self):
            ''' number of samples in a single mini-batch '''
            return tf.cast(tf.shape(self.inputs)[0],
                           tf.convert_to_tensor(self.inputs).dtype)

        @tdl.core.LazzyProperty
        def _evidence(self):
            # loc = Kxm @ inv(Kmm) @ fm.loc
            loc = self.kmx.linop.matvec(
                tdl.linalg.solvevec(self.kmm.cholesky, self.fm.loc),
                adjoint=True)
            if isinstance(self.y_scale,
                          tf.linalg.LinearOperatorScaledIdentity):
                normal = tdlb.distributions.MVNScaledIdentity(
                    loc=loc, scale=self.y_scale.multiplier)
            elif tdl.linalg.is_diagonal_linop(self.y_scale):
                normal = tdlb.distributions.MVNDiag(
                    loc=loc, scale=self.y_scale.diag_part())
            else:
                normal = tdlb.distributions.MVN(
                    loc=loc, scale=self.y_scale)
            return tdl.core.SimpleNamespace(
                value=normal.log_prob(self.labels),
                dist=normal)

        @tdl.core.LazzyProperty
        def fm_prior(self):
            '''prior of variational parameters fm '''
            return tdlb.distributions.MVN(
                loc=tf.zeros(shape=tf.convert_to_tensor(self.kmm).shape[:-1]),
                covariance=tf.convert_to_tensor(self.kmm))

        @tdl.core.LazzyProperty
        def fm_kl(self):
            # fm = tdlb.distributions.MVN(
            #     loc=self.fm.loc, covariance=self.fm.covariance)
            return tdlb.losses.KLDivergence(self.fm, self.fm_prior)

        @tdl.core.LazzyProperty
        def _elbo_tr(self):
            kii = self.kernel.batch_eval(self.inputs, self.inputs)
            # tr1 = Tr( inv(Cov_y) @ Kxx )
            #       - frob(inv(Lmm) @ Kmx )^2
            if tdl.linalg.is_diagonal_linop(self.y_scale):
                y_cov = tdl.linalg.M_times_Mt(self.y_scale)
                tr1_a = tf.reduce_sum(kii/y_cov.diag_part(), axis=-1)
            else:
                # TODO: add implementation for non diagonal
                raise NotImplementedError(
                    'elbo not implemented for non-diagonal y_scale')
            tr1_b = _frob_squared(
                self.y_scale.solve(
                    self.kmm.cholesky.linop.solve(self.kmx.value),
                    adjoint_arg=True))
            tr1 = tr1_a - tr1_b
            # tr2 = frob( inv(Ly) @ Kxm @ inv(Kmm) @ La )^2
            tr2 = _frob_squared(
                self.y_scale.solve(
                    self.kmx.linop.matmul(
                        tdl.linalg.solvemat(self.kmm.cholesky, self.fm.scale),
                        adjoint=True)))
            return (tr1 + tr2)

        @tdl.core.LazzyProperty
        def elbo(self):
            fm_kl = tf.convert_to_tensor(self.fm_kl)
            evidence = tf.convert_to_tensor(self._evidence)
            return ((evidence - 0.5*self._elbo_tr)/self.batch_size
                    - fm_kl/tf.cast(self.dataset_size, fm_kl.dtype))

        @tdl.core.LazzyProperty
        def loss(self):
            return -self.elbo

        @tdl.core.OutputValue
        def value(self, _):
            return -self.elbo

        @tdl.core.LazzyProperty
        def shape(self):
            tdl.core.assert_initialized(self, 'shape', ['value'])
            return self.value.shape

    def predict(self, inputs, tolerance=None):
        return VariationalGP.VGPEstimate(model=self, inputs=inputs,
                                         tolerance=tolerance)

    def neg_elbo(self, labels, inputs, dataset_size):
        posterior = VariationalGP.VGPEstimate(model=self, inputs=inputs)
        return posterior.neg_elbo(labels=labels, dataset_size=dataset_size)

    def __init__(self, m=None, input_dim=None, name=None, **kargs):
        if m is not None:
            kargs['m'] = m
        if input_dim is not None:
            kargs['input_dim'] = input_dim
        super(VariationalGP, self).__init__(name=name, **kargs)


class ExplicitVGP(tdl.core.TdlModel):
    @tdl.core.InputParameter
    def tolerance(self, value):
        if value is None or isinstance(value, float):
            value = (value if value is not None
                     else tdl.core.global_options.tolerance)
        return value

    @tdl.core.InputArgument
    def basis_dim(self, value):
        ''' number of dimensions of the explicit basis '''
        if tdl.core.is_property_set(self, 'beta_prior'):
            value = self.beta_prior.event_shape[-1].value
        if tdl.core.is_property_set(self, 'beta'):
            value = self.beta.event_shape[-1].value
        if value is None:
            tdl.core.assert_initialized(self, 'basis_dim',
                                        ['basis', 'input_dim'])
            if isinstance(self.basis, tdl.kernels.ConcatOnes):
                tdl.core.assert_initialized(self, 'basis_dim', ['input_dim'])
                value = self.input_dim + 1
            else:
                raise ValueError('basis_dim has not been specified')
        return value

    @tdl.core.InputModel
    def basis(self, value):
        if value is None:
            value = tdl.kernels.ConcatOnes()
        assert (callable(value)), 'explicit basis should be callable'
        return value

    @tdl.core.InputModel
    def beta_prior(self, value, AutoType=None):
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

    @tdl.core.SimpleParameter
    def beta(self, value):
        ''' linear weights '''
        if value is None:
            tdl.core.assert_initialized(self, 'beta', ['beta_prior'])
            value = tdlb.distributions.MVN(
                shape=self.batch_shape.concatenate(self.basis_dim),
                scale=tdl.AutoVariable(),
                # loc=tdl.AutoVariable(),
                loc=(tf.convert_to_tensor(self.beta_prior.loc),
                     tdl.AutoVariable()),
                tolerance=self.tolerance)
        return value

    @tdl.core.InputArgument
    def input_dim(self, value):
        ''' number of input dimensions '''
        if tdl.core.is_property_set(self, 'xm'):
            value = tf.convert_to_tensor(self.xm).shape[-1].value
        if value is None:
            raise ValueError('input_dim has not been specified')
        if not isinstance(value, int):
            raise TypeError('input_dim should be an integer')
        return value

    @tdl.core.InputArgument
    def batch_shape(self, value):
        ''' number of output dimensions '''
        if value is None:
            value = 1
        if not isinstance(value, tf.TensorShape):
            value = tf.TensorShape(value)
        return value

    @tdl.core.InputArgument
    def y_scale(self, value, AutoType=None):
        ''' Measurement scale y \sim N(g, y_scale**2 I) '''
        if AutoType is None:
            AutoType = tdl.constrained.PositiveVariableExp
        if value is None:
            value = 1.0
        if isinstance(value, (int, float, np.ndarray)):
            tdl.core.assert_initialized_if_available(
                self, 'y_scale', ['batch_shape'])
            if tdl.core.is_property_set(self, 'batch_shape'):
                if isinstance(value, (int, float)):
                    value = np.full(shape=self.batch_shape, fill_value=value)
                assert value.shape == tuple(self.batch_shape.as_list())
            value = AutoType(value)
        if value is not None:
            value = DynamicScaledIdentity(multipliers=value,
                                          tolerance=self.tolerance)
        return value

    @tdl.core.InputArgument
    def m(self, value):
        ''' number of inducing points '''
        if tdl.core.is_property_set(self, 'xm'):
            return tf.convert_to_tensor(self.xm).shape[-1].value
        if value is None:
            raise ValueError('m has not been specified')
        if not isinstance(value, int):
            raise TypeError('m should be an integer')
        return value

    @tdl.core.SimpleParameter
    def xm(self, value):
        ''' inducing input points for the model '''
        if value is None:
            xinit = tf.truncated_normal(shape=[self.m, self.input_dim])
            value = tdl.core.variable(xinit)
        return value

    @tdl.core.SimpleParameter
    def gm(self, value):
        ''' inducing observed points for the model '''
        if value is None:
            value = tdlb.distributions.MVN(
                shape=self.batch_shape.concatenate(self.m),
                scale=tdl.AutoVariable(),
                loc=tdl.AutoVariable(),
                tolerance=2*self.tolerance)
        return value

    @tdl.core.Submodel
    def kernel(self, value):
        if value is None:
            tdl.core.assert_initialized(self, 'kernel', ['m'])
            value = tdl.kernels.GaussianKernel(
                l_scale=np.power(1.0/self.m, 1.0/np.prod(self.input_dim)),
                f_scale=1.0, name='kernel')
        return value

    class EVGPInference(tdl.core.TdlModel):
        @property
        def gm(self):
            return self.model.gm

        @property
        def xm(self):
            return self.model.xm

        @property
        def kernel(self):
            return self.model.kernel

        @tdl.core.InputModel
        def model(self, value):
            ''' ExplicitVGP model '''
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
        def kmm(self):
            kmm = self.kernel.evaluate(self.xm, self.xm)
            kmm = tdl.core.array.add_diagonal_shift(kmm, self.model.tolerance)
            return tdl.core.SimpleNamespace(
                value=kmm, cholesky=Cholesky(kmm))

        @tdl.core.LazzyProperty
        def kmx(self):
            kmx = self.kernel.evaluate(self.xm, self.inputs)
            linop = tf.linalg.LinearOperatorFullMatrix(
                tf.convert_to_tensor(kmx))
            return tdl.core.SimpleNamespace(value=kmx, linop=linop)

        @tdl.core.LazzyProperty
        def basis_x(self):
            return self.model.basis(self.inputs)

        @tdl.core.LazzyProperty
        def basis_m(self):
            return self.model.basis(self.xm)

        @staticmethod
        def _At_times_A(A):
            linop = tf.linalg.LinearOperatorFullMatrix(A)
            return linop.matmul(A, adjoint=True)

        @staticmethod
        def _A_times_At(A):
            linop = tf.linalg.LinearOperatorFullMatrix(A)
            return linop.matmul(A, adjoint_arg=True)

        @tdl.core.Submodel
        def mu_f_given_a(self, _):
            ''' Kxm @ inv(Kmm) @ gm.loc'''
            return self.kmx.linop.matvec(
                tdl.linalg.solvevec(self.kmm.cholesky, self.gm.loc),
                adjoint=True)

        @tdl.core.Submodel
        def cov_f_given_gm(self, _):
            # Kxm @ inv(Kmm) @ u
            kxx = self.kernel.evaluate(self.inputs, self.inputs)
            # Kxm @ inv(Kmm) @ Kmx
            kxmx = tdl.linalg.Mt_times_M(
                self.kmm.cholesky.linop.solve(self.kmx))
            return kxx - kxmx

        @tdl.core.Submodel
        def r(self, _):
            basis_x = tf.convert_to_tensor(self.basis_x)
            basis_m = tf.convert_to_tensor(self.basis_m)
            r = basis_x - self.kmx.linop.matmul(
                tdl.linalg.solvemat(self.kmm.cholesky, basis_m),
                adjoint=True)
            return tdl.core.SimpleNamespace(
                value=r, linop=tf.linalg.LinearOperatorFullMatrix(r))

    @tdl.core.PropertyShortcuts({'posterior': ['model', 'inputs']})
    class EVGPElbo(tdl.core.TdlModel):
        @tdl.core.InputModel
        def posterior(self, value):
            return value

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

        @tdl.core.LazzyProperty
        def batch_size(self):
            ''' number of samples in a single mini-batch '''
            return tf.cast(tf.shape(self.inputs)[0],
                           tf.convert_to_tensor(self.inputs).dtype)

        @tdl.core.LazzyProperty
        def elbo(self):
            obj = self.posterior
            dist = tdlb.distributions.MVNDiag(
                loc=(obj.r.linop.matvec(self.model.beta.loc)
                     + obj.mu_f_given_a),
                scale=obj.y_scale.diag_part())
            # Tr(M3 A) = frob( inv(Ly) @ Kxm @ inv(Kmm) @ La )**2
            Tr_M3_A = _frob_squared(
                obj.y_scale.solve(
                    obj.kmx.linop.matmul(
                        tdl.linalg.solvemat(obj.kmm.cholesky,
                                            self.model.gm.scale),
                        adjoint=True)))

            # Tr(M1 B) = frob( inv(Ly) @ r @ Lb )**2
            # TODO: profile with Tr(M1 B) implementation
            Tr_M1_B = _frob_squared(
                obj.y_scale.solve(
                    obj.r.linop.matmul(self.model.beta.scale)))
            # Tr( inv(Cov_y) @ cov_f_given_gm
            Tr_Covy_Covg = tf.linalg.trace(
                tdl.linalg.solvemat(obj.y_scale, obj.cov_f_given_gm))

            # KL_gm
            _KL_gm1 = tdlb.losses.KLDivergence(
                self.model.gm,
                tdlb.distributions.MVN(
                    loc=obj.basis_m.linop.matvec(self.model.beta.loc),
                    covariance=obj.kmm.value))
            _KL_gm2 = 0.5 * _frob_squared(
                obj.kmm.cholesky.linop.solve(
                    obj.basis_m.linop.matmul(self.model.beta.scale)))
            KL_gm = _KL_gm1 + _KL_gm2
            # KL_b
            KL_beta = tdlb.losses.KLDivergence(
                self.model.beta, self.model.beta_prior)

            # elbo
            evidence = (dist.log_prob(self.labels)
                        - 0.5*(Tr_M3_A + Tr_M1_B + Tr_Covy_Covg))
            return (evidence/self.batch_size
                    - (KL_gm + KL_beta)/self.dataset_size)

        @tdl.core.OutputValue
        def value(self, _):
            ''' negative variational lower bound '''
            return - self.elbo

    class EVGPPrediction(EVGPInference, tdlb.distributions.MVN):
        @tdl.core.LazzyProperty
        def loc(self):
            ''' Kxm @ inv(Kmm) @ gm.loc + R @ beta.mu'''
            return (self.mu_f_given_a
                    + self.r.linop.matvec(self.model.beta.loc))

        @tdl.core.LazzyProperty
        def covariance(self):
            ''' cov_f_given_gm
                + R @ beta.cov @ R^t
                + Kxm @ inv(Kmm) @ A @ inv(Kmm) @ Kmx'''
            term1 = self.cov_f_given_gm
            term2 = tdl.linalg.M_times_Mt(
                self.r.linop.matmul(self.model.beta.scale))
            # term2 = self.r.linop.matmul(
            #     self.model.beta.covariance.linop.matmul(
            #         self.r, adjoint_arg=True))
            term3 = tdl.linalg.M_times_Mt(
                self.kmx.linop.matmul(
                    tdl.linalg.solvemat(self.kmm.cholesky, self.gm.scale),
                    adjoint=True))
            covariance = term1 + term2 + term3
            if self.tolerance is None:
                tolerance = self.model.tolerance
            else:
                tolerance = self.tolerance
            return tdl.core.array.add_diagonal_shift(
                covariance, tolerance)

        def neg_elbo(self, labels, dataset_size=None):
            return ExplicitVGP.EVGPElbo(posterior=self, labels=labels,
                                        dataset_size=dataset_size)

    def neg_elbo(self, inputs, labels, dataset_size=None):
        posterior = ExplicitVGP.EVGPPrediction(model=self, inputs=inputs)
        return posterior.neg_elbo(labels=labels, dataset_size=dataset_size)

    def predict(self, inputs, tolerance=None):
        return ExplicitVGP.EVGPPrediction(model=self, inputs=inputs,
                                          tolerance=tolerance)
