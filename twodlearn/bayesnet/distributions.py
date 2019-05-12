import numpy as np
import tensorflow as tf
import twodlearn as tdl
import collections
from .losses import KLDivergence, RegisterKL
from twodlearn.linalg import Cholesky, PDMatrix, PDMatrixDiag, PDScaledIdentity
try:
    import tensorflow_probability as tfp
    TFP_AVAILABLE = True
except ImportError:
    TFP_AVAILABLE = False


class Normal(tdl.core.TdlModel):
    ''' normal distribution x= loc + scale*e, where e \sim N(0, I) '''

    @tdl.core.InputParameter
    def loc(self, value):
        ''' mean of the distribution '''
        return value

    @tdl.core.InputParameter
    def scale(self, value):
        ''' standard deviation of the distribution '''
        return value

    @tdl.core.Submodel
    def _distribution(self, value):
        if value is None:
            value = tf.distributions.Normal(loc=self.loc, scale=self.scale)
        return value

    def __init__(self, loc, scale, name='McNormal', **kargs):
        super(Normal, self).__init__(loc=loc, scale=scale,
                                     name=name, **kargs)


def is_diagonal(x):
    """Helper to identify if `LinearOperator` has only a diagonal
    component."""
    return (
        isinstance(x, PDMatrixDiag) or
        isinstance(x, tf.linalg.LinearOperatorIdentity) or
        isinstance(x, tf.linalg.LinearOperatorScaledIdentity) or
        isinstance(x, tf.linalg.LinearOperatorDiag))


class MVN(tdl.core.TdlModel):
    ''' multivariate normal distribution'''
    _PDMatrixClass = PDMatrix

    @tdl.core.InputArgument
    def tolerance(self, value):
        ''' added tolerance to the covariance, useful when optimizing
        over the covariance '''
        return value

    @tdl.core.InputArgument
    def shape(self, value):
        ''' shape of the distribution
        The shape assumes the last dimention corresponds to a set of
        mvn vectors.
        Shape is divided as [batch_shape, event_shape]. Event shape is
        the shape of the samples for a single distribution.
        Batch_shape corresponds to the number of independent MVN distributions.
        '''
        if value is None:
            tdl.core.assert_initialized_if_available(
                self, 'shape', ['loc', 'covariance'])
            tdl.core.assert_any_available(self, 'shape', ['loc', 'covariance'])
            if tdl.core.is_property_initialized(self, 'loc'):
                value = tf.convert_to_tensor(self.loc).shape
            if tdl.core.is_property_initialized(self, 'covariance'):
                cov_shape = tf.convert_to_tensor(self.covariance).shape[:-1]
                value = (cov_shape if value is None
                         else tf.broadcast_static_shape(value, cov_shape))
        if not isinstance(value, tf.TensorShape):
            value = tf.TensorShape(value)
        return value

    @tdl.core.LazzyProperty
    def batch_shape(self):
        ''' leading dimensions of shape '''
        tdl.core.assert_initialized(self, 'batch_shape', ['shape'])
        return self.shape[:-1]

    @tdl.core.LazzyProperty
    def event_shape(self):
        ''' last dimension of shape '''
        tdl.core.assert_initialized(self, 'event_shape', ['shape'])
        return self.shape[-1:]

    @tdl.core.LazzyProperty
    def dynamic_shape(self):
        ''' dynamic shape of the distribution
        The shape assumes the last dimention corresponds to a set of
        mvn vectors.
        Shape is divided as [batch_shape, event_shape]. Event shape is
        the shape of the samples for a single distribution.
        Batch_shape corresponds to the number of independent MVN distributions.
        '''
        tdl.core.assert_initialized_if_available(
            self, 'dynamic_shape', ['loc', 'covariance'])
        tdl.core.assert_any_available(
            self, 'dynamic_shape', ['loc', 'covariance'])
        value = None
        if tdl.core.is_property_set(self, 'loc'):
            value = tf.shape(tf.convert_to_tensor(self.loc))
        if tdl.core.is_property_set(self, 'covariance'):
            cov_shape = tf.shape(tf.convert_to_tensor(self.covariance))[:-1]
            value = (cov_shape if value is None
                     else tf.broadcast_dynamic_shape(value, cov_shape))
        return value

    @tdl.core.LazzyProperty
    def dynamic_batch_shape(self):
        ''' leading dimensions of shape '''
        tdl.core.assert_initialized(self, 'dynamic_batch_shape',
                                    ['dynamic_shape'])
        return self.dynamic_shape[:-1]

    @tdl.core.LazzyProperty
    def dynamic_event_shape(self):
        ''' leading dimensions of shape '''
        tdl.core.assert_initialized(
            self, 'dynamic_event_shape', ['dynamic_shape'])
        return self.dynamic_shape[-1:]

    @tdl.core.InputParameter
    def loc(self, value, AutoType=None):
        ''' mean of the distribution '''
        if value is not None and AutoType is not None:
            value = AutoType(value)
        if AutoType is None:
            AutoType = tdl.core.variable
        if value is None:
            tdl.core.assert_initialized(self, 'loc', ['shape'])
            value = AutoType(tf.zeros(self.shape))
        return value

    @tdl.core.InputParameter
    def covariance(self, value, AutoType=None):
        ''' covariance matrix '''
        if AutoType is None:
            AutoType = tdl.core.autoinit.AutoVariable()
        if value is None:
            value = self._PDMatrixClass(
                raw=AutoType,
                shape=self.shape.concatenate(self.shape[-1]),
                tolerance=self.tolerance)
        elif isinstance(value, tf.Tensor):
            value = tdl.core.SimpleNamespace(
                value=value,
                linop=tf.linalg.LinearOperatorFullMatrix(value))
        return value

    @tdl.core.InputParameter
    def scale(self, value, AutoType=None):
        ''' scale matrix y = loc + scale * e
        covariance = scale * scale^t '''
        if value is not None:
            if isinstance(value, (int, float)):
                tdl.core.assert_initialized(self, 'scale', ['shape'])
                self.covariance = self._PDMatrixClass(
                    raw=(value, AutoType),
                    shape=self.shape.concatenate(self.shape[-1]))
            else:
                self.covariance = self._PDMatrixClass(
                    raw=(value, AutoType))
            value = Cholesky(self.covariance)

        else:
            tdl.core.assert_initialized(self, 'scale', ['covariance'])
            value = Cholesky(self.covariance)
        return value

    @tdl.core.LazzyProperty
    def mean(self):
        ''' mean '''
        return self.loc

    @tdl.core.LazzyProperty
    def stddev(self):
        ''' Standard deviation sqrt( diag(covariance) ).
        Notice that this is different from the scale '''
        return tf.sqrt(tf.linalg.diag_part(self.covariance))

    def sample(self, sample_shape=None):
        if sample_shape is None:
            sample_shape = ()
        if not isinstance(sample_shape, tf.TensorShape):
            sample_shape = tf.TensorShape(sample_shape)
        shape = sample_shape.concatenate(self.shape)
        if not shape.is_fully_defined():
            event_shape = tf.broadcast_dynamic_shape(
                tf.shape(self.loc), self.scale.shape_tensor()[:-1])
            shape = tf.concat(
                [tf.constant(sample_shape.as_list(), dtype=event_shape.dtype),
                 event_shape],
                axis=0)
        # return self.loc + tf.matmul(tf.random_normal(shape=shape),
        #                             self.scale)
        return self.loc + self.scale.matvec(tf.random_normal(shape=shape))

    def log_prob(self, value):
        # log_det(A) = 2*sum(log(diag(chol(A))))
        log_det_cov = 2.0*self.scale.linop.log_abs_determinant()
        # H^t @ Cov^-1 @ H = (L \ H)^t (L \ H); L = chol(Cov)
        Lov_h = self.scale.linop.solvevec(value - self.loc)
        ht_Lov_h = tdl.core.array.reduce_sum_rightmost(Lov_h*Lov_h, ndims=1)
        k = self.shape.as_list()[-1]
        if k is None:
            k = tf.cast(self.dynamic_shape[-1], log_det_cov.dtype)
        return -0.5*(log_det_cov + ht_Lov_h + k*tf.log(2.0*np.pi))


class MVNDiag(MVN):
    ''' multivariate normal distribution'''
    _PDMatrixClass = PDMatrixDiag

    @tdl.core.InputParameter
    def diag(self, value, AutoType=None):
        if value is None and AutoType is not None:
            raise NotImplementedError(
                'Auto initialization of diag using an user provided type is '
                'not available. Auto initialize scale instead')
        if AutoType is None:
            AutoType = tdl.core.autoinit.AutoVariable()
        if value is None:
            tdl.core.assert_initialized(self, 'diag', ['scale'])
            value = self.covariance.linop.diag_part()
        else:
            self.covariance = PDMatrixDiag(raw=(tf.sqrt(value), AutoType))
        return value

    @tdl.core.LazzyProperty
    def stddev(self):
        ''' Standard deviation sqrt( diag(covariance) ).
        Notice that this is different from the scale '''
        return self.scale.linop.diag_part()

    def sample(self, sample_shape):
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        shape = tf.concat([sample_shape,
                           tf.constant(self.shape.as_list(), tf.int32)], 0)
        return (self.loc
                + tf.random_normal(shape=shape) * self.scale.linop.diag_part())


class MVNScaledIdentity(MVNDiag):
    ''' Multivariate Normal Distribution with Scaled Identity covariance
    scale correspond to the covariance scale.
    shape is required
    Parameters are instantiated as Trainable by default:
    tdl.bayesnet.distributions.MVNScaledIdentity(
                shape=[10], scale=0.5)

    To instantiate parameters as tensors (not trainable) use AutoInit classes:
    tdl.bayesnet.distributions.MVNScaledIdentity(
        shape=[10],
        scale=(0.5, tdl.AutoTensor()), # initialize scale using 0.5 tensor
        loc=tdl.AutoTensor()) # initialize scale using the default value (0)
    '''
    def _PDMatrixClass(self, raw, shape=None):
        tdl.core.assert_initialized(self, '_PDMatrixClass', ['shape'])
        if self.event_shape[-1].value is None:
            return PDScaledIdentity(
                raw=raw, domain_dimension=self.dynamic_event_shape[-1])
        else:
            return PDScaledIdentity(
                raw=raw, domain_dimension=self.event_shape[-1])


@RegisterKL(MVN, MVN)
@RegisterKL(MVN, MVNDiag)
@RegisterKL(MVNDiag, MVN)
def _kl_mvn_mvn(a_dist, b_dist):
    a_scale_op = (a_dist.scale.linop if isinstance(a_dist, MVN)
                  else a_dist.scale)
    a_mean = (a_dist.mean if isinstance(a_dist, MVN)
              else a_dist.mean())
    b_scale_op = (b_dist.scale.linop if isinstance(b_dist, MVN)
                  else b_dist.scale)
    b_mean = (b_dist.mean if isinstance(b_dist, MVN)
              else b_dist.mean())

    def fnorm_squared(mat):
        return tf.reduce_sum(tf.square(mat), axis=[-2, -1])

    if is_diagonal(a_scale_op) and is_diagonal(b_scale_op):
        a_stddev = (a_dist.stddev if isinstance(a_dist, MVN)
                    else a_dist.stddev())
        b_stddev = (b_dist.stddev if isinstance(a_dist, MVN)
                    else b_dist.stddev())
        lb_inv_la = (a_stddev / b_stddev)[..., tf.newaxis]
    else:
        a_scale = (a_dist.scale.value if isinstance(a_dist, MVN)
                   else a_dist.scale.to_dense())
        lb_inv_la = b_scale_op.solve(a_scale)
    lb_inv_mudiff = b_scale_op.solvevec(b_mean - a_mean)

    kl = (
        b_scale_op.log_abs_determinant()
        - a_scale_op.log_abs_determinant()
        + 0.5*(- tf.cast(a_scale_op.domain_dimension_tensor(),
                         a_scale_op.dtype)
               + fnorm_squared(lb_inv_la)
               + fnorm_squared(lb_inv_mudiff[..., tf.newaxis]))
        )

    return kl


if TFP_AVAILABLE:
    RegisterKL(
        MVN, tfp.distributions.MultivariateNormalFullCovariance)(_kl_mvn_mvn)
    RegisterKL(
        tfp.distributions.MultivariateNormalFullCovariance, MVN)(_kl_mvn_mvn)
