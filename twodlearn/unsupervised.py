import tensorflow as tf
import twodlearn as tdl
from twodlearn import kernels, losses


class Tsne(tdl.core.TdlModel):
    _submodels = ['pj_given_i', 'pij', 'perplexity', 'qij', 'loss',
                  'perplexity_loss']

    def _init_options(self, options):
        default = {'tolerance': 1e-10,
                   'y/init/stddev': 10e-4}
        options = tdl.core.check_defaults(options, default)
        options = super(Tsne, self)._init_options(options)
        return options

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def target_perplexity(self):
        return self._target_perplexity

    @property
    def tolerance(self):
        return self.options['tolerance']

    @staticmethod
    def diag_to_zero(x):
        # return x*(1-tf.eye(x.shape[0].value))
        return tf.matrix_set_diag(x, tf.zeros([x.shape[0].value]))

    @tdl.core.InputArgument
    def x(self, value):
        self._n_samples = value.shape[0]
        return tf.convert_to_tensor(value)

    @tdl.core.SimpleParameter
    def sigma(self, value):
        sigma = tdl.core.PositiveVariable(
            lambda x: tf.constant(x, shape=[self.n_samples, 1]),
            initial_value=10.0)
        return sigma

    @tdl.core.SimpleParameter
    def y(self, value):
        y = tf.Variable(tf.truncated_normal(
                            shape=[self.n_samples, self.output_dim],
                            stddev=self.options['y/init/stddev']))
        return y

    @tdl.core.Submodel
    def pj_given_i(self, value):
        K = tf.negative(kernels.PairwiseL2(self.x, self.x))/(2*self.sigma**2)
        num = self.diag_to_zero(tf.exp(K))
        den = tf.reduce_sum(num, axis=1)
        pji = num/tf.expand_dims(den, 1)
        self.K = K
        self.num = num
        self.den = den
        return pji

    @tdl.core.Submodel
    def pij(self, value):
        pji = self.pj_given_i
        return (pji + tf.transpose(pji)) / (2.0*self.n_samples)

    @tdl.core.Submodel
    def perplexity(self, value):
        pji = self.pj_given_i
        log_pji = self.diag_to_zero(pji * (tf.log(pji + self.tolerance) /
                                           tf.log(2.0)))
        entropy = - tf.reduce_sum(log_pji, 1)
        return tf.pow(2.0, entropy)

    @tdl.core.Submodel
    def qij(self, value):
        num = 1.0/tf.add(1.0, kernels.PairwiseL2(self.y, self.y))
        num = self.diag_to_zero(num)
        den = tf.reduce_sum(num, axis=1)
        return num/den

    @tdl.core.Submodel
    def loss(self, value):
        K = self.pij * (tf.log(self.pij + self.tolerance) -
                        tf.log(self.qij + self.tolerance))
        K = self.diag_to_zero(K)
        return tf.reduce_sum(K)

    @tdl.core.Submodel
    def perplexity_loss(self, value):
        return losses.L2Loss(y=self.perplexity,
                             labels=self.target_perplexity)

    def __init__(self, output_dim, x, target_perplexity=10, name='tsne',
                 options=None):
        self._output_dim = output_dim
        self._target_perplexity = target_perplexity
        super(Tsne, self).__init__(x=x, name=name, options=options)
