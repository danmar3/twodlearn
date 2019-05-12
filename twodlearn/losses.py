#  ***********************************************************************
#   This file defines some common Losses for deterministic models
#      - ConvNet: convolutional neural network
#      - MlpNet: multilayer perceptron
#
#   Wrote by: Daniel L. Marino (marinodl@vcu.edu)
#    Modern Heuristics Research Group (MHRG)
#    Virginia Commonwealth University (VCU), Richmond, VA
#    http://www.people.vcu.edu/~mmanic/
#
#   ***********************************************************************

from __future__ import division
from __future__ import print_function


import tensorflow as tf
import twodlearn as tdl
import warnings


class Loss(tdl.common.TdlModel):
    @property
    def value(self):
        if hasattr(self, '_value'):
            return self._value
        else:
            warnings.warn('use of _loss in {} is misleading, '
                          'use _value instead'.format(type(self)))
            return self._loss

    @value.setter
    def value(self, value):
        assert not hasattr(self, '_value'),\
            'value can only be set once'
        self._value = value

    def __add__(self, other):
        if isinstance(other, Loss):
            return AddLoss(self, other)
        else:
            return (self if (other == 0)
                    else AddLoss(self, other))
    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, Loss):
            return MultipliedLosses(self, other)
        else:
            return ScaledLoss(alpha=other, loss=self)
    __rmul__ = __mul__


class LossMethod(tdl.ModelMethod):
    ''' Decorator used to specify an operation for a loss inside a model.
    The decorator works similar to @property, but the specified
    method correponds to the definition of the operation
    Usage:
    class MyModel(tdl.TdlModel):
        _submodels = ['evaluate']
        @tdl.LossMethod(['y'], # list of outputs
                        ['x']  # list of inputs
                        )
        def mean_loss(self, x):
            return tf.reduce_mean(x)
    '''
    _OutputClass = Loss


class ScaledLoss(Loss):
    @property
    def pre_scaled(self):
        return self._pre_scaled

    @property
    def alpha(self):
        return self._alpha

    def __init__(self, alpha, loss, name='ScaledLoss'):
        super(ScaledLoss, self).__init__(name=name)
        self._alpha = alpha
        self._pre_scaled = loss
        with tf.name_scope(self.scope):
            self._value = alpha * loss.value


class MultipliedLosses(Loss):
    @property
    def loss1(self):
        return self._loss1

    @property
    def loss2(self):
        return self._loss2

    def __init__(self, loss1, loss2, name='MultipliedLosses'):
        super(AddLoss, self).__init__(name=name)
        self._loss1 = loss1
        self._loss2 = loss2
        with tf.name_scope(self.scope):
            self._value = loss1.value * loss2.value


class EmpiricalLoss(Loss):
    @property
    def labels(self):
        ''' Labels for computing the loss, if not provided,
        they are created automatically '''
        if not hasattr(self, '_labels'):
            assert hasattr(self, '_y'), \
                'attempting to create labels with undefined y'
            self._labels = tf.placeholder(tf.float32,
                                          shape=self.y.shape,
                                          name='labels')
        return self._labels


class EmpiricalLossWrapper(EmpiricalLoss):
    @property
    def loss(self):
        return self._loss

    @property
    def value(self):
        return tf.convert_to_tensor(self.loss)

    def __init__(self, loss, labels, name='EmpiricalLoss'):
        self._loss = loss
        self._labels = labels


class AddLoss(Loss):
    @property
    def loss1(self):
        return self._loss1

    @property
    def loss2(self):
        return self._loss2

    def __init__(self, loss1, loss2, name='AddLoss'):
        super(AddLoss, self).__init__(name=name)
        self._loss1 = loss1
        self._loss2 = loss2
        with tf.name_scope(self.scope):
            loss1 = (loss1.value if isinstance(loss1, Loss)
                     else loss1)
            loss2 = (loss2.value if isinstance(loss2, Loss)
                     else loss2)
            self._value = loss1 + loss2


class AddNLosses(Loss):
    @property
    def losses(self):
        return self._losses

    def mean(self):
        return ScaledLoss(alpha=(1.0/len(self.losses)),
                          loss=self)

    def __getitem__(self, item):
        return self.losses[item]

    def __len__(self):
        return len(self.losses)

    def __init__(self, losses, name='AddNLosses'):
        super(AddNLosses, self).__init__(name=name)
        self._losses = losses
        with tf.name_scope(self.scope):
            losses = [(loss.value if isinstance(loss, Loss)
                       else loss) for loss in self.losses]
            self._value = tf.add_n(losses)


class EmpiricalWithRegularization(Loss):
    ''' Linear combination of a Empirical and a Reguralizer loss:
    loss = empirical + alpha * regularizer
    If alpha is None, the result is the sum of the empirical and regularizer
    losses:
    loss = empirical + regularizer
    '''
    @property
    def empirical(self):
        return self._empirical

    @property
    def regularizer(self):
        return self._regularizer

    @property
    def alpha(self):
        return self._alpha

    @property
    def labels(self):
        return self.empirical.labels

    def __init__(self, empirical, regularizer, alpha=None, name='ERLoss'):
        super(EmpiricalWithRegularization, self).__init__(name=name)
        assert isinstance(empirical, EmpiricalLoss),\
            'empirical loss must be an instance of EmpiricalLoss'
        self._empirical = empirical
        self._regularizer = regularizer
        self._alpha = alpha
        with tf.name_scope(self.scope):
            if alpha is None:
                self._value = (tf.convert_to_tensor(self.empirical) +
                               tf.convert_to_tensor(self.regularizer))
            else:
                self._value = (tf.convert_to_tensor(self.empirical) +
                               alpha * tf.convert_to_tensor(self.regularizer))


class ClassificationLoss(EmpiricalLoss):
    _input_args = ['logits', 'labels']

    @tdl.common.InputArgument
    def logits(self, value):
        return value

    @property
    def n_outputs(self):
        return self.logits.shape[1].value

    @property
    def n_classes(self):
        return (2 if self.n_outputs == 1
                else self.n_outputs)

    @tdl.common.InputArgument
    def labels(self, value):
        ''' Labels for computing the loss, if not provided,
        they are created automatically '''
        if value is None:
            value = tf.placeholder(tdl.common.global_options.float.tftype,
                                   shape=self.logits.shape)
        return value

    @tdl.common.Submodel
    def cross_entropy(self, _):
        if self.n_outputs == 1:
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.labels, logits=self.logits)
        else:
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.labels, logits=self.logits)
        return loss

    @tdl.common.OutputValue
    def value(self, _):
        return tf.reduce_mean(self.cross_entropy)

    def __init__(self, logits, labels=None, name=None):
        super(ClassificationLoss, self).__init__(
            logits=logits, labels=labels, name=name)

    @tdl.common.LazzyProperty
    def correct_prediction(self):
        if self.n_outputs == 1:
            return tf.equal(tf.round(self.labels),
                            tf.round(tf.nn.sigmoid(self.logits)))
        else:
            return tf.equal(tf.argmax(self.labels, axis=1),
                            tf.argmax(tf.nn.softmax(self.logits), axis=1))

    @tdl.common.LazzyProperty
    def accuracy(self):
        correct_prediction = tf.cast(self.correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        return accuracy


class L2Loss(EmpiricalLoss):
    ''' computes (1/M)sum( (y - labels)**2 )'''
    @property
    def y(self):
        return self._y

    @property
    def n_outputs(self):
        return self.y.shape[1].value

    def __init__(self, y, labels=None, name='L2Loss'):
        self._name = name
        self._y = y
        with tf.name_scope(self.scope):
            if labels is not None:
                self._labels = labels
            self._value = self.define_fit_loss(self.y, self.labels)

    def define_fit_loss(self, y, labels):
        norm_2 = tf.pow(y - labels, 2)
        if len(norm_2.shape) > 1:
            norm_2 = tf.reduce_sum(norm_2, 1)
        loss = tf.reduce_mean(norm_2, 0)
        return loss


class L2Regularizer(Loss):
    @property
    def weights(self):
        return self._weights

    @property
    def scale(self):
        return self._scale

    def __init__(self, weights, scale=None, name='L2Regularizer'):
        self._name = name
        self._weights = weights
        self._scale = scale
        with tf.name_scope(self.scope):
            self._value = self.define_loss(self.weights, self.scale)

    def define_loss(self, weights, scale):
        if isinstance(weights, list):
            if scale is not None:
                weights = [w/scale for w in weights]
            loss = tf.add_n([tf.nn.l2_loss(w) for w in weights])
        else:
            if scale is not None:
                weights = weights / scale
            loss = tf.nn.l2_loss(weights)
        return loss


class L1Regularizer(L2Regularizer):
    def define_loss(self, weights, scale):
        if isinstance(weights, list):
            if scale is not None:
                weights = [w/scale for w in weights]
            loss = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in weights])
        else:
            if scale is not None:
                weights = weights / scale
            loss = tf.reduce_sum(tf.abs(weights))
        return loss


class QuadraticLoss(Loss):
    ''' Defines a cuadratic loss that takes the form:
    loss = (X-target) q (X-target)^T '''
    _input_args = ['x', 'target']
    _parameters = ['q']

    @tdl.SimpleParameter
    def q(self, value):
        # assert isinstance(value, (np.Array, tf.Tensor)),\
        #    ''
        return value

    @tdl.InputArgument
    def x(self, value):
        return value

    @tdl.InputArgument
    def target(self, value):
        return value

    @LossMethod(['value', 'quadratic_loss', 'target'], [])
    def mean(self, object):
        return tf.reduce_mean(self.value), self, self.target

    def __init__(self, x, q=None, target=None, name='QuadraticLoss'):
        super(QuadraticLoss, self).__init__(x=x, q=q, target=target, name=name)
        with tf.name_scope(self.scope):
            diff = (self.x if self.target is None
                    else self.x-self.target)
            if tdl.common.tensor_rank(self.q) == 2:
                self._value = tf.matmul(tf.matmul(diff, self.q),
                                        tf.transpose(diff))
            else:
                axis = list(range(1, len(diff.shape)))
                self._value = tf.reduce_sum(q*diff**2, axis=axis)


class LessThan(Loss):
    ''' Defines a loss that punishes the variable being smaller than
    a given value
    loss = func(reference - x) '''
    _input_args = ['x', 'reference', 'mask']

    @tdl.InputArgument
    def x(self, value):
        # TODO: check arguments
        return value

    @tdl.InputArgument
    def reference(self, value):
        # TODO: check arguments
        return value

    @tdl.InputArgument
    def mask(self, value):
        # TODO: check arguments
        return value

    @LossMethod(['value', 'lessthan_loss', 'reference'], [])
    def mean(self, object):
        return tf.reduce_mean(self.value), self, self.reference

    def loss_eval(self, x, reference, mask):
        loss = self._func(reference - x)
        if mask is not None:
            loss = loss*mask
        return loss

    def __init__(self, x, reference, mask=None, func=tf.nn.softplus,
                 name='LessThanLoss'):
        self._func = func
        super(LessThan, self).__init__(x=x, reference=reference, mask=mask,
                                       name=name)
        with tf.name_scope(self.scope):
            self._value = self.loss_eval(self.x, self.reference, self.mask)


class GreaterThan(LessThan):
    ''' Defines a loss that punishes the variable being larger than
    a given value
    loss = func(x - reference) '''

    def loss_eval(self, x, reference, mask):
        return self._func(x - reference)

    def __init__(self, x, reference, mask=None, func=tf.nn.softplus,
                 name='GreaterThanLoss'):
        super(GreaterThan, self).__init__(x=x, reference=reference,
                                          func=func, name=name)


def convert_loss_to_tensor(value, dtype=None, name=None, as_ref=False):
    if dtype is None:
        return value.value
    else:
        return tf.convert_to_tensor(value.value, dtype=dtype, name=name)


tf.register_tensor_conversion_function((Loss, EmpiricalLossWrapper),
                                       conversion_func=convert_loss_to_tensor,
                                       priority=100)
