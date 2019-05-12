from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf
import twodlearn as tdl
from twodlearn.templates.supervised import (Supervised, MlModel)
from twodlearn.datasets.mnist import MnistDataset
import twodlearn.datasets.cifar10 as Cifar10
import twodlearn.monitoring as tdlm


class SupervisedObjective(object):
    def __init__(self, op, loss, fit_loss, labels):
        self.op = op
        self.loss = loss
        self.fit_loss = fit_loss
        self.labels = labels


class MnistMlp(MlModel):
    def _init_options(self, options):
        default = {'model/n_inputs': 28*28,
                   'model/n_outputs': 10,
                   'model/n_hidden': [50, 50],
                   'model/afunction': tf.nn.softplus,
                   'model/regularizer/scale': 10000}
        options = tdl.common.check_defaults(options, default)
        options = super(MnistMlp, self)._init_options(options)
        return options

    def _init_model(self):
        model = tdl.feedforward.MlpClassifier(
            n_inputs=self.options['model/n_inputs'],
            n_classes=self.options['model/n_outputs'],
            n_hidden=self.options['model/n_hidden'],
            afunction=self.options['model/afunction'])
        return model

    def _init_train_model(self):
        x = tf.placeholder(tf.float32)
        train = self.model(x, name=self.model.scope)
        with tf.name_scope('loss'):
            fit_loss = tdl.losses.ClassificationLoss(train.output.affine)
            reg_loss = self.model.regularizer.init(
                self.options['model/regularizer/scale'])
            loss = tdl.losses.EmpiricalWithRegularization(fit_loss, reg_loss)

        train = SupervisedObjective(op=train,
                                    loss=loss,
                                    fit_loss=fit_loss,
                                    labels=loss.labels)
        return train

    def _init_training_monitor(self, logger_path):
        # monitoring
        ml_monitor = tdlm.TrainingMonitorManager(
            log_folder=logger_path,
            tf_graph=self.session.graph)

        train = self.train.fit_loss.accuracy
        train = (train.value if isinstance(train, tdl.losses.Loss)
                 else train)
        ml_monitor.train.add_monitor(
            tdlm.OpMonitor(train,
                           buffer_size=self.options['monitor/loggers/buffer_size'],
                           name="train/loss"))

        valid = self.valid.fit_loss.accuracy
        valid = (valid.value if isinstance(valid, tdl.losses.Loss)
                 else valid)
        ml_monitor.valid.add_monitor(
            tdlm.OpMonitor(
                valid,
                buffer_size=self.options['monitor/loggers/buffer_size'],
                name="valid/loss"))
        return ml_monitor

    def _init_valid_model(self):
        return self.train


class SimpleMnistSupervised(Supervised):
    def _init_options(self, options):
        default = {'train/batch_size': 500,
                   'valid/batch_size': 500,
                   'optim/train/max_steps': 2000}
        options = tdl.common.check_defaults(options, default)
        options = super(SimpleMnistSupervised, self)._init_options(options)
        return options

    def _init_dataset(self):
        dataset = MnistDataset(work_directory='Data/', one_hot=True)
        return dataset

    def _init_ml_model(self, logger_path):
        return MnistMlp(options=self.options, logger_path=logger_path)

    def feed_train(self):
        x, y = self.dataset.train.next_batch(self.options['train/batch_size'])
        feed_dict = {self.ml_model.train.op.inputs: x,
                     self.ml_model.train.labels: y}
        return feed_dict

    def feed_valid(self):
        x, y = self.dataset.valid.next_batch(
            self.options['valid/batch_size'])
        feed_dict = {self.ml_model.valid.op.inputs: x,
                     self.ml_model.valid.labels: y}
        return feed_dict

    def run_training(self):
        self.ml_model.optimizer.run(
            n_train_steps=self.options['optim/train/max_steps'],
            feed_train=self.feed_train,
            feed_valid=self.feed_valid)


class MnistCustomMlp(MnistMlp):
    def _init_options(self, options):
        default = {'model/n_inputs': 28*28,
                   'model/n_outputs': 10,
                   'model/n_hidden': [50, 50],
                   'model/afunction': tf.nn.softplus,
                   'model/regularizer/scale': 10000}
        options = tdl.common.check_defaults(options, default)
        options = super(MnistCustomMlp, self)._init_options(options)
        return options

    def _init_model(self):
        assert len(self.options['model/n_hidden']) == 2,\
            'Only two hidden layers allowed'
        model = tdl.feedforward.StackedModel(layers=None)
        with tf.name_scope(model.scope):
            n_inputs = self.options['model/n_inputs']
            n_outputs = self.options['model/n_outputs']
            n_hidden = self.options['model/n_hidden']

            model.add(tdl.dense.DenseLayer(
                input_shape=n_inputs, units=n_hidden[0],
                activation=self.options['model/afunction']))
            model.add(tdl.dense.DenseLayer(
                input_shape=n_hidden[0], units=n_hidden[1],
                activation=self.options['model/afunction']))
            model.add(tdl.dense.DenseLayer(
                input_shape=n_hidden[1], units=n_outputs,
                activation=tf.nn.softmax))

        return model

    def _init_train_model(self):
        x = tf.placeholder(tf.float32)
        train = self.model(x, name=self.model.scope)
        with tf.name_scope('loss'):
            fit_loss = tdl.losses.ClassificationLoss(train.output.affine)
            reg_loss = self.model.regularizer.init(
                self.options['model/regularizer/scale'])
            loss = tdl.losses.EmpiricalWithRegularization(fit_loss, reg_loss)

        train = SupervisedObjective(op=train,
                                    loss=loss,
                                    fit_loss=fit_loss,
                                    labels=loss.labels)
        return train


class RandomLinear(tdl.feedforward.LinearLayer):
    def _init_weights(self, init_method=None, alpha=None, name='W'):
        if alpha is not None:
            self._alpha = alpha
        else:
            self._alpha = tdl.feedforward.options.weight_initialization_alpha

        if init_method is None:
            self.weight_init_method = tdl.feedforward.options.weight_initialization
        else:
            self.weight_init_method = init_method
        # weight initialization
        alpha = self.alpha
        sigma = self._init_sigma(self.weight_init_method, self.alpha)

        weights = tf.Variable(
            tf.truncated_normal([self.input_shape[-1], self.units],
                                stddev=sigma),
            trainable=False,
            name=name)
        return weights


class MnistTestMlp(MnistCustomMlp):
    def _init_options(self, options):
        default = {'model/n_inputs': 28*28,
                   'model/n_outputs': 10,
                   'model/n_hidden': [50, 200],
                   'model/afunction': tf.nn.softplus,
                   'model/regularizer/scale': 100000}
        options = tdl.common.check_defaults(options, default)
        options = super(MnistTestMlp, self)._init_options(options)
        return options

    def _init_model(self):
        assert len(self.options['model/n_hidden']) == 2,\
            'Only two hidden layers allowed'
        model = tdl.feedforward.StackedModel(layers=None)
        with tf.name_scope(model.scope):
            n_inputs = self.options['model/n_inputs']
            n_outputs = self.options['model/n_outputs']
            n_hidden = self.options['model/n_hidden']

            # model.add(tdl.feedforward.FullyconnectedLayer(
            #    n_inputs, n_hidden[0],
            #    afunction=self.options['model/afunction']))
            model.add(RandomLinear(input_shape=n_inputs,
                                   units=n_hidden[0]))
            model.add(tdl.feedforward.FullyconnectedLayer(
                n_hidden[0], n_hidden[1],
                afunction=self.options['model/afunction']))
            model.add(tdl.feedforward.FullyconnectedLayer(
                n_hidden[1], n_outputs,
                afunction=tf.nn.softmax))

        return model


class MnistSupervised(SimpleMnistSupervised):
    def _init_options(self, options):
        default = {'model/class': MnistCustomMlp,
                   'train/batch_size': 500,
                   'valid/batch_size': 500,
                   'optim/train/max_steps': 2000}
        options = tdl.common.check_defaults(options, default)
        options = super(MnistSupervised, self)._init_options(options)
        return options

    def _init_ml_model(self, logger_path):
        return self.options['model/class'](options=self.options,
                                           logger_path=logger_path)


# ---------------------------------- Cifar10 -------------------------------- #


class Cifar10TestMlp(MnistTestMlp):
    def _init_options(self, options):
        default = {'model/n_inputs': 32*32,
                   'model/n_outputs': 10,
                   'model/n_hidden': [500, 1000],
                   'model/afunction': tf.nn.softplus,
                   'model/regularizer/scale': 100000,
                   'optim/train/learning_rate': 0.001}
        options = tdl.common.check_defaults(options, default)
        options = super(Cifar10TestMlp, self)._init_options(options)
        return options


class Cifar10Supervised(MnistSupervised):
    def _init_dataset(self):
        dataset = Cifar10.read_data_sets('Data/', reshape=False, one_hot=True)
        return dataset

    def feed_train(self, batch_size=None):
        if batch_size is None:
            batch_size = self.options['train/batch_size']
        batch_x, batch_y = self.dataset.train.next_batch(batch_size)
        batch_x = np.mean(batch_x, axis=3)
        batch_x = np.reshape(batch_x, [-1, 32*32])
        feed_dict = {self.ml_model.train.op.inputs: batch_x,
                     self.ml_model.train.labels: batch_y}
        return feed_dict

    def feed_valid(self, batch_size=None):
        if batch_size is None:
            batch_size = self.options['valid/batch_size']
        batch_x, batch_y = self.dataset.test.next_batch(batch_size)
        batch_x = np.mean(batch_x, axis=3)
        batch_x = np.reshape(batch_x, [-1, 32*32])
        feed_dict = {self.ml_model.valid.op.inputs: batch_x,
                     self.ml_model.valid.labels: batch_y}
        return feed_dict
