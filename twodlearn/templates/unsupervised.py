from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import tensorflow as tf
import twodlearn.common
import twodlearn.losses
import twodlearn.templates
import twodlearn.monitoring
import twodlearn.unsupervised
from twodlearn import optim


class Tsne(twodlearn.templates.UnconstrainedOptimization):
    def _init_options(self, options=None):
        default = {
            'optim/n_logging': 100,
            'monitor/loggers/buffer_size': 1000,
            'monitor/logger_path': 'tmp/',
            'train/optim/learning_rate': 0.02,
            'perplexity/optim/learning_rate': 0.02}
        options = twodlearn.common.check_defaults(options, default)
        options = super(Tsne, self)._init_options(options=options)
        return options

    def _init_optimizer(self):
        loss = (self.loss if isinstance(self.loss, tf.Tensor)
                else tf.convert_to_tensor(self.loss))

        optimizer = tf.train.AdamOptimizer(
            self.options['train/optim/learning_rate'])
        optimizer = optim.OptimizationManager(
            session=self.session,
            optimizer=optimizer,
            step_op=optimizer.minimize(loss, var_list=self.model.y),
            monitor_manager=self.monitor,
            n_logging=self.options['optim/n_logging'])
        return optimizer

    def _init_perplexity_optimizer(self):
        loss = tf.convert_to_tensor(self.model.perplexity_loss)
        var_list = twodlearn.common.get_trainable(self.model.sigma)
        # self.options['optim/train/learning_rate']
        optimizer = tf.train.AdamOptimizer(
            self.options['perplexity/optim/learning_rate'])

        self.monitor.train.add_monitor(
            twodlearn.monitoring.OpMonitor(
                loss, name="train/perplexity_loss"))

        optimizer = optim.OptimizationManager(
            session=self.session,
            optimizer=optimizer,
            step_op=optimizer.minimize(loss, var_list=var_list),
            monitor_manager=self.monitor,
            n_logging=self.options['optim/n_logging'])
        return optimizer

    def run(self, max_iter=1000):
        self.optimizer.run(n_train_steps=max_iter)

    def __init__(self, output_dim, x, target_perplexity=10,
                 options=None, session=None, name='tsne'):
        tsne = twodlearn.unsupervised.Tsne(
            output_dim, x, target_perplexity=target_perplexity, name=name,
            options=options)
        super(Tsne, self).__init__(loss=tsne.loss, model=tsne, options=None,
                                   session=session)
        self.perplexity_optimizer = self._init_perplexity_optimizer()
        tf.global_variables_initializer().run()
