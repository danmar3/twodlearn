import tensorflow as tf
from twodlearn import common, monitoring, optim
import twodlearn.templates.supervised


class UnconstrainedOptimization(common.TdlProgram):
    @property
    def loss(self):
        return self._loss

    @property
    def model(self):
        return self._model

    @property
    def session(self):
        return self._session

    def _init_options(self, options=None):
        default = {
            'optim/n_logging': 100,
            'monitor/loggers/buffer_size': 1000,
            'monitor/logger_path': 'tmp/',
            'optim/train/learning_rate': 0.02}
        options = common.check_defaults(options, default)
        options = super(UnconstrainedOptimization, self)\
            ._init_options(options=options)
        return options

    def _init_monitor(self):
        # monitoring
        monitor = monitoring.TrainingMonitorManager(
            log_folder=self.options['monitor/logger_path'],
            tf_graph=self.session.graph)

        loss = (self.loss if isinstance(self.loss, tf.Tensor)
                else tf.convert_to_tensor(self.loss))
        monitor.train.add_monitor(
            monitoring.OpMonitor(
                loss, buffer_size=self.options['monitor/loggers/buffer_size'],
                name="train/loss"))
        return monitor

    def _init_optimizer(self):
        loss = (self.loss if isinstance(self.loss, tf.Tensor)
                else tf.convert_to_tensor(self.loss))

        optimizer = tf.train.AdamOptimizer(
            self.options['optim/train/learning_rate'])\
            .minimize(loss)
        optimizer = optim.OptimizationManager(
            sess=self.session,
            optimizer_op=optimizer,
            monitor_manager=self.monitor,
            n_logging=self.options['optim/n_logging'])
        return optimizer

    def run(self, max_iter=1000):
        self.optimizer.run(n_train_steps=max_iter)

    def __init__(self, loss, model=None, options=None, session=None):
        super(UnconstrainedOptimization, self).__init__(options=options)
        self._loss = loss
        self._model = model
        self._session = (tf.InteractiveSession() if session is None
                         else session)
        self.monitor = self._init_monitor()
        self.optimizer = self._init_optimizer()
        tf.global_variables_initializer().run()
