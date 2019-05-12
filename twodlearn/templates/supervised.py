from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import warnings
import collections
import tensorflow as tf
import twodlearn as tdl
import twodlearn.losses
import twodlearn.feedforward
import twodlearn.monitoring as tdlm
from twodlearn import optim


class MlModel(tdl.core.TdlProgram):
    @property
    def session(self):
        return self._session

    @property
    def model(self):
        return self._model

    @property
    def train(self):
        return self._train

    @property
    def valid(self):
        return self._valid

    @property
    def test(self):
        return self._test

    @property
    def monitor(self):
        return self._monitor

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def options(self):
        return self._options

    def _init_options(self, options=None, default=None):
        local_default = {
            'optim/n_logging': 100,
            'monitor/loggers/buffer_size': 1000,
            'optim/train/learning_rate': 0.02}

        def update_dict(options, default):
            if options is None:
                options = dict()
            if default is not None:
                for key, value in default.items():
                    if key not in options:
                        options[key] = value
            return options
        default = update_dict(default, local_default)
        options = update_dict(options, default)
        return options

    def _init_model(self):
        raise NotImplementedError('_init_model is not defined in '
                                  'the base MlModel class')

    def _init_train_model(self):
        raise NotImplementedError('_init_train_model is not defined in '
                                  'the base MlModel class')

    def _init_valid_model(self):
        return None

    def _init_test_model(self):
        return None

    def _init_training_monitor(self, logger_path):
        # monitoring
        ml_monitor = tdlm.TrainingMonitorManager(
            log_folder=logger_path,
            tf_graph=self.session.graph)

        train = self.train.fit_loss
        train = (train.value if isinstance(train, twodlearn.losses.Loss)
                 else train)
        ml_monitor.train.add_monitor(
            tdlm.OpMonitor(
                train,
                buffer_size=self.options['monitor/loggers/buffer_size'],
                name="train/loss"))
        if self.valid is not None:
            valid = self.valid.fit_loss
            valid = (valid.value if isinstance(valid, twodlearn.losses.Loss)
                     else valid)
            ml_monitor.valid.add_monitor(
                tdlm.OpMonitor(
                    valid,
                    buffer_size=self.options['monitor/loggers/buffer_size'],
                    name="valid/loss"))
        return ml_monitor

    def _init_optimizer(self, loss=None):
        if loss is None and isinstance(self.train.loss, twodlearn.losses.Loss):
            loss = self.train.loss.value
        else:
            warnings.warn('Loss for class {} is not an instance of tdl.Loss. '
                          'Consider using tdl losses for improved clarity'
                          ''.format(self.train))
            loss = self.train.loss

        optimizer = tf.train.AdamOptimizer(
            self.options['optim/train/learning_rate'])
        optimizer = optim.OptimizationManager(
            session=self.session,
            optimizer=optimizer,
            step_op=optimizer.minimize(loss),
            monitor_manager=self.monitor,
            n_logging=self.options['optim/n_logging'])
        return optimizer

    def _init_session(self, session):
        if session is None:
            session = tf.InteractiveSession()
        return session

    def __init__(self, options=None, logger_path='tmp', session=None):
        self._options = self._init_options(options)
        self._session = self._init_session(session)
        self._model = self._init_model()
        self._train = self._init_train_model()
        self._valid = self._init_valid_model()
        self._test = self._init_test_model()
        self._monitor = self._init_training_monitor(logger_path)
        self._optimizer = self._init_optimizer()
        # tdl decorators
        super(MlModel, self).__init__()
        # initialize variables
        tf.global_variables_initializer().run()
        print('TF Variables Initialized')

    @classmethod
    def default_options(cls):
        dummy = cls.__new__(cls)
        return dummy._init_options(None)


class SupervisedObjective(object):
    def __init__(self, op, loss, fit_loss, labels):
        self.op = op
        self.loss = loss
        self.fit_loss = fit_loss
        self.labels = labels


class SupervisedWrapper(object):
    def __init__(self, model, inputs, loss, labels,
                 params=None):
        self.model = model
        self.inputs = inputs
        self.loss = loss
        self.labels = labels
        if params is None:
            params = dict()
        self.params = params


class SupervisedMlModel(MlModel):
    def fit(self, dataset):
        def h_feed_train():
            x, y = dataset.train.next_batch(self.options['train/batch_size'])
            feed_dict = {self.train.op.inputs: x,
                         self.train.labels: y}
            return feed_dict

        def h_feed_valid():
            x, y = dataset.valid.next_batch(self.options['valid/batch_size'])
            feed_dict = {self.valid.op.inputs: x,
                         self.valid.labels: y}
            return feed_dict
        feed_valid = (h_feed_valid if dataset.valid is not None
                      else None)
        self.ml_model.optimizer.run(
            n_train_steps=self.options['optim/train/max_steps'],
            feed_train=h_feed_train,
            feed_valid=feed_valid)


class Supervised(tdl.core.TdlProgram):
    @property
    def options(self):
        return self._options

    @property
    def dataset(self):
        return self._dataset

    @property
    def tmp_path(self):
        return self._tmp_path

    @property
    def ml_model(self):
        return self._ml_model

    def _init_options(self, options, default=None):
        options = tdl.core.check_defaults(options, default)
        if not type(options) == tdl.core.Options:
            options = tdl.core.Options(options)
        return options

    def feed_train(self):
        x, y = self.dataset.train.next_batch(self.options['train/batch_size'])
        feed_dict = {self.ml_model.train.op.inputs: x,
                     self.ml_model.train.labels: y}
        return feed_dict

    def feed_valid(self):
        x, y = self.dataset.valid.next_batch(self.options['valid/batch_size'])
        feed_dict = {self.ml_model.valid.op.inputs: x,
                     self.ml_model.valid.labels: y}
        return feed_dict

    def run_training(self):
        self.ml_model.optimizer.run(
            n_train_steps=self.options['optim/train/max_steps'],
            feed_train=self.feed_train)

    def _init_dataset(self):
        raise NotImplementedError('_init_dataset is not defined in '
                                  'the Supervised base class')

    def _init_ml_model(self, logger_path):
        raise NotImplementedError('_init_ml_model is not defined in '
                                  'the Supervised base class')

    def __init__(self, options=None, tmp_path='tmp'):
        self._tmp_path = tmp_path
        self._options = self._init_options(options)
        self._dataset = self._init_dataset()
        self._ml_model = self._init_ml_model(tmp_path)
        super(Supervised, self).__init__()

    @classmethod
    def default_options(cls):
        dummy = cls.__new__(cls)
        return dummy._init_options(None)


class SupervisedEstimator(tdl.core.TdlModel):
    _submodels = ['model', 'train', 'valid', 'test']

    @property
    def logger_path(self):
        return self._logger_path

    @tdl.core.InputArgument
    def session(self, value):
        return (value if value is not None
                else tf.get_default_session()
                if tf.get_default_session() is not None
                else tf.InteractiveSession())

    @tdl.core.Submodel
    def model(self, value):
        if value is None:
            raise NotImplementedError('model is not defined in '
                                      'the base SupervisedEstimator class')
        return value

    @tdl.core.Submodel
    def train(self, value):
        if value is None:
            raise NotImplementedError('train is not defined in '
                                      'the base SupervisedEstimator class')
        return value

    @tdl.core.Submodel
    def valid(self, value):
        return (value if value is not None
                else self.train)

    @tdl.core.Submodel
    def test(self, value):
        return (value if value is not None
                else self.valid)

    @tdl.core.OptionalProperty
    def monitor(self):
        # monitoring
        ml_monitor = tdlm.SimpleTrainingMonitor(
            train_vars={'train/loss': self.train.loss},
            valid_vars=({'valid/loss': self.valid.loss}
                        if self.valid is not None
                        else None),
            log_folder=self.logger_path,
            tf_graph=self.session.graph)
        return ml_monitor

    @tdl.core.OptionalProperty
    def optimizer(self):
        if not self.monitor.is_set:
            self.monitor.init()
        with tf.name_scope('optimizer') as scope:
            optimizer = optim.Optimizer(
                    loss=self.train.loss,
                    var_list=tdl.core.get_trainable(self.model),
                    session=self.session,
                    metrics=(self.monitor.value),
                    n_logging=self.options['optim/n_logging'],
                    learning_rate=self.options['train/optim/learning_rate'])
        # tf.global_variables_initializer().run()
        # list(map( lambda x: x.run(),
        #           [v.initializer
        #            for v in tf.trainable_variables(scope)]))
        return optimizer

    def fit(self, dataset, max_steps=None, feed_train=None, feed_valid=None):
        def h_feed_train():
            x, y = dataset.train.next_batch(self.options['train/batch_size'])
            feed_dict = {self.train.inputs: x,
                         self.train.loss.labels: y}
            if feed_train is not None:
                if isinstance(feed_train, dict):
                    feed_dict.update(feed_train)
                else:
                    feed_dict.update(feed_train())
            return feed_dict

        def h_feed_valid():
            x, y = dataset.valid.next_batch(self.options['valid/batch_size'])
            feed_dict = {self.valid.inputs: x,
                         self.valid.loss.labels: y}
            if feed_valid is not None:
                if isinstance(feed_valid, dict):
                    feed_valid.update(feed_valid)
                else:
                    feed_valid.update(feed_valid())
            return feed_dict
        if not self.optimizer.is_set:
            self.optimizer.init()
        max_steps = (max_steps if max_steps is not None
                     else self.options['train/optim/max_steps'])
        self.optimizer.run(
            n_train_steps=max_steps,
            feed_train=h_feed_train,
            feed_valid=(h_feed_valid if dataset.valid is not None
                        else None))

    def predict(self, inputs):
        pred_op = tf.convert_to_tensor(self.test.value)
        feed_dict = {self.test.inputs: inputs}
        pred = self.session.run(pred_op, feed_dict=feed_dict)
        return pred

    def _init_options(self, options=None):
        default = {
            'optim/n_logging': 100,
            'train/batch_size': 500,
            'valid/batch_size': 500,
            'train/optim/max_steps': 1000,
            'train/optim/learning_rate': 0.02}

        options = tdl.core.check_defaults(options, default)
        options = super(SupervisedEstimator, self)._init_options(options)
        return options

    def reset_vars(self):
        ''' initialize tf variables '''
        list(map(lambda x: x.initializer.run(),
                 tdl.core.get_trainable(self.model)))

    @tdl.core.EncapsulatedMethod
    def _init_vars(self, local, value):
        ''' initialize tf variables for the first time '''
        local.not_initialized = True

    @_init_vars.eval
    def _init_vars(self, local):
        if local.not_initialized and tdl.core.is_property_set(self, 'model'):
            local.not_initialized = False
            list(map(lambda x: x.initializer.run(),
                 tdl.core.get_variables(self.model)))

    def get_save_data(self):
        init_args = {
            'model': tdl.core.save.get_save_data(self.model),
            'name': self.scope
            }
        return tdl.core.save.ModelData(cls=type(self),
                                       init_args=init_args)

    def save(self, filepath):
        data = tdl.core.save.get_save_data(self)
        with open(filepath, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as handle:
            data = pickle.load(handle)
        model = data.load()
        assert isinstance(model, cls),\
            'type of loaded model is {}, loader was expecting {} '\
            ''.format(type(model), cls)
        return model

    def __init__(self, options=None, logger_path='tmp', session=None,
                 **kargs):
        self._logger_path = logger_path
        # tdl decorators
        super(SupervisedEstimator, self).__init__(
            options=options, session=session, **kargs)
        # initialize variables
        self._init_vars()
        # tf.global_variables_initializer().run()


class LinearClassifier(SupervisedEstimator):
    def _init_options(self, options=None):
        default = {'model/regularizer/scale': None,
                   'train/loss/alpha': 1e-5,
                   'train/optim/max_steps': 2000}
        options = tdl.core.check_defaults(options, default)
        options = super(LinearClassifier, self)._init_options(options)
        return options

    @tdl.core.Submodel
    def model(self, value):
        if isinstance(value, dict):
            model = (tdl.feedforward.LinearClassifier(**value)
                     if isinstance(value, dict)
                     else value)
            model.regularizer.init(self.options['model/regularizer/scale'])
        elif isinstance(value, tdl.feedforward.LinearClassifier):
            model = value
        else:
            raise ValueError('Provided model is not valid')
        return model

    @tdl.core.Submodel
    def train(self, value):
        train = (self.model.evaluate() if value is None
                 else value)
        if not train.loss.is_set:
            train.loss.init(self.options['train/loss/alpha'])
        return train

    @tdl.core.OptionalProperty
    def monitor(self):
        # monitoring
        train_vars = {'train/loss': self.train.loss.value}
        valid_vars = (collections.OrderedDict(
                        {'valid/loss': self.valid.loss.value,
                         'valid/accuracy': self.valid.loss.empirical.accuracy})
                      if self.valid is not None
                      else None)
        ml_monitor = tdlm.SimpleTrainingMonitor(
            train_vars=train_vars,
            valid_vars=valid_vars,
            log_folder=self.logger_path,
            tf_graph=self.session.graph)
        return ml_monitor

    def __init__(self, n_inputs=None, n_classes=None,
                 options=None, logger_path='tmp', session=None,
                 **kargs):
        if 'model' not in kargs:
            kargs['model'] = {'n_inputs': n_inputs, 'n_classes': n_classes}
        super(LinearClassifier, self).__init__(
            options=options, logger_path=logger_path, session=session,
            **kargs)


class MlpClassifier(LinearClassifier):
    @tdl.core.InputArgument
    def keep_prob(self, value):
        return value

    @tdl.core.Submodel
    def model(self, value):
        if isinstance(value, dict):
            model = (tdl.feedforward.MlpClassifier(**value)
                     if isinstance(value, dict)
                     else value)
            model.regularizer.init(self.options['model/regularizer/scale'])
        elif isinstance(value, tdl.feedforward.MlpClassifier):
            model = value
        else:
            raise ValueError('Provided model is not valid')
        return model

    @tdl.core.Submodel
    def train(self, value):
        train = (self.model.evaluate(keep_prob=self.keep_prob)
                 if value is None
                 else value)
        if not train.loss.is_set:
            train.loss.init(self.options['train/loss/alpha'])
        return train

    @tdl.core.Submodel
    def valid(self, value):
        valid = (value if value is not None
                 else self.model.evaluate())
        if not valid.loss.is_set:
            valid.loss.init(self.options['train/loss/alpha'])
        return valid

    def __init__(self, n_inputs=None, n_classes=None, n_hidden=None,
                 afunction=tf.nn.relu, options=None, logger_path='tmp',
                 session=None, **kargs):
        if 'model' not in kargs:
            kargs['model'] = {'n_inputs': n_inputs, 'n_classes': n_classes,
                              'n_hidden': n_hidden, 'afunction': afunction}
        super(LinearClassifier, self).__init__(
            options=options, logger_path=logger_path, session=session,
            **kargs)


class AlexNetClassifier(LinearClassifier):
    @tdl.core.Submodel
    def model(self, value):
        model = (tdl.feedforward.AlexNetClassifier(**value)
                 if isinstance(value, dict)
                 else value)
        model.regularizer.init(self.options['model/regularizer/scale'])
        return model

    def __init__(self, input_shape, n_classes, n_filters,
                 filter_sizes, pool_sizes, n_hidden,
                 options=None, logger_path='tmp', session=None,
                 **kargs):
        super(LinearClassifier, self).__init__(
            model={'input_shape': input_shape, 'n_classes': n_classes,
                   'n_filters': n_filters, 'filter_sizes': filter_sizes,
                   'pool_sizes': pool_sizes, 'n_hidden': n_hidden},
            options=options, logger_path=logger_path, session=session,
            **kargs)
