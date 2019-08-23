import numpy as np
import twodlearn as tdl
import tensorflow as tf
from twodlearn.bayesnet.gaussian_process import \
    GaussianProcess, GpWithExplicitMean, VariationalGP, ExplicitVGP
from twodlearn.bayesnet import GaussianNegLogLikelihood


@tdl.core.add_autoinit_class
class GpEstimator(tdl.templates.supervised.SupervisedEstimator):
    _submodels = []

    @tdl.core.SubmodelInit
    def model(self, train_x, train_y):
        # if isinstance(train_x, np.nparray):
        #     train_x = tf.Variable(train_x, trainable=False)
        # if isinstance(train_y, np.nparray):
        #     train_y = tf.Variable(train_y, trainable=False)
        train_x = (train_x if self.normalizer is None
                   else self.normalizer(train_x))
        model = GaussianProcess(xm=train_x, ym=np.transpose(train_y),
                                y_scale=0.1)
        return model

    def fit(self, train_x, train_y, max_iter=100):
        """fit the gaussian process to the training data.
        By default, the parameters of the kernel are optimized.

        Args:
            train_x: input training data [batch_size, x_size].
            train_y: output training data [batch_size, y_size].
            max_iter: number of maximum iterations to run the optimization.
                Defaults to 100.
        """
        if not tdl.core.is_property_initialized(self, 'model'):
            self.model.init(train_x=train_x, train_y=train_y)
        if not tdl.core.is_property_initialized(self, 'optimizer'):
            self.optimizer.init()
            self._init_vars()
        self.optimizer.run(max_iter)

    @tdl.core.Submodel
    def normalizer(self, value):
        return value

    def predict(self, x):
        feed_dict = {tdl.core.get_placeholder(self.test.inputs): x}
        mean, stddev = self.session.run([self.test.mean, self.test.stddev],
                                        feed_dict=feed_dict)
        return tdl.core.SimpleNamespace(mean=mean.transpose(),
                                        stddev=stddev.transpose())

    @tdl.core.Submodel
    def train(self, _):
        train = self.model.marginal_likelihood()
        return tdl.core.SimpleNamespace(
            loss=tf.reduce_sum(train.loss),
            posterior=train)

    @tdl.core.Submodel
    def valid(self, _):
        return self.train

    @tdl.core.Submodel
    def test(self, _):
        x_train = self.model.xm
        x_train = (x_train if isinstance(x_train, (np.ndarray, tf.Tensor))
                   else tf.convert_to_tensor(x_train))
        input_dim = (x_train.shape[1] if isinstance(x_train.shape[1], int)
                     else x_train.shape[1].value)
        test_x = tf.placeholder(dtype=tdl.core.global_options.float.tftype,
                                shape=[None, input_dim])
        test_x = (test_x if self.normalizer is None
                  else self.normalizer(test_x))
        return self.model.predict(test_x)

    def __init__(self, options=None, logger_path='tmp', session=None,
                 **kargs):
        super(GpEstimator, self).__init__(
            logger_path=logger_path,
            options=options, session=session, **kargs)

    def get_save_data(self):
        init_args = {
            'model': tdl.core.save.get_save_data(self.model),
            'normalizer': tdl.core.save.get_save_data(self.normalizer),
            'name': self.scope
            }
        return tdl.core.save.ModelData(cls=type(self),
                                       init_args=init_args)

    @tdl.core.SubmodelInit(lazzy=True)
    def optimizer(self, learning_rate=0.01, valid_freq=10, **kargs):
        if not self.monitor.is_set:
            self.monitor.init()
        with tf.name_scope('optimizer'):
            optimizer = tdl.optimv2.Optimizer(
                    loss=self.train.loss,
                    var_list=tdl.core.get_trainable(self.model),
                    session=self.session,
                    monitor_manager={'metrics': self.monitor.value,
                                     'valid_freq': valid_freq},
                    learning_rate=learning_rate,
                    **kargs)
        tdl.core.initialize_variables(self.model)
        return optimizer


@tdl.core.add_autoinit_class
class ExplicitGpEstimator(GpEstimator):
    '''Explicit gp estimator'''
    @tdl.core.SubmodelInit
    def model(self, train_x, train_y):
        train_x = (train_x if self.normalizer is None
                   else self.normalizer(train_x))
        gp = GaussianProcess(train_x, train_y)
        model = GpWithExplicitMean(gp_model=gp)
        return model

    def fit(self, train_x, train_y, max_iter=100):
        if not tdl.core.is_property_initialized(self, 'model'):
            self.model.init(train_x=train_x,
                            train_y=np.transpose(train_y))
            tdl.core.assert_initialized(self, 'test')
        if not tdl.core.is_property_initialized(self, 'optimizer'):
            self.optimizer.init()
            self._init_vars()
        self.optimizer.run(max_iter)

    def __init__(self, options=None,
                 logger_path='tmp', session=None, **kargs):
        self._logger_path = logger_path
        super(GpEstimator, self).__init__(options=options, session=session,
                                          **kargs)

    @tdl.core.Submodel
    def test(self, _):
        input_dim = tf.convert_to_tensor(self.model.gp_model.xm).shape[1].value
        test_x = tf.placeholder(dtype=tdl.core.global_options.float.tftype,
                                shape=[None, input_dim])
        test_x = (test_x if self.normalizer is None
                  else self.normalizer(test_x))
        return self.model.predict(test_x)


@tdl.core.add_autoinit_class
class VGPEstimator(tdl.templates.supervised.SupervisedEstimator):
    _submodels = []

    @tdl.core.InputArgument
    def m(self, value):
        return value

    @tdl.core.SubmodelInit
    def model(self, dataset):
        xdims = dataset.train.x.shape[-1]
        ydims = dataset.train.y.shape[-1]
        model = VariationalGP(m=self.m, input_shape=[None, xdims],
                              y_scale=np.array([1.0]*ydims))
        return model

    def fit(self, dataset, max_iter=100, feed_dict=None):
        if not tdl.core.is_property_set(self, 'model'):
            self.model.init(dataset=dataset)
        if not tdl.core.is_property_set(self, 'train'):
            self.train.init(dataset=dataset)
        if not tdl.core.is_property_set(self, 'test'):
            self.test.init(dataset=dataset)
        if not tdl.core.is_property_initialized(self, 'optimizer'):
            self.optimizer.init()
            self._init_vars()
        batch_size = tf.convert_to_tensor(self.train.inputs)\
                       .shape[0].value

        def feed_train():
            batch_x, batch_y = dataset.train.next_batch(batch_size)
            feed = {tdl.core.get_placeholder(self.train.labels): batch_y,
                    tdl.core.get_placeholder(self.train.inputs): batch_x}
            if feed_dict is not None:
                feed.update(feed_dict)
            return feed

        self.optimizer.run(max_iter, feed_train=feed_train)

    @tdl.core.Submodel
    def normalizer(self, value):
        return value

    def predict(self, x):
        feed_dict = {tdl.core.get_placeholder(self.test.inputs): x}
        mean, stddev = self.session.run([self.test.mean, self.test.stddev],
                                        feed_dict=feed_dict)
        return tdl.core.SimpleNamespace(mean=mean.transpose(),
                                        stddev=stddev.transpose())

    @tdl.core.SubmodelInit(lazzy=True)
    def train(self, dataset, batch_size=50):
        n_inputs = dataset.train.x.shape[-1]
        n_outputs = dataset.train.y.shape[-1]
        x_train = tf.placeholder(
            shape=[batch_size, n_inputs],
            dtype=tdl.core.global_options.float.tftype)
        y_train = tf.placeholder(
            shape=[batch_size, n_outputs],
            dtype=tdl.core.global_options.float.tftype)
        if self.normalizer is not None:
            x_train = self.normalizer(x_train)
        train = self.model.neg_elbo(
            labels=tdl.feedforward.Transpose(inputs=y_train),
            inputs=x_train,
            dataset_size=dataset.train.n_samples)
        return train

    @tdl.core.Submodel
    def valid(self, _):
        return None

    @tdl.core.SubmodelInit
    def test(self, dataset, tolerance=1e-4):
        x_shape = dataset.train.x.shape[-1]
        test_x = tf.placeholder(dtype=tdl.core.global_options.float.tftype,
                                shape=[None, x_shape])
        test_x = (test_x if self.normalizer is None
                  else self.normalizer(test_x))
        return self.model.predict(test_x, tolerance)

    @tdl.core.SubmodelInit(lazzy=True)
    def optimizer(self, learning_rate=0.01, valid_freq=10, **kargs):
        if not self.monitor.is_set:
            self.monitor.init()
        with tf.name_scope('optimizer'):
            optimizer = tdl.optimv2.Optimizer(
                    loss=self.train.loss,
                    var_list=tdl.core.get_trainable(self.model),
                    session=self.session,
                    monitor_manager={'metrics': self.monitor.value,
                                     'valid_freq': valid_freq},
                    learning_rate=learning_rate,
                    **kargs)
        tdl.core.initialize_variables(self.model)
        return optimizer

    def __init__(self, options=None, logger_path='tmp', session=None,
                 **kargs):
        super(VGPEstimator, self).__init__(
            logger_path=logger_path,
            options=options, session=session, **kargs)


class EVGPEstimator(VGPEstimator):
    @tdl.core.SubmodelInit(lazzy=True)
    def model(self, dataset, prior=1.0):
        xdims = dataset.train.x.shape[-1]
        model = ExplicitVGP(m=self.m, input_shape=[None, xdims],
                            prior=prior)
        return model

    @tdl.core.SubmodelInit(lazzy=True)
    def train(self, dataset, batch_size=50):
        x_shape = dataset.train.x.shape[-1]
        x_train = tf.placeholder(
            shape=[batch_size, x_shape],
            dtype=tdl.core.global_options.float.tftype)
        y_train = tf.placeholder(
            shape=[batch_size, 1],
            dtype=tdl.core.global_options.float.tftype)
        if self.normalizer is not None:
            x_train = self.normalizer(x_train)
        train = self.model.predict(inputs=x_train)
        return tdl.templates.supervised.SupervisedWrapper(
            model=train, inputs=tdl.core.get_placeholder(train.inputs),
            loss=train.neg_elbo(
                labels=tf.linalg.transpose(y_train),
                dataset_size=dataset.train.n_samples),
            labels=y_train)
