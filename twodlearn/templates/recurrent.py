import numpy as np
import tensorflow as tf
import twodlearn as tdl
import matplotlib.pyplot as plt
import twodlearn.recurrent


def next_batch_for_narx(dataset, window_size, n_steps, batch_size):
    """Returns a batch in a narx format: x0, y, inputs.
    Args:
        dataset (TSDataset): time series dataset.
        window_size (int): number of steps to include in x0.
        n_steps (int): number of steps to include in y and u.
        batch_size (int): number of samples in the batch.
    Returns:
        (x0, y, u): batch.
    """
    def format_batch(batch, window_size, n_steps):
        _y = [batch['y'][i, :, :]
              for i in range(batch['y'].shape[0])]
        x0 = _y[:window_size]
        y = _y[window_size:]
        # dy = [_y[i] - _y[i - 1] for i in range(window_size, len(_y))]
        u = [batch['inputs'][i, :, :]
             for i in range(window_size - 1, batch['inputs'].shape[0] - 1)]

        assert (len(x0) == window_size and
                len(y) == n_steps and
                len(u) == n_steps), \
            'len of x0, y or u do not coincide with provided '\
            'window_size and n_steps'
        return x0, y, u
    batch = dataset.next_batch(window_size=window_size + n_steps,
                               batch_size=batch_size)
    x0, y, u = format_batch(batch, window_size, n_steps)
    return x0, y, u


class Mlp2LstmEstimator(tdl.templates.supervised.SupervisedEstimator):
    _submodels = []

    @tdl.core.InputModel
    def model(self, value):
        if not isinstance(value, tdl.recurrent.Mlp2Lstm):
            raise ValueError('model type must be Mlp2Lstm')
        return value

    @tdl.core.SubmodelInit
    def normalizer(self, dataset):
        stats = dataset.train.get_stats(['y', 'inputs'])
        normalizer = tdl.core.SimpleNamespace(
            y=tdl.normalizer.Normalizer(loc=stats.mean['y'],
                                        scale=stats.stddev['y']),
            inputs=tdl.normalizer.Normalizer(loc=stats.mean['inputs'],
                                             scale=stats.stddev['inputs'])
        )
        return normalizer

    def build_graph(self, init_x0, init_inputs, n_unrollings, batch_size):
        tdl.core.assert_initialized(self, 'build_graph', ['model'])

        def h_init_x0(dtype, shape):
            out = init_x0(dtype=dtype, shape=shape)
            if self.normalizer is not None:
                out = self.normalizer.y.normalize(out)
            return out

        def h_init_u(dtype, shape):
            out = init_inputs(dtype=dtype, shape=shape)
            if self.normalizer is not None:
                out = self.normalizer.inputs.normalize(out)
            return out

        test = self.model.evaluate()
        test.x0.init(Type=h_init_x0, batch_size=batch_size)
        test.inputs.init(Type=h_init_u, n_unrollings=n_unrollings,
                         batch_size=batch_size)
        return test

    @tdl.core.SubmodelInit
    def train(self, n_unrollings, regularizer_scale):
        tdl.core.assert_initialized(self, 'train', ['model'])
        if not tdl.core.is_property_set(self, 'normalizer'):
            self.normalizer = None

        def init_x0(dtype, shape):
            out = tf.placeholder(dtype=dtype, shape=shape)
            if self.normalizer is not None:
                out = self.normalizer.y.normalize(out)
            return out

        def init_u(dtype, shape):
            out = tf.placeholder(dtype=dtype, shape=shape)
            if self.normalizer is not None:
                out = self.normalizer.inputs.normalize(out)
            return out

        train = self.model.evaluate()
        train.x0.init(Type=init_x0)
        train.inputs.init(Type=init_u, n_unrollings=n_unrollings)
        outputs = train.lstm.y
        losses = [tdl.losses.L2Loss(yi) for yi in outputs]
        labels = [l.labels for l in losses]
        regularizer = tdl.losses.L2Regularizer(
            weights=self.model.weights,
            scale=regularizer_scale
        )
        fit_loss = tdl.losses.AddNLosses(losses)
        _loss = fit_loss + regularizer
        train.loss = tdl.core.SimpleNamespace(
            value=tf.convert_to_tensor(_loss),
            losses=losses,
            fit_loss=tf.convert_to_tensor(fit_loss)/np.float32(len(losses)),
            labels=labels,
            regularizer=regularizer
        )
        return train

    @tdl.core.OptionalProperty
    def monitor(self):
        train_fit_loss = tf.convert_to_tensor(self.train.loss.fit_loss)
        valid_fit_loss = tf.convert_to_tensor(self.valid.loss.fit_loss)
        ml_monitor = tdl.monitoring.SimpleTrainingMonitor(
            train_vars={'train/loss': self.train.loss,
                        'train/fit_loss': train_fit_loss},
            valid_vars=({'valid/loss': self.valid.loss,
                         'valid/fit_loss': valid_fit_loss}
                        if self.valid is not None
                        else None),
            log_folder=self.logger_path,
            tf_graph=self.session.graph)
        return ml_monitor

    @staticmethod
    def _feed_dict(dataset, model, batch_size):
        x0, y, u = next_batch_for_narx(
            dataset=dataset, window_size=model.window_size,
            n_steps=model.n_unrollings,
            batch_size=batch_size)

        feed_dict = dict()
        feed_dict.update({tdl.core.get_placeholder(model.inputs[t]): u[t]
                          for t in range(model.n_unrollings)})
        feed_dict.update({tdl.core.get_placeholder(model.loss.labels[t]): y[t]
                          for t in range(model.n_unrollings)})
        feed_dict.update({tdl.core.get_placeholder(model.x0[t]): x0[t]
                          for t in range(model.window_size)})
        return feed_dict

    def fit(self, dataset, max_iter=100, batch_size=100,
            feed_train=None, feed_valid=None):
        tdl.core.assert_initialized(self, 'fit', ['model', 'train'])

        def h_feed_train():
            feed_dict = self._feed_dict(dataset.train, self.train, batch_size)
            if feed_train is not None:
                if isinstance(feed_train, dict):
                    feed_dict.update(feed_train)
                else:
                    feed_dict.update(feed_train())
            return feed_dict

        def h_feed_valid():
            feed_dict = self._feed_dict(dataset.valid, self.valid, batch_size)
            if feed_valid is not None:
                if isinstance(feed_valid, dict):
                    feed_valid.update(feed_valid)
                else:
                    feed_valid.update(feed_valid())
            return feed_dict

        if not tdl.core.is_property_set(self, 'optimizer'):
            self.optimizer.init()
            # init vars
            list(map(lambda x: x.initializer.run(),
                 tdl.core.get_variables(self.train)))
        max_iter = (max_iter if max_iter is not None
                    else self.options['train/optim/max_steps'])
        self.optimizer.run(
            n_train_steps=max_iter,
            feed_train=h_feed_train,
            feed_valid=(h_feed_valid if dataset.valid is not None
                        else None))

    @tdl.core.SubmodelInit
    def test(self, n_unrollings):
        tdl.core.assert_initialized(self, 'test', ['model'])

        def init_x0(dtype, shape):
            out = tf.placeholder(dtype=dtype, shape=shape)
            if self.normalizer is not None:
                out = self.normalizer.y.normalize(out)
            return out

        def init_u(dtype, shape):
            out = tf.placeholder(dtype=dtype, shape=shape)
            if self.normalizer is not None:
                out = self.normalizer.inputs.normalize(out)
            return out

        test = self.model.evaluate()
        test.x0.init(Type=init_x0)
        test.inputs.init(Type=init_u, n_unrollings=n_unrollings)
        return test

    def predict(self, x0, inputs):
        tdl.core.assert_initialized(self, 'predict', ['test'])
        feed_dict = dict()
        feed_dict.update({
            tdl.core.get_placeholder(self.test.inputs[t]): inputs[t]
            for t in range(len(inputs))})
        feed_dict.update({
            tdl.core.get_placeholder(self.test.x0[t]): x0[t]
            for t in range(self.test.window_size)})
        y = self.session.run(
            [tf.convert_to_tensor(yt)
             for yt in self.test.lstm.y[:len(inputs)]],
            feed_dict=feed_dict)
        return y

    def visualize_fit(self, dataset, n_steps=None, ax=None):
        if n_steps is None:
            n_steps = self.test.n_unrollings
        x0, y, u = next_batch_for_narx(
            dataset=dataset, window_size=self.test.window_size,
            n_steps=n_steps,
            batch_size=1)
        yp = self.predict(x0=x0, inputs=u)
        y = np.concatenate(y, 0)
        yp = np.concatenate(yp, 0)
        if ax is None:
            fig, ax = plt.subplots(yp.shape[1], 1)
        for i in range(yp.shape[1]):
            ax[i].plot(yp[:, i])
            ax[i].plot(y[:, i], 'k+')
        return yp

    def visualize_predictions(self, x0, inputs, targets, ax=None):
        yp = self.predict(x0=x0, inputs=inputs)
        yp = np.concatenate(yp, 0)
        y = (np.concatenate(targets, 0) if isinstance(targets, list)
             else targets)
        if ax is None:
            fig, ax = plt.subplots(yp.shape[1], 1)
        for i in range(yp.shape[1]):
            ax[i].plot(yp[:, i])
            ax[i].plot(y[:, i], 'k+')
        return yp
