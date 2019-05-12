import os
import unittest
import numpy as np
import pandas as pd
import twodlearn.debug
import tensorflow as tf
from twodlearn import optim
from twodlearn.datasets import tsdataset

TESTS_PATH = os.path.dirname(os.path.abspath(__file__))


class TSDatasetTest(unittest.TestCase):
    def setUp(self):
        n_columns = 5
        n_samples = [100, 50, 30]
        column_names = ['x{}'.format(i) for i in range(n_columns)]
        records = [tsdataset.Record(pd.DataFrame(np.random.rand(i, n_columns),
                                                 columns=column_names))
                   for i in n_samples]
        self.dataset = tsdataset.TSDatasets(train=records)

    def test_single_batch_continuous(self):
        batch_window = 10
        for i in range(3):
            batch1 = self.dataset.train.next_batch(batch_window, 1)
            batch1 = batch1['all'].squeeze()
            batch2 = self.dataset.train\
                         .records[0].data\
                         .iloc[i * batch_window:(i + 1) * batch_window, :]\
                         .values

            np.testing.assert_array_equal(batch1, batch2)

    def test_multiple_batch_continuous(self):
        batch_window = 10
        n_batch = 5
        for i in range(3):
            batch1 = self.dataset.train.next_batch(batch_window, 5)
            batch1 = batch1['all'][:, 0, :].squeeze()
            batch2 = self.dataset\
                         .train.records[0].data\
                         .iloc[i * batch_window:(i + 1) * batch_window, :]\
                         .values

            np.testing.assert_array_equal(batch1, batch2)

    def test_windowed_batch(self):
        n_columns = 2
        n_samples = [100, 50, 30]
        window_size = 3
        sequences_length = 20
        column_names = ['x{}'.format(i) for i in range(n_columns)] + ['y']

        records = [tsdataset.Record(
            pd.DataFrame(np.concatenate([np.random.rand(i, n_columns),
                                         np.expand_dims(np.arange(i), axis=1)],
                                        axis=1),
                         columns=column_names))
                   for i in n_samples]
        dataset = tsdataset.TSDatasets(train=records)
        dataset.set_groups({'x': 'x.', 'y': 'y'})

        batch = dataset.train.next_windowed_batch(
            sequences_length, 10, window_size, groups=['x'])
        batch_y = dataset.train.records[0].data['y'].iloc[(
            window_size - 1):sequences_length + window_size - 1].values
        batch_y = np.expand_dims(batch_y, axis=1)

        batch_x = dataset.train.records[0].data[[
            'x0', 'x1']].iloc[:window_size, :].values
        batch_x = np.reshape(batch_x, (n_columns * window_size))
        np.testing.assert_array_equal(batch['y'][:, 0, :],
                                      batch_y)

        np.testing.assert_array_equal(batch['x'][0, 0, :],
                                      batch_x)

    def test_single_batch_continuous_parallel(self):
        batch_window = 10

        def feed_train():
            batch1 = self.dataset.train.next_batch(batch_window, 1)
            batch1 = batch1['all'].squeeze()
            return batch1

        data_feeder = optim.DataFeeder(feed_train)

        for i in range(10):
            batch1 = data_feeder.feed_train()
            batch2 = self.dataset\
                         .train.records[0].data\
                         .iloc[i * batch_window:(i + 1) * batch_window, :]\
                         .values

            np.testing.assert_array_equal(batch1, batch2)

        data_feeder.stop()

    def test_saver(self):
        filename = os.path.join(TESTS_PATH, 'test_dataset')
        self.dataset.save(filename)
        dataset = tsdataset.TSDatasets.from_saved_file(filename)
        for i, record in enumerate(dataset.train.records):
            np.testing.assert_array_almost_equal(
                record.data,
                self.dataset.train.records[i].data)

    def test_saver2(self):
        data_filename = os.path.join(TESTS_PATH, 'data/cartpole_dataset.pkl')
        tmp_filename = os.path.join(TESTS_PATH, 'test_dataset')

        def assert_tsdatasets_equal(dataset1, dataset2):
            for record1, record2 in zip(dataset1.records, dataset2.records):
                np.testing.assert_array_almost_equal(record1.data,
                                                     record2.data)
        dataset1 = tsdataset.TSDatasets\
                            .from_saved_file(data_filename, encoding='latin1')
        dataset1.save(tmp_filename)
        dataset2 = tsdataset.TSDatasets\
                            .from_saved_file(tmp_filename)

        assert_tsdatasets_equal(dataset1.train, dataset2.train)

    def test_dense(self):
        features = 3
        nsamples = 50
        records = [tsdataset.Record(
            pd.DataFrame(np.random.normal(size=[13, features])))
                   for i in range(nsamples)]
        records += [tsdataset.Record(
            pd.DataFrame(np.random.normal(size=[33, features])))
                    for i in range(nsamples)]

        dataset = tsdataset.TSDataset(records=records)
        data, length = dataset.to_dense()
        np.testing.assert_array_equal(data[0, :13, :],
                                      records[0].data.to_numpy())

    def test_sample_batch_window(self):
        features = 3
        nsamples = 50
        records = [tsdataset.Record(
            pd.DataFrame(np.random.normal(size=[13, features])))
                   for i in range(nsamples)]
        records += [tsdataset.Record(
            pd.DataFrame(np.random.normal(size=[33, features])))
                    for i in range(nsamples)]

        dataset = tsdataset.TSDataset(records=records)
        tfdata = dataset.to_tf_dataset().repeat()\
                        .batch(32, drop_remainder=True)
        iterator = tfdata.make_one_shot_iterator()
        sample = iterator.get_next()
        window_size = 11
        windows, index = tsdataset.sample_batch_window(
            sample['data'], sample['length'], window_size)

        with tf.Session() as sess:
            data, idx = sess.run([windows, index])
            start_index = idx[0][1]
            expected = dataset.records[0].data\
                              .iloc[start_index:start_index+window_size]
            np.testing.assert_almost_equal(expected, data[0, ...])

            for i in range(10):
                data, idx = sess.run([windows, index])
                if np.isnan(data).any():
                    raise ValueError('got nan when sampling window')
                if (data == 0.0).any():
                    raise ValueError('got 0.0 when sampling window')

    def test_sample_batch_window2(self):
        features = 3
        nsamples = 50
        records = [tsdataset.Record(
            pd.DataFrame(np.random.normal(size=[13, features])))
                   for i in range(nsamples)]
        records += [tsdataset.Record(
            pd.DataFrame(np.random.normal(size=[33, features])))
                    for i in range(nsamples)]

        dataset = tsdataset.TSDataset(records=records)
        tfdata = dataset.to_tf_dataset().repeat()\
                        .batch(32)
        iterator = tfdata.make_one_shot_iterator()
        sample = iterator.get_next()
        window_size = 11
        windows, index = tsdataset.sample_batch_window(
            sample['data'], sample['length'], window_size)

        with tf.Session() as sess:
            data, idx = sess.run([windows, index])
            start_index = idx[0][1]
            expected = dataset.records[0].data\
                              .iloc[start_index:start_index+window_size]
            np.testing.assert_almost_equal(expected, data[0, ...])

            for i in range(50):
                data, idx = sess.run([windows, index])
                if np.isnan(data).any():
                    raise ValueError('got nan when sampling window')
                if (data == 0.0).any():
                    raise ValueError('got 0.0 when sampling window')

    def test_sample_batch_window3(self):
        features = 6
        nsamples = 50
        records = [tsdataset.Record(
            pd.DataFrame(np.random.normal(size=[13, features]),
                         columns=['q{}'.format(i) for i in range(features)]))
                   for i in range(nsamples)]
        records += [tsdataset.Record(
            pd.DataFrame(np.random.normal(size=[33, features]),
                         columns=['q{}'.format(i) for i in range(features)]))
                    for i in range(nsamples)]

        dataset = tsdataset.TSDataset(records=records)
        dataset.set_groups(
            {'pos':
             ['q{}'.format(i) for i in range(features//2)],
             'vel':
             ['q{}'.format(i) for i in range(features//2, features)]})
        tfdata = dataset.to_tf_dataset().repeat()\
                        .batch(32)
        iterator = tfdata.make_one_shot_iterator()
        sample = iterator.get_next()
        window_size = 11
        windows, index = tsdataset.sample_batch_window(
            sample['data'], sample['length'], window_size)

        with tf.Session() as sess:
            data, idx = sess.run([windows, index])
            start_index = idx[0][1]
            expected = dataset.records[0].data[dataset.group_tags['pos']]\
                              .iloc[start_index:start_index+window_size]
            np.testing.assert_almost_equal(expected, data['pos'][0, ...])
            expected = dataset.records[0].data[dataset.group_tags['vel']]\
                              .iloc[start_index:start_index+window_size]
            np.testing.assert_almost_equal(expected, data['vel'][0, ...])

            for i in range(50):
                data_dict, idx = sess.run([windows, index])
                for data in data_dict.values():
                    if np.isnan(data).any():
                        raise ValueError('got nan when sampling window')
                    if (data == 0.0).any():
                        raise ValueError('got 0.0 when sampling window')

    def test_sample_window(self):
        features = 6
        nsamples = 50
        records = [tsdataset.Record(
            pd.DataFrame(np.random.normal(size=[23, features]),
                         columns=['q{}'.format(i) for i in range(features)]))
                   for i in range(nsamples)]
        records += [tsdataset.Record(
            pd.DataFrame(np.random.normal(size=[33, features]),
                         columns=['q{}'.format(i) for i in range(features)]))
                    for i in range(nsamples)]

        dataset = tsdataset.TSDataset(records=records)
        dataset.set_groups(
            {'pos':
             ['q{}'.format(i) for i in range(features//2)],
             'vel':
             ['q{}'.format(i) for i in range(features//2, features)]})
        tfdata = dataset.to_tf_dataset().repeat()
        iterator = tfdata.make_one_shot_iterator()
        sample = iterator.get_next()
        window_size = 11
        windows, t0 = tsdataset.sample_window(
            sample['data'], sample['length'], window_size)

        with tf.Session() as sess:
            data, idx = sess.run([windows, t0])
            start_index = idx
            expected = dataset.records[0].data[dataset.group_tags['pos']]\
                              .iloc[start_index:start_index+window_size]
            np.testing.assert_almost_equal(expected, data['pos'])
            expected = dataset.records[0].data[dataset.group_tags['vel']]\
                              .iloc[start_index:start_index+window_size]
            np.testing.assert_almost_equal(expected, data['vel'])

            for i in range(1000):
                data_dict, idx = sess.run([windows, t0])
                for data in data_dict.values():
                    if np.isnan(data).any():
                        raise ValueError('got nan when sampling window')
                    if (data == 0.0).any():
                        raise ValueError('got 0.0 when sampling window')


if __name__ == "__main__":
    unittest.main()
