import unittest
import numpy as np
import pandas as pd
import numpy.testing
import twodlearn as tdl
import twodlearn.datasets.base
import twodlearn.datasets.cyber as cyber
import twodlearn.datasets.unbalanced


class UnbalancedTests(unittest.TestCase):
    def test_df_unbalanced(self):
        data = {i: pd.DataFrame(
            np.random.normal(loc=np.random.normal(scale=10.0),
                             scale=np.random.normal(scale=10.0)**2,
                             size=[1000, 10]))
                for i in range(4)}
        dataset = tdl.datasets.unbalanced.DfUnbalancedDataset(data)
        train, valid, test = dataset.split([0.5, 0.3])
        dataset = tdl.datasets.base.Datasets(train, valid, test)
        dataset.normalizer = \
            tdl.datasets.base.Normalizer(dataset.train.get_stats())

        mean = np.abs(dataset.train.next_batch(1000)[0].mean())
        assert (mean < 0.05),\
            'mean after normalization is outside expected range '\
            '(got {})'.format(mean)

        stddev = np.abs(dataset.train.next_batch(1000)[0].std())
        assert (stddev < 2.0),\
            'stddev after normalization is outside expected range '\
            '(got {})'.format(stddev)

    def test_unbalanced(self):
        data = {i: np.random.normal(loc=np.random.normal(scale=10.0),
                                    scale=np.random.normal(scale=10.0)**2,
                                    size=[1000, 10])
                for i in range(4)}
        dataset = tdl.datasets.unbalanced.UnbalancedDataset(data)
        train, valid, test = dataset.split([0.5, 0.3])
        dataset = tdl.datasets.base.Datasets(train, valid, test)
        dataset.normalizer = \
            tdl.datasets.base.Normalizer(dataset.train.get_stats())

        mean = np.abs(dataset.train.next_batch(1000)[0].mean())
        assert (mean < 0.05),\
            'mean after normalization is outside expected range '\
            '(got {})'.format(mean)

        stddev = np.abs(dataset.train.next_batch(1000)[0].std())
        assert (stddev < 2.0),\
            'stddev after normalization is outside expected range '\
            '(got {})'.format(stddev)

    def test_sample(self):
        data = {i: np.random.normal(loc=np.random.normal(scale=10.0),
                                    scale=np.random.normal(scale=0.5)**2,
                                    size=[2000, 10])
                for i in range(4)}
        dataset = tdl.datasets.unbalanced.UnbalancedDataset(data)
        dataset.normalizer = \
            tdl.datasets.base.Normalizer(dataset.get_stats())

        stats = dataset.get_stats()
        batch_x, batch_y = dataset.sample(7900, replace=False)
        np.testing.assert_almost_equal(stats.mean, batch_x.mean(axis=0), 2)


if __name__ == "__main__":
    unittest.main()
