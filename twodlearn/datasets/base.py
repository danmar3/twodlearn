""" Creates a generic dataset from a given data """

import os
import math
import gzip
import tempfile
import numpy as np
import pandas as pd
import twodlearn as tdl
from shutil import copyfile

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


def maybe_download(filename, work_directory, source_url):
    """Download the data from source url, unless it's already here.
    Args:
        filename: string, name of the file in the directory.
        work_directory: string, path to working directory.
        source_url: url to download from if file doesn't exist.
    Returns:
        Path to resulting file.
    """
    if not os.path.exists(work_directory):
        os.makedirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        with tempfile.NamedTemporaryFile() as tmpfile:
            temp_file_name = tmpfile.name
            urlretrieve(source_url, temp_file_name)
            copyfile(temp_file_name, filepath)
            size = os.path.getsize(filepath)
            print('Successfully downloaded', filename, size, 'bytes.')
    else:
        print('File already downloaded.')
    return filepath


def dense_to_one_hot(labels, num_classes=None):
    """Convert class labels from scalars to one-hot vectors."""
    if isinstance(labels, int):
        labels_one_hot = np.zeros((1, num_classes))
        labels_one_hot[0, labels] = 1.0
        return labels_one_hot
    if num_classes is None:
        num_classes = len(np.unique(labels))
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot


def load_gz_as_nparray(work_directory, filename, dtype=np.float32):
    filepath = os.path.join(work_directory, filename)
    with gzip.open(filepath) as file:
        return np.frombuffer(file.read(), dtype=dtype)


class Dataset(object):
    '''
    Attributes:
        x: samples, [sample_id, sample ...]
        y: labels for each sample
        epochs_completed:
        index_in_epoch:
        n_samples:
    '''

    def __init__(self, data_x, data_y=None):
        if data_x is not None:
            self._n_samples = data_x.shape[0]
        else:
            self._n_samples = 0

        self._x = data_x
        self._y = data_y

        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    class Iloc(object):
        @property
        def dataset(self):
            return self._dataset

        def __init__(self, dataset):
            self._dataset = dataset

        def __getitem__(self, key):
            batch_x = (self.dataset.x.iloc[key]
                       if isinstance(self.dataset.x, pd.DataFrame)
                       else self.dataset.x[key])
            batch_y = (self.dataset.y.iloc[key]
                       if isinstance(self.dataset.y, pd.DataFrame)
                       else self.dataset.y[key]
                       if self.dataset.y is not None
                       else None)
            return batch_x, batch_y

    @tdl.core.LazzyProperty
    def iloc(self):
        return Dataset.Iloc(dataset=self)

    def shuffle(self):
        perm = np.arange(self.n_samples)
        np.random.shuffle(perm)
        self._x, self._y = self.iloc[perm]

    @tdl.core.EncapsulatedMethod
    def sample(self, local, value):
        local.samples_in_epoch = 0

    @sample.eval
    def sample(self, local, n_samples, replace=False):
        items = np.random.choice(self.n_samples, n_samples, replace=replace)
        batch_x, batch_y = self.iloc[items]

        local.samples_in_epoch += n_samples
        if local.samples_in_epoch >= self.n_samples:
            self._epochs_completed += local.samples_in_epoch // self.n_samples
            local.samples_in_epoch = local.samples_in_epoch % self.n_samples
        return batch_x, batch_y

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._n_samples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if (shuffle is True):
                self.shuffle()
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._n_samples

        end = self._index_in_epoch
        batch_x, batch_y = self.iloc[start:end]

        if self.normalizer.is_set:
            batch_x = self.normalizer.normalize(batch_x)
        return batch_x, batch_y

    def next_batch_x(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        batch_x, _ = self.next_batch(batch_size=batch_size, shuffle=shuffle)
        return batch_x

    def get_stats(self):
        stats = tdl.common.SimpleNamespace(
            mean=self.x.mean(axis=0),
            stddev=self.x.std(axis=0),
            min=self.x.min(axis=0),
            max=self.x.max(axis=0))
        return stats

    @tdl.core.OptionalProperty
    def normalizer(self, value):
        if value is None:
            stats = self.get_stats()
            value = Normalizer(stats=stats)
        return value

    def update_data(self, data_x, data_y=None):
        self._n_samples = data_x.shape[0]
        self._x = data_x
        self._y = data_y
        self._index_in_epoch = 0


class DfDataset(Dataset):
    pass


class Datasets(object):
    def __init__(self, train, valid=None, test=None):
        self.train = train
        self.valid = valid
        self.test = test

    def split_train(self, p_train, p_valid, p_test=0.0, shuffle=True):

        assert (p_train + p_valid + p_test == 1.0), 'percentajes must sum one'
        assert (p_train < 0.0 or p_valid < 0.0 or p_test <
                0.0), 'percentajes must be positive'

        # shuffle the dataset
        if (shuffle is True):
            self.train.shuffle()

        # get number of samples for each dataset
        n_samples = self.train.n_samples()

        n_test = int(math.floor(n_samples * p_test))
        n_valid = int(math.floor(n_samples * p_valid))
        n_train = int(n_samples) - n_test - n_valid

        # update datasets
        data_x = self.train.x
        data_y = self.train.y
        if data_y is not None:
            self.test.update_data(data_x[0:n_test], data_y[0:n_test])
            self.valid.update_data(
                data_x[n_test:n_test + n_valid], data_y[n_test:n_test + n_valid])
            self.train.update_data(
                data_x[n_test + n_valid:], data_y[n_test + n_valid:])

        else:
            self.test.update_data(data_x[0:n_test], None)
            self.valid.update_data(data_x[n_test:n_test + n_valid], None)
            self.train.update_data(data_x[n_test + n_valid:], None)

    @tdl.core.OptionalProperty
    def normalizer(self, value=None):
        if value is None and self.train.normalizer.is_set:
            value = self.train.normalizer.value
        elif value is None:
            stats = self.train.get_stats()
            value = Normalizer(stats=stats)
        self.train.normalizer = value
        if self.valid is not None:
            self.valid.normalizer = value
        if self.test is not None:
            self.test.normalizer = value
        return value

    @normalizer.setter
    def normalizer(self, value):
        self.train.normalizer = value
        if self.valid is not None:
            self.valid.normalizer = value
        if self.test is not None:
            self.test.normalizer = value
        return value

    def normalize(self):
        ''' Normalize dataset for having a training dataset with
        zero mean and standard deviation of one.
        '''
        if not self.normalizer.is_set:
            self.normalizer.init()
        train = type(self.train)(
            data_x=self.normalizer.normalize(self.train.x),
            data_y=self.train.y)
        if self.valid is not None:
            valid = type(self.valid)(
                data_x=self.normalizer.normalize(self.valid.x),
                data_y=self.valid.y)
        else:
            valid = None
        if self.test is not None:
            test = type(self.test)(
                data_x=self.normalizer.normalize(self.test.x),
                data_y=self.test.y)
        else:
            test = None
        return Datasets(train=train, valid=valid, test=test)

    @tdl.core.OptionalProperty
    def mapper(self, x_mapper=None, y_mapper=None):
        mapper = tdl.core.SimpleNamespace(x=x_mapper,
                                          y=y_mapper)
        self.train.mapper = mapper
        if self.valid is not None:
            self.valid.mapper = mapper
        if self.test is not None:
            self.test.mapper = mapper
        return mapper

    @mapper.setter
    def mapper(self, value):
        self.train.mapper = value
        if self.valid is not None:
            self.valid.mapper = value
        if self.test is not None:
            self.test.mapper = value
        return value


class Normalizer(object):
    def __init__(self, stats):
        self.stats = stats

    def normalize(self, batch):
        return (batch - self.stats.mean) / self.stats.stddev

    def denormalize(self, batch):
        return (batch * self.stats.stddev) + self.stats.mean

    def __call__(self, batch):
        return self.normalize(batch)


class MaxMinNormalizer(Normalizer):
    def __init__(self, stats):
        self.stats = stats

    def normalize(self, batch):
        medium = (self.stats.max + self.stats.min)/2.0
        diff = (self.stats.max - self.stats.min)
        return (batch - medium) / diff

    def denormalize(self, batch):
        medium = (self.stats.max + self.stats.min)/2.0
        diff = (self.stats.max - self.stats.min)
        return (batch * diff) + medium

    def __call__(self, batch):
        return self.normalize(batch)


class StackedMapper(object):
    @property
    def mappers(self):
        return self._mappers

    def __init__(self, mappers=None):
        self._mappers = (list() if mappers is None
                         else mappers)

    def append(self, mapper):
        self.mappers.append(mapper)

    def __call__(self, x):
        for mapper in self.mappers:
            x = mapper(x)
        return x


class ReplaceMap(object):
    def __init__(self, map):
        self.map = map

    def __call__(self, x):
        return (x.replace(self.map) if isinstance(x, pd.DataFrame)
                else self.map[x])


class OneHotMap(object):
    @property
    def n_classes(self):
        return self._n_classes

    def __init__(self, n_classes):
        self._n_classes = n_classes

    def __call__(self, x):
        return dense_to_one_hot(x, num_classes=self.n_classes)
