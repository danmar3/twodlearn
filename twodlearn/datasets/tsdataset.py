from __future__ import division
from __future__ import print_function

import re
import pickle
import twodlearn as tdl
import numbers
import datetime
import warnings
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import random
import xarray as xr
import pdb

USE_XARRAY = False      # Return batch arrays in xarray format


def signal_edges(series, dtype=int):
    ''' detect signal edges '''
    return series.ne(series.shift().bfill()).astype(dtype)

# ------------------------- Dataset ------------------------------ #


class AsynchronousRecord:
    # data: DataFrame
    # start_time: date-time where the corresponding recording starts
    # end_time: date-time where the corresponding recording ends
    # prop: system configuration during the recording of the data
    def __init__(self, data,
                 start_time=None, prop=None, name=''):
        if isinstance(data, list):
            self._data = data
        else:
            assert(isinstance(data, pd.DataFrame))
            self._data = [data]

        if start_time is None:
            # self._start_time = datetime.datetime.now()
            self._relative_index = False
        else:
            self._relative_index = True
            self._start_time = start_time
            assert not isinstance(self._data[0].index[0],
                                  pd._libs.tslib.Timestamp), \
                'Index cannot be a date if relative'

        self.prop = prop
        self.name = name

    @property
    def start_time(self):
        if self._relative_index:
            return self._start_time + \
                datetime.timedelta(0, self._data[0].index[0])
        else:
            return self._data[0].index[0]

    @property
    def end_time(self):
        if self._relative_index:
            return self.start_time + \
                datetime.timedelta(0, self._data[0].index[-1])
        else:
            return self._data[0].index[-1]

    @property
    def data(self):
        if len(self._data) == 1:
            return self._data[0]
        else:
            raise NotImplementedError('When multiple data, use _data')

    @property
    def n_samples(self):
        if len(self._data) == 1:
            return self._data[0].shape[0]
        else:
            raise NotImplementedError('When multiple data, use _data')

    def prop2list(self):
        return list(self.prop.values())

    def prop2array(self):
        return np.array(self.prop2list())

    def __getitem__(self, idx):
        if len(self._data) == 1:
            return self._data[idx]
        else:
            return pd.concat([i[idx] for i in self._data if (idx in i.keys())],
                             axis=1)

    def collapse(self):
        ''' Collapse the list of DataFrame in data to a single DataFrame
        '''
        if len(self._data) > 1:
            # resample to one second
            for d in range(len(self._data)):
                self._data[d] = self._data[d].resample('1S').mean()
            # collapse
            self._data = [pd.concat(self._data, axis=1)]
            # move coordinates to seconds
            self._start_time = self._data[0].index[0]
            self._data[0].index = (self._data[0].index -
                                   self.start_time).total_seconds()
            # self.end_time = self.start_index +
            #                 pd.tseries.offsets.Milli(1000*self._data[0].index[-1])

    def mean(self, group):
        try:
            return (self.data[group]).mean()
        except KeyError:
            return np.nan

    def std(self, group):
        try:
            return (self.data[group]).std()
        except KeyError:
            return np.nan

    def set_groups(self, group_tags):
        self._groups = dict()
        for group, tags in group_tags.items():
            self._groups[group] = self.data.filter(items=tags).values

    def get_group(self, group_name):
        ''' returns the data for the given group as an np.ndarray '''
        return self._groups[group_name]


RecordSaveData = collections.namedtuple('RecordSaveData',
                                        ['data', 'start_time', 'prop', 'name'])


class Record(object):
    # data: DataFrame
    # start_time: date-time where the corresponding recording starts
    # end_time: date-time where the corresponding recording ends
    # prop: system configuration during the recording of the data
    def __init__(self, data,
                 start_time=None, prop=None, name=''):

        assert isinstance(data, pd.DataFrame),\
            'provided data is not a pandas DataFrame'
        self._data = data
        self.start_time = start_time
        self.prop = prop
        self.name = name
        self._group_tags = None

    def copy(self):
        """Creates a new copy of the record.

        Returns:
            Record: new copy of the record.
        """
        i = 0
        while 'copy{}'.format(i) in self.name:
            i += 1
        new_name = '{}_copy{}'.format(self.name, i)
        new_record = Record(data=self.data.copy(),
                            prop=self.prop,
                            name=new_name)
        if self.group_tags is not None:
            new_record.set_groups(self.group_tags)
        return new_record

    @classmethod
    def from_saved_data(cls, data):
        ''' Create record from saved Record.SaveData '''
        return cls(data=data.data,
                   prop=data.prop,
                   start_time=data.start_time,
                   name=data.name)

    def get_save_data(self):
        return RecordSaveData(
            data=self.data,
            prop=self.prop,
            start_time=(self.start_time if self._relative_index
                        else None),
            name=self.name)

    @property
    def start_time(self):
        if self._relative_index:
            if isinstance(self._start_time, numbers.Real):
                return self._start_time + self.data.index[0]
            else:
                return self._start_time + \
                    datetime.timedelta(0, self.data.index[0])
        else:
            return None  # self.data.index[0]

    @start_time.setter
    def start_time(self, start_time):
        # warnings.warn('start_time and relative index is deprecated and will '
        #               'soon be removed', DeprecationWarning)
        if start_time is None:
            self._relative_index = False
        elif isinstance(start_time, numbers.Real):
            if start_time == 0.0:
                self._relative_index = False
            else:
                self._relative_index = True
                self._start_time = start_time
        else:
            self._relative_index = True
            self._start_time = start_time
            assert not isinstance(self.data.index[0],
                                  pd._libs.tslib.Timestamp), \
                'Index cannot be a date if relative'
        if self._relative_index is True:
            warnings.warn("start_time and relative index is in development "
                          "and it might be unstable",
                          RuntimeWarning)

    @property
    def end_time(self):
        # warnings.warn('start_time, end_time and relative index is deprecated '
        #               'and will soon be removed', DeprecationWarning)
        if self._relative_index:
            return self.start_time + \
                datetime.timedelta(0, self.data.index[-1])
        else:
            return self.data.index[-1]

    @property
    def data(self):
        return self._data

    @property
    def n_samples(self):
        return self.data.shape[0]

    @property
    def columns(self):
        return self.data.columns.values

    def prop2list(self):
        return list(self.prop.values())

    def prop2array(self):
        return np.array(self.prop2list())

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def group_tags(self):
        """ dictionary with the names of the groups and the associated columns
        """
        return self._group_tags

    def set_groups(self, group_tags):
        """Group the features on the data according to the provided tags.

        Args:
            group_tags (dict): dictionary where the keys correspond to the name
                for the groups and the values the columns for the groups
        """
        self._groups = dict()
        for group, tags in group_tags.items():
            self._groups[group] = self.data.filter(items=tags).values
        self._group_tags = group_tags

    def get_group(self, group_name):
        """Get the np.Array corresponding to the given group.

        Args:
            group_name (str): name of the group.

        Returns:
            np.Array: Array corresponding to the data of the group.
        """
        return self._groups[group_name]

    def split_continuous(self, column_name):
        ''' splits the record following continuous chunks of
        data from the provided series '''
        edges = signal_edges(self.data[column_name]).cumsum()
        records = list()
        for i in range(edges.max() + 1):
            data_i = self.data.iloc[edges.eq(i).values].copy()
            prop_i = self.prop.copy()
            prop_i[column_name] = data_i[column_name].iloc[0]
            records.append(Record(data=data_i,
                                  prop=prop_i))
        return records


TSDatasetSaver = collections.namedtuple(
    'TSDatasetSaver', ['records_data', 'group_tags'])


class TSDataset(object):
    # records: list of Record
    class BatchNormalizer(object):
        @property
        def mu(self):
            return self._mu

        @mu.setter
        def mu(self, mu):
            if mu is None:
                self._mu = None
                return
            assert isinstance(mu, dict), 'mu should be a dictionary'
            self._mu = dict()
            for group_name, group_mu in mu.items():
                if isinstance(group_mu, pd.Series):
                    self._mu[group_name] = group_mu.values
                elif isinstance(group_mu, np.ndarray):
                    self._mu[group_name] = group_mu
                else:
                    raise ValueError("elements of mu should be pd.Series "
                                     "or np.ndarray")

        @property
        def std(self):
            return self._std

        @std.setter
        def std(self, std):
            if std is None:
                self._std = None
                return
            assert isinstance(std, dict), 'std should be a dictionary'
            self._std = dict()
            for group_name, group_std in std.items():
                if isinstance(group_std, pd.Series):
                    self._std[group_name] = group_std.values
                elif isinstance(group_std, np.ndarray):
                    self._std[group_name] = group_std
                else:
                    raise ValueError("elements of std should be pd.Series "
                                     "or np.ndarray")

        def reset(self):
            self._mu = None
            self._std = None

        def normalize(self, batch):
            if self.mu is not None:
                for group, mu in self.mu.items():
                    batch[group] = batch[group] - mu
            if self.std is not None:
                for group, std in self.std.items():
                    batch[group] = batch[group] / std
            return batch

        def __init__(self):
            self._mu = None
            self._std = None

    @property
    def n_samples(self):
        ''' total number of samples in all records '''
        return self._n_samples

    @property
    def normalizer(self):
        return self._normalizer

    def __getitem__(self, idx):
        return self.records[idx]

    def __init__(self, records=[]):
        # For batch generation
        self.records = list()
        self.batch_size = 0
        self.window_size = 0
        self._normalizer = TSDataset.BatchNormalizer()

        # Add the provided records
        if(len(records) > 0):
            self.records = records
            self.set_groups({'all': None})
            # update props
            self.update_props()

    @classmethod
    def from_saved_data(cls, saved_data):
        if saved_data is None:
            # TODO: improve this
            return None

        records = [Record.from_saved_data(record_data)
                   for record_data in saved_data.records_data]
        dataset = cls(records=records)
        dataset.set_groups(saved_data.group_tags)
        return dataset

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            dataset = cls.from_saved_data(data),
        return dataset

    def get_save_data(self):
        data = TSDatasetSaver(records_data=[r.get_save_data()
                                            for r in self.records],
                              group_tags=self.group_tags)
        return data

    def save(self, filename):
        data = self.get_save_data()
        output_file = open(filename, 'wb')
        pickle.dump(data, output_file)
        output_file.close()

    def update_props(self):
        ''' updates static properties. This functions is called by add_record()
        after a new record has been added to the dataset
        '''
        self._n_samples = sum([r.data.shape[0] for r in self.records])
        self._next_batch_array = None  # used to accelerate "next_batch"
        self.cummulative_n_samples(recompute=True)
        if self.batch_size > 0:
            self.reset_cursors()

    def as_array(self):
        '''Returns the dataset as arrays separated in the groups'''
        out_arrays = dict()
        for group in self.group_tags.keys():
            out_arrays[group] = \
                np.concatenate([r.get_group(group)
                                for r in self.records], axis=0)
        return out_arrays

    @property
    def columns(self):
        columns = self.records[0].columns
        assert all([(r.columns == columns).all() for r in self.records])
        return columns

    def n_vars(self, group=None):
        if group is None:
            # return the total number of elements
            return max([r.data.shape[1] for r in self.records])
        else:
            # return max([r.data.filter(items=self.group_tags[group]).shape[1]
            #            for r in self.records])
            return max([r.get_group(group).shape[1]
                        for r in self.records])

    def add_record(self, other):
        ''' add a record into the dataset
        @type other: Record
        @param other: record to be added into the dataset
        '''
        if isinstance(other, list):
            assert(isinstance(other[0], Record))
            raise NotImplementedError('Adding a list of records has not been '
                                      'implemented')
        else:
            assert(isinstance(other, Record))
            self.records.append(other)

            # set groups in the record
            self.records[-1].set_groups(self.group_tags)

        # update properties
        self.update_props()

    def set_groups(self, group_tags):
        self.group_tags = dict()
        for group, tags in group_tags.items():
            if isinstance(tags, list):
                self.group_tags[group] = tags
            elif tags is None:
                # TODO: improve this
                self.group_tags[group] = self.records[0].data.keys().tolist()
            else:
                cols_name = self.records[0].data.keys()
                tag_list = [m.group(0) for m in [re.search(tags, tag)
                                                 for tag in cols_name] if m]
                # pdb.set_trace()  # (DEBUG)
                self.group_tags[group] = tag_list

        # setup groups in the records
        for r in self.records:
            r.set_groups(self.group_tags)

        # re-initialize normalizer
        self.normalizer.reset()

    DatasetStats = collections.namedtuple('DatasetStats',
                                          ['mean', 'stddev', 'min', 'max',
                                           'n_samples'])

    @property
    def dtypes(self):
        dtypes = collections.defaultdict(set)
        for r in self.records:
            for k, v in r.data.dtypes.items():
                dtypes[v].add(k)
        return dtypes

    def get_stats(self, groups=None):
        ''' Obtain mean and standard deviation of the dataset to be used
        for normalization

        @param groups: list of the group names that you want to measure
        '''
        mu_comp = dict()
        std_comp = dict()
        min_comp = dict()
        max_comp = dict()
        for group in groups:
            # compute mu
            _mu = np.zeros(self.n_vars(group))
            _min = np.full(self.n_vars(group), np.inf)
            _max = np.full(self.n_vars(group), -np.inf)
            for r in self.records:
                _mu += r.get_group(group).mean(axis=0)
                _min = np.minimum(_min, r.get_group(group).min(axis=0))
                _max = np.maximum(_max, r.get_group(group).max(axis=0))
            mu_comp[group] = _mu / len(self.records)
            min_comp[group] = _min
            max_comp[group] = _max
            # compute stddev
            _std = np.zeros(self.n_vars(group))
            _n = 0
            for r in self.records:
                n_r = r.get_group(group).shape[0]
                mu_r = r.get_group(group).mean(axis=0)
                std_r = r.get_group(group).std(axis=0)

                _std += n_r * (std_r**2 + ((mu_r - mu_comp[group])**2))
                _n += n_r

                # mu[group] += r.data.filter(items=tags).mean()
                # std[group] += r.data.filter(items=tags).std()
            std_comp[group] = np.sqrt(_std / (_n - 1))
        stats = TSDataset.DatasetStats(mean=mu_comp, stddev=std_comp,
                                       min=min_comp, max=max_comp,
                                       n_samples=self.n_samples)
        return stats

    def normalize(self, groups=None, mu=None, std=None):
        ''' Obtain mean and standard deviation of the dataset to be used
        for normalization

        @param groups: list of the group names that you want to normalize
        '''
        if groups is None:
            groups = self.group_tags.keys()

        # initialize mu and std
        self.normalizer.reset()

        # calculate mu and std
        if (mu is None or std is None):
            mu_comp = dict()
            std_comp = dict()
            for group in groups:
                _mu = np.zeros(self.n_vars(group))
                for r in self.records:
                    _mu += r.get_group(group).mean(axis=0)
                mu_comp[group] = _mu / len(self.records)

                _std = np.zeros(self.n_vars(group))
                _n = 0
                for r in self.records:
                    n_r = r.get_group(group).shape[0]
                    mu_r = r.get_group(group).mean(axis=0)
                    std_r = r.get_group(group).std(axis=0)

                    _std += n_r * (std_r**2 + ((mu_r - mu_comp[group])**2))
                    _n += n_r

                    # mu[group] += r.data.filter(items=tags).mean()
                    # std[group] += r.data.filter(items=tags).std()
                std_comp[group] = np.sqrt(_std / (_n - 1))
        if mu is not None:
            self.normalizer.mu = mu
        else:
            self.normalizer.mu = mu_comp
        if std is not None:
            self.normalizer.std = std
        else:
            self.normalizer.std = std_comp

    def next_batch_discontinuous(self, batch_size):
        ''' Get a batch when window_size is 1

        This function is used by next_batch, is not intended for being
        used outside the class.
        '''
        raise NotImplementedError('not yet implemented')

        def shuffle(dataset):
            perm = np.arange(dataset.n_samples)
            np.random.shuffle(perm)
            for group in dataset._next_batch_array.keys():
                dataset._next_batch_array[group] = \
                    dataset._next_batch_array[group][perm]

        if self._next_batch_array is None:
            self._next_batch_array = self.as_array()
            shuffle(self)

        # use cursor[0]
        pointer = self.cursors[0].global_pointer
        if (pointer + batch_size) > (self.n_samples):
            pointer = (pointer + batch_size) % self.n_samples
            shuffle(self)

        batch = dict()
        for group in self._next_batch_array.keys():
            batch[group] = \
                self._next_batch_array[group][pointer:(
                    pointer + batch_size), :]
            batch[group] = np.expand_dims(batch[group], axis=0)
        return batch

    def cummulative_n_samples(self, recompute=False):
        if recompute:
            self._cummulative_n_samples = \
                np.cumsum(np.array([r.n_samples for r in self.records]))
        return self._cummulative_n_samples

    class Cursor(object):
        ''' Manages one of the continuous elements of the batch.
        Hence, there are as many cursors as elements in the batch '''
        @property
        def cummulative_n_samples(self):
            return self.dataset.cummulative_n_samples()

        @property
        def record_id(self):
            return np.argmax(self.global_pointer < self.cummulative_n_samples)

        @property
        def record(self):
            return self.dataset.records[self.record_id]

        @property
        def local_pointer(self):
            return (self.global_pointer -
                    (self.cummulative_n_samples[self.record_id] -
                     self.record.n_samples))

        @local_pointer.setter
        def local_pointer(self, value):
            self.global_pointer = (self.cummulative_n_samples[self.record_id] -
                                   self.record.n_samples) + value

        def next_sequence(self, window_size):
            # confirm that the current window stays in the page
            counter = 0
            while (self.global_pointer + window_size >
                   self.cummulative_n_samples[self.record_id]):
                self.global_pointer = ((self.global_pointer + window_size) %
                                       self.dataset.n_samples)
                if (self.global_pointer + window_size >
                        self.cummulative_n_samples[self.record_id]):
                    self.local_pointer = 0
                    space_left = self.record.n_samples - window_size
                    if space_left > 0:
                        self.global_pointer += random.randrange(space_left)
                counter += 1
                if counter > 100:
                    raise RuntimeError('Unable to fetch continuous data '
                                       'after {} trials'.format(100))
            # Obtain next window_size sequence
            sequence = dict()
            for group, tags in self.dataset.group_tags.items():
                data_group = self.record.get_group(group)
                sequence[group] = \
                    data_group[self.local_pointer:
                               self.local_pointer + window_size, :]
            # Update pointer
            self.global_pointer = ((self.global_pointer + window_size) %
                                   self.dataset.n_samples)
            return sequence

        def __init__(self, dataset, global_pointer):
            self.dataset = dataset
            self.global_pointer = global_pointer

    def reset_cursors(self, batch_size=None):
        batch_size = (batch_size if batch_size is not None
                      else self.batch_size)
        spacing = self.n_samples // batch_size  # space between cursors
        # set the cursor for each batch
        self.cursors = [TSDataset.Cursor(self, offset * spacing)
                        for offset in range(batch_size)]

    def next_batch(self, window_size, batch_size, reset=False):
        '''Returns the next batch_size sequences of length window_size.

        Args:
            window_size:
            batch_size:
            reset: reset the cursors that point where data is currently being
                   extracted
        Returns:
            dict: A dictionary with the batch samples, the format is: ::

                batch[group] = array[window_size, batch_size, n_vars(group)]
        '''
        if (reset or
                (batch_size != self.batch_size) or
                (window_size != self.window_size)):
            self.batch_size = batch_size
            self.window_size = window_size
            spacing = self.n_samples // batch_size  # space between cursors
            # set the cursor for each batch
            self.cursors = [TSDataset.Cursor(self, offset * spacing)
                            for offset in range(batch_size)]

        # If a window size of 1
        if window_size == 1:
            batch = self.next_batch_discontinuous(batch_size)
        else:
            # Generate batch
            batch = dict()
            for group in self.group_tags.keys():
                batch[group] = np.zeros(
                    [window_size, batch_size, self.n_vars(group)])

            # each element of the batch has a cursor
            for b in range(batch_size):
                data_b = self.cursors[b].next_sequence(window_size)
                for group, tags in self.group_tags.items():
                    batch[group][:, b, :] = data_b[group]

        # Normalize
        batch = self.normalizer.normalize(batch)
        # Convert to xarray
        if USE_XARRAY:
            batch = {name: xr.DataArray(
                data,
                coords={'features': self.group_tags[name]},
                #        pd.RangeIndex(0, output_x.shape[1]),
                #        pd.RangeIndex(0, output_x.shape[2])],
                dims=['time', 'batch', 'features'],
                name=name)
                    for name, data in batch.items()}
        return batch

    def next_windowed_batch(self, sequences_length, batch_size, window_size,
                            groups=None, reset=False):
        ''' Returns the next batch where each sample contains a sequence off
        window_size elements

        Args:
            sequences_length: length of the sequences
            batch_size: number of sequences
            window_size: size of the window
        Returns:
            dict: A dictionary with the batch samples. The format is: ::

                batch[group] = array[sequences_length, batch_size, n_vars(group)*window_size]
        '''
        batch = self.next_batch(sequences_length + window_size - 1,
                                batch_size,
                                reset)
        if groups is None:
            groups = list(batch.keys())
        out_batch = dict()
        for name, group_data in batch.items():
            if name in groups:
                unrolls = [group_data[k, :, :]
                           for k in range(group_data.shape[0])]
                sequences = list()
                for k in range(len(unrolls) - window_size + 1):
                    sequences.append(np.concatenate(unrolls[k:k + window_size],
                                                    axis=1))
                out_batch[name] = np.stack(sequences, axis=0)
            else:
                out_batch[name] = group_data[(window_size - 1):, :, :]

        return out_batch

    def get_prop_mat(self):
        prop_table = np.zeros([len(self.records),
                               len(self.records[0].prop.values())])

        for i in range(len(self.records)):
            record_i = self.records[i]
            prop_table[i, :] = record_i.prop2array()

        return prop_table, list(self.records[0].prop.keys())

    def get_prop_table(self):
        prop_table = np.zeros([len(self.records),
                               len(self.records[0].prop.values())])

        for i in range(len(self.records)):
            record_i = self.records[i]
            prop_table[i, :] = record_i.prop2array()

        return pd.DataFrame(prop_table,
                            columns=list(self.records[0].prop.keys()))

    def split_continuous(self, column_name, min_samples=None):
        ''' splits the records following continuous chunks of
        data from the provided column '''
        records = list()
        assert column_name in self.columns,\
            'provided column is not in the records'
        for r in self.records:
            if column_name in r.columns:
                records += r.split_continuous(column_name)
        # Remove small records
        if min_samples is not None:
            records = [r for r in records
                       if r.n_samples > min_samples]
        # Build dataset
        dataset = TSDataset(records=records)
        if not ('all' in self.group_tags and
                len(self.group_tags) == 1):
            dataset.set_groups(self.group_tags)
        return dataset

    def to_dense(self):
        """Return a dense representation of the dataset.

        Returns:
            (array, length): a tuple of the dense array and the length
            of each record. The records are padded with nan values.
        """
        columns = self.records[0].columns

        def pad_data(data, max_len):
            return np.pad(data[columns], ((0, max_len-data.shape[0]),
                                          (0, 0)),
                          'constant', constant_values=np.nan)

        length = np.array([record.data.shape[0] for record in self.records])
        max_len = max(length)
        padded = xr.DataArray(
            np.stack([pad_data(record.data, max_len)
                      for record in self.records], axis=0),
            coords={'features': self.columns},
            dims=['batch', 'time', 'features'])
        return padded, length

    def to_tf_dataset(self, dtype=np.float32):
        """Get a tf.data.Dataset with a dense representation of the dataset.

        Returns:
            tf.data.Dataset: with elements 'data', 'length'. 'data' is a dense
            tensor representation of the dataset formated as
            (record, time, features). Records are padded with nan values.
        """
        data, length = self.to_dense()
        if len(self.group_tags) == 1:
            dataset = tf.data.Dataset.from_tensor_slices(
                {'data': data.values.astype(dtype), 'length': length})
        else:
            data = {group: data.loc[:, :, features].values.astype(dtype)
                    for group, features in self.group_tags.items()}
            dataset = tf.data.Dataset.from_tensor_slices(
                {'data': data, 'length': length})
        return dataset


def sample_batch_window(data, length, window_size, batch_size=None):
    """Sample continuous windows of window_size from tensor data.

    Args:
        data (tf.Tensor): dense representation of a set of continuous records.
            The format should be (record, time, features).
        length (type): length of each record.
        window_size (type): window size of the window to sample.
        batch_size (type): batch size of data. If not provided,
            batch_size = data.shape[0]
    Returns:
        tf.Tensor: continuous random continuous windows. The format is
        (record, time, features)
    """
    flatten = tdl.core.nest.flatten(data)

    if any([di.shape.ndims != 3 for di in flatten]):
        raise ValueError('provided data is not in the expected format: '
                         '(record, time, features).')
    if not all([di.shape[0].value == flatten[0].shape[0].value
                for di in flatten]):
        raise ValueError('provided nested data has different batch size.')

    if (batch_size is None) and (flatten[0].shape[0].value is not None):
        batch_size = flatten[0].shape[0].value
    else:
        batch_size = tf.shape(flatten[0])[0]
    batch_range = tf.range(0, batch_size, dtype=tf.int32)[..., tf.newaxis]
    u_index = tf.tile(batch_range, [1, window_size])
    v_init = tf.cast(
        tf.floor(tf.random.uniform(shape=[batch_size]) //
                 (1./(tf.cast(length, tf.float32)-window_size+1))),
        tf.int32)[..., tf.newaxis]
    v_index = (tf.range(0, window_size, dtype=tf.int32)[tf.newaxis, ...] +
               v_init)
    index = tf.stack([tf.reshape(u_index, [-1]),
                      tf.reshape(v_index, [-1])], axis=1)
    flatten_output = [tf.reshape(tf.gather_nd(di, index),
                                 [batch_size, window_size, di.shape[-1].value])
                      for di in flatten]
    return tdl.core.nest.pack_sequence_as(
        structure=data, flat_sequence=flatten_output), index


def sample_window(data, length, window_size):
    """Sample continuous windows of window_size from tensor data.

    Args:
        data (tf.Tensor): dense representation of a set of continuous records.
            The format should be (time, features).
        length (type): length of each record.
        window_size (type): window size of the window to sample.
    Returns:
        tf.Tensor: continuous random continuous windows. The format is
        (record, time, features)
    """
    flatten = tdl.core.nest.flatten(data)

    if any([di.shape.ndims != 2 for di in flatten]):
        raise ValueError('provided data is not in the expected format: '
                         '(record, time, features).')
    if not all([di.shape[0].value == flatten[0].shape[0].value
                for di in flatten]):
        raise ValueError('provided nested data has different time size.')
    t0 = tf.cast(
        tf.floor(tf.random.uniform(shape=()) //
                 (1./(tf.cast(length, tf.float32)-window_size+1))),
        tf.int32)
    flatten_output = [di[t0:t0+window_size, :] for di in flatten]
    return tdl.core.nest.pack_sequence_as(
        structure=data, flat_sequence=flatten_output), t0


DatasetsSaver = collections.namedtuple(
    'DatasetsSaver', ['train', 'valid', 'test'])


class TSDatasets(object):
    def __init__(self, train=None, valid=None, test=None):
        if isinstance(train, list):
            if isinstance(train[0], Record):
                self.train = TSDataset(records=train)
        elif isinstance(train, TSDataset):
            self.train = train
        elif train is None:
            self.train = None
        else:
            raise NotImplementedError('Only records supported for the moment')

        if isinstance(valid, list):
            if isinstance(valid[0], Record):
                self.valid = TSDataset(records=valid)
        elif isinstance(valid, TSDataset):
            self.valid = valid
        elif valid is None:
            self.valid = None
        else:
            raise NotImplementedError('Only records supported for the moment')

        if isinstance(test, list):
            if isinstance(test[0], Record):
                self.test = TSDataset(records=test)
        elif isinstance(test, TSDataset):
            self.test = test
        elif test is None:
            self.test = None
        else:
            raise NotImplementedError('Only records supported for the moment')

    def normalize(self, groups):
        self.train.normalize(groups)
        if self.valid is not None:
            self.valid.normalize(mu=self.train.normalizer.mu,
                                 std=self.train.normalizer.std)
        if self.test is not None:
            self.test.normalize(mu=self.train.normalizer.mu,
                                std=self.train.normalizer.std)

    def set_groups(self, group_tags):
        if self.train is not None:
            self.train.set_groups(group_tags)

        if self.valid is not None:
            self.valid.set_groups(group_tags)

        if self.test is not None:
            self.test.set_groups(group_tags)

    def get_save_data(self):
        if self.train is not None:
            train = self.train.get_save_data()
        else:
            train = None
        if self.valid is not None:
            valid = self.valid.get_save_data()
        else:
            valid = None
        if self.test is not None:
            test = self.test.get_save_data()
        else:
            test = None
        data = DatasetsSaver(train=train, valid=valid, test=test)
        return data

    def save(self, filename):
        data = self.get_save_data()
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    @classmethod
    def from_saved_file(cls, filename, encoding=None):
        data = Path(filename)
        if data.is_file():
            with open(filename, 'rb') as file:
                if encoding is not None:
                    data = pickle.load(file, encoding=encoding)
                else:
                    data = pickle.load(file)
            datasets = cls(train=TSDataset.from_saved_data(data.train),
                           valid=TSDataset.from_saved_data(data.valid),
                           test=TSDataset.from_saved_data(data.test))
            return datasets
        else:
            raise AttributeError('provided filename is not a file')
