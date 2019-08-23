import numpy as np
import pandas as pd
import twodlearn as tdl
import twodlearn.datasets as tdld
import twodlearn.datasets.base


class UnbalancedDataset(object):
    @property
    def datasets(self):
        return self._datasets

    @tdl.core.LazzyProperty
    def n_samples(self):
        return sum([data.n_samples for _, data in self.datasets.items()])

    @property
    def histogram(self):
        if not hasattr(self, '_histogram'):
            histogram = pd.DataFrame.from_dict(
                {c: data.n_samples
                 for c, data in self.datasets.items()},
                orient='index')
            self._histogram = histogram.rename({0: 'labels'}, axis=1)\
                                       .sort_values('labels', ascending=False)
        return self._histogram

    @property
    def labels(self):
        if not hasattr(self, '_labels'):
            self._labels = tuple(self.datasets.keys())
        return self._labels

    def filter_by_size(self, minimum_size):
        datasets = [self.datasets[label]
                    for label in self.labels
                    if self.histogram['labels'][label] > minimum_size]
        return UnbalancedDataset(datasets=datasets)

    def _next_batch_replace(self, dataset, batch_size):
        '''Next batch with replacement if needed'''
        return (dataset.next_batch(batch_size)[0]
                if dataset.n_samples > batch_size
                else dataset.sample(batch_size, replace=True)[0])

    def _samples_per_class(self, n_samples):
        # TODO: profile!!!
        samp_per_class = {label: n_samples // len(self.datasets)
                          for label in self.labels}
        missing = n_samples % len(self.datasets)
        if missing:
            extra = np.random.choice(len(self.datasets), missing,
                                     replace=True)
            for i in extra:
                samp_per_class[samp_per_class.keys()[i]] += 1
        return samp_per_class

    def sample(self, n_samples, replace=False):
        samp_per_class = self._samples_per_class(n_samples)
        samples_x = [self.datasets[label].sample(samp_per_class[label],
                                                 replace=replace)[0]
                     for label in self.labels]
        if self.mapper.is_set:
            samples_x = [self.mapper.x(batch)
                         for batch in samples_x]
        samples_y = [np.full(shape=[samp_per_class[label], 1],
                             fill_value=(self.mapper.y(label)
                                         if self.mapper.is_set
                                         else label))
                     for label in self.labels]
        return (np.concatenate(samples_x, axis=0),
                np.concatenate(samples_y, axis=0))

    def next_batch(self, batch_size):
        samp_per_class = self._samples_per_class(batch_size)
        batches_x = [self._next_batch_replace(
                                dataset=self.datasets[label],
                                batch_size=samp_per_class[label])
                     for label in self.labels]
        if self.mapper.is_set:
            batches_x = [self.mapper.x(batch)
                         for batch in batches_x]

        batches_y = list()
        for label in self.labels:
            mapped_label = (self.mapper.y(label) if self.mapper.is_set
                            else label)
            label_shape = (1 if isinstance(mapped_label, int)
                           else mapped_label.shape[-1])
            label_array = np.full(shape=[samp_per_class[label], label_shape],
                                  fill_value=mapped_label)
            batches_y.append(label_array)

        if self.normalizer.is_set:
            batches_x = [self.normalizer.normalize(batch)
                         for batch in batches_x]
        batch_x = np.concatenate(batches_x, axis=0)
        batch_y = np.concatenate(batches_y, axis=0)
        return batch_x, batch_y

    def get_stats(self):
        def get_class_stats(data):
            data = (self.mapper.x(data.x) if self.mapper.is_set
                    else data.x)
            stats = tdl.common.SimpleNamespace(
                mean=data.mean(axis=0),
                stddev=data.std(axis=0),
                min=data.min(axis=0),
                max=data.min(axis=0))
            return stats
        stats = [get_class_stats(data)
                 for _, data in self.datasets.items()]

        stats = tdl.common.SimpleNamespace(
            mean=np.stack([s.mean for s in stats], axis=0).mean(axis=0),
            stddev=np.stack([s.stddev for s in stats], axis=0).mean(axis=0),
            min=np.stack([s.min for s in stats], axis=0).min(axis=0),
            max=np.stack([s.max for s in stats], axis=0).max(axis=0))
        return stats

    @tdl.core.OptionalProperty
    def normalizer(self, value=None):
        if value is None:
            stats = self.get_stats()
            value = tdld.base.Normalizer(stats=stats)
        return value

    @tdl.core.OptionalProperty
    def mapper(self, x_mapper=None, y_mapper=None):
        x_mapper = (lambda x: x if x_mapper is None
                    else x_mapper)
        y_mapper = (lambda y: y if y_mapper is None
                    else y_mapper)
        return tdl.core.SimpleNamespace(x=x_mapper, y=y_mapper)

    def shuffle(self):
        [data.shuffle() for _, data in self.datasets.items()]

    def split(self, values, shuffle=True):
        """Split the dataset according to the percentages provided in values

        Args:
            values (list): percentages used for splitting the dataset
            shuffle (bool): true if shuffle the dataset before splitting
                (default True)
        Returns:
            list(UnbalancedDataset): list with the new datasets
        """
        if not isinstance(values, list):
            raise ValueError('values must be a list with the percentages of '
                             'each split')
        if (any([not 0.0 < value < 1.0 for value in values])
                or sum(values) > 1.0):
            raise ValueError('values must be a list of percentages that '
                             '(0 to 1) that sum up to one')
        if sum(values) < 1.0:
            values.append(1.0 - sum(values))
        if shuffle:
            [data.shuffle() for _, data in self.datasets.items()]
        cum_values = np.cumsum(values)
        cum_values[-1] = 1   # make sure that last is exactly one
        splits_data = list()
        for i in range(len(values)):
            def start(x):
                return int(x.n_samples * (0 if i == 0 else cum_values[i-1]))

            def end(x):
                return int(x.n_samples * cum_values[i])
            split_i = {label:
                       type(data)(*(data.iloc[start(data):end(data), :]))
                       for label, data in self.datasets.items()}
            splits_data.append(split_i)

        DatasetType = (DfUnbalancedDataset
                       if isinstance(self, DfUnbalancedDataset)
                       else UnbalancedDataset)
        return [DatasetType(datasets=data) for data in splits_data]

    def __init__(self, datasets):
        if any([not isinstance(data, (np.ndarray, tdld.base.Dataset))
                for _, data in datasets.items()]):
            raise ValueError('provided data must be a dictionary of '
                             'tdld.base.Dataset or np.ndarray elements')
        self._datasets = {label: (data if isinstance(data, tdld.base.Dataset)
                                  else tdld.base.Dataset(data))
                          for label, data in datasets.items()}

    @staticmethod
    def _split_data(data_x, data_y, DatasetClass=tdld.base.Dataset):
        datasets = {yi: DatasetClass(data_x=data_x.loc[data_y == yi])
                    for yi in data_y.unique()}
        return datasets

    @classmethod
    def from_data(cls, data_x, data_y, DatasetClass=tdld.base.Dataset):
        datasets = cls._split_data(data_x=data_x, data_y=data_y,
                                   DatasetClass=DatasetClass)
        return cls(datasets=datasets)


class DfUnbalancedDataset(UnbalancedDataset):
    @tdl.core.LazzyProperty
    def dtypes(self):
        dtypes = [data.x.dtypes for _, data in self.datasets.items()]
        assert all([all(dtypes[0] == dtypes_i) for dtypes_i in dtypes]),\
            'datasets must have same dtypes'
        return dtypes[0]

    @tdl.core.LazzyProperty
    def columns(self):
        columns = [data.x.columns for _, data in self.datasets.items()]
        assert all([all(columns[0] == columns_i) for columns_i in columns]),\
            'datasets must have same columns'
        return columns[0]

    def sample(self, n_samples, replace=False):
        samp_per_class = self._samples_per_class(n_samples)
        samples_x = [self.datasets[label].sample(samp_per_class[label],
                                                 replace=replace)[0]
                     for label in self.labels]
        if self.mapper.is_set:
            samples_x = [self.mapper.x(batch)
                         for batch in samples_x]

        samples_y = list()
        for label in self.labels:
            mapped_label = (self.mapper.y(label) if self.mapper.is_set
                            else label)
            label_shape = (1 if isinstance(mapped_label, int)
                           else mapped_label.shape[-1])
            label_array = np.full(shape=[samp_per_class[label], label_shape],
                                  fill_value=mapped_label)
            samples_y.append(label_array)

        samples_x = pd.concat(samples_x, axis=0)
        if samples_y[0].shape[1] == 1:
            samples_y = pd.DataFrame(np.concatenate(samples_y, axis=0),
                                     index=samples_x.index,
                                     columns=['labels'])
        else:
            samples_y = pd.DataFrame(np.concatenate(samples_y, axis=0),
                                     index=samples_x.index)
        return samples_x, samples_y

    @tdl.core.OptionalProperty
    def maps(self):
        '''maps used to convert strings to integers'''
        def unique(col):
            values = [set(data.x[col].unique())
                      for _, data in self.datasets.items()]
            return set.union(*values)

        maps = dict()
        for col in self.columns:
            if self.dtypes[col] == np.object:
                maps[col] = {val: num for num, val in enumerate(unique(col))}
        maps['labels'] = \
            {val: num for num, val in enumerate(self.labels)}
        if self.mapper.is_set:
            raise ValueError('mapper is already set')
        else:
            self.mapper.init(x_mapper=maps, y_mapper=maps)
        return maps

    @maps.setter
    def maps(self, value):
        self.mapper.init(x_mapper=value, y_mapper=value)
        return value

    @tdl.core.OptionalProperty
    def mapper(self, x_mapper=None, y_mapper=None):
        if (x_mapper is None) and self.maps.is_set:
            x_mapper = tdld.base.ReplaceMap(self.maps.value)
        if isinstance(x_mapper, dict):
            x_mapper = tdld.base.ReplaceMap(x_mapper)

        if (y_mapper is None) and self.maps.is_set:
            y_mapper = tdld.base.ReplaceMap(self.maps.value['labels'])
        if isinstance(y_mapper, dict):
            y_mapper = tdld.base.ReplaceMap(y_mapper['labels'])

        x_mapper = (x_mapper if isinstance(x_mapper, tdld.base.StackedMapper)
                    else tdld.base.StackedMapper([x_mapper]))
        y_mapper = (y_mapper if isinstance(y_mapper, tdld.base.StackedMapper)
                    else tdld.base.StackedMapper([y_mapper]))
        return tdl.core.SimpleNamespace(x=x_mapper,
                                        y=y_mapper)

    def filter_by_size(self, minimum_size):
        datasets = {label: self.datasets[label]
                    for label in self.labels
                    if self.histogram['labels'][label] > minimum_size}
        return DfUnbalancedDataset(datasets=datasets)

    def get_stats(self):
        def get_class_stats(data):
            data = (self.mapper.x(data.x) if self.mapper.is_set
                    else data.x)
            stats = tdl.common.SimpleNamespace(
                mean=data.mean(axis=0),
                stddev=data.std(axis=0),
                min=data.min(axis=0),
                max=data.max(axis=0))
            return stats
        stats = [get_class_stats(data)
                 for _, data in self.datasets.items()]

        stats = tdl.common.SimpleNamespace(
            mean=pd.DataFrame([s.mean for s in stats]).mean(axis=0),
            stddev=pd.DataFrame([s.stddev for s in stats]).mean(axis=0),
            min=pd.DataFrame([s.min for s in stats]).min(axis=0),
            max=pd.DataFrame([s.max for s in stats]).max(axis=0))
        return stats

    def __init__(self, datasets):
        if any([not isinstance(data, (pd.DataFrame, tdld.base.DfDataset))
                for _, data in datasets.items()]):
            raise ValueError('provided data must be a dictionary of '
                             'tdld.base.DfDataset or pd.DataFrame elements')
        self._datasets = {label: (data if isinstance(data, tdld.base.DfDataset)
                                  else tdld.base.DfDataset(data))
                          for label, data in datasets.items()}

    def __getitem__(self, key):
        assert isinstance(key, list),\
            'key must be a list of columns'
        datasets = {label: tdld.base.DfDataset(data.x[key])
                    for label, data in self.datasets.items()}
        return DfUnbalancedDataset(datasets=datasets)
