
import os
import gzip
import numpy as np
import pandas as pd
import twodlearn.base
import twodlearn.datasets as tdld
import twodlearn.datasets.base
from twodlearn.datasets import unbalanced


class Kdd99Train(unbalanced.DfUnbalancedDataset):
    src = {'data': 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz',
           'names': 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names'}
    filename = {'data': 'kddcup.data.gz',
                'names': 'kddcup.names'}

    @property
    def work_directory(self):
        return self._data_folder

    @staticmethod
    def get_names(filename, work_directory):
        with open(os.path.join(work_directory, filename), 'r') as file:
            labels = file.readline().split(',')
            labels[-1] = labels[-1].replace('.\n', '')
            columns = list()
            for line in file:
                columns.append(line.split(':')[0])
        return labels, columns

    def _load_raw(self, work_directory):
        filepath = os.path.join(work_directory, self.filename['data'])
        labels, columns = self.get_names(self.filename['names'],
                                         work_directory)
        with gzip.open(filepath) as file:
            data = pd.read_csv(file, names=columns + ['labels'])
        data['labels'] = data['labels'].apply(lambda x: x.replace('.', ''))
        return data

    def __init__(self, work_directory, multi_class=True):
        for name, src in self.src.items():
            tdld.base.maybe_download(filename=self.filename[name],
                                     work_directory=work_directory,
                                     source_url=src)
        self.raw = self._load_raw(work_directory)
        if not multi_class:
            map = {'labels': {k: 'attack' for k in self.raw['labels'].unique()
                              if k != 'normal'}}
            self.raw = self.raw.replace(map)

        data_x = self.raw[[col for col in self.raw.columns
                          if col != 'labels']]
        data_y = self.raw['labels']

        self._datasets = self._split_data(data_x, data_y, tdld.base.DfDataset)


class Kdd99Test(Kdd99Train):
    src = {'data': 'http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz',
           'names': 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names'}
    filename = {'data': 'kddcup.data.gz',
                'names': 'kddcup.names'}


class Kdd99(tdld.base.Datasets):
    def __init__(self, work_directory, train_size=0.7,
                 min_samples=1000, attrs=None,
                 multi_class=True, one_hot=True):
        if attrs is None:
            attrs = ['urgent', 'su_attempted', 'root_shell',
                     'num_failed_logins', 'num_shells', 'num_access_files',
                     'num_file_creations', 'num_root', 'is_guest_login',
                     'dst_host_srv_diff_host_rate', 'num_compromised',
                     'diff_srv_rate', 'srv_diff_host_rate',
                     'dst_host_serror_rate', 'serror_rate',
                     'dst_host_srv_serror_rate', 'srv_serror_rate',
                     'dst_host_rerror_rate', 'rerror_rate',
                     'dst_host_srv_rerror_rate', 'srv_rerror_rate',
                     'dst_host_diff_srv_rate', 'logged_in',
                     'dst_host_same_src_port_rate', 'dst_host_same_srv_rate',
                     'same_srv_rate', 'hot', 'protocol_type', 'flag',
                     'service', 'srv_count', 'dst_host_srv_count', 'count',
                     'dst_host_count', 'duration', 'dst_bytes', 'src_bytes']

            # basic_attrs = ['duration', 'protocol_type', 'service',
            #                'src_bytes', 'dst_bytes', 'flag', 'land',
            #                'wrong_fragment', 'urgent']
        train = Kdd99Train(work_directory,
                           multi_class=multi_class).filter_by_size(1000)[attrs]
        maps = train.maps.init()
        splits = train.split([train_size])
        train = splits[0]
        valid = splits[1]
        test = Kdd99Test(work_directory,
                         multi_class=multi_class)[attrs]
        test_data = {label: test.datasets[label]
                     for label in train.labels}
        test = unbalanced.DfUnbalancedDataset(test_data)
        super(Kdd99, self).__init__(train=train, valid=valid, test=test)
        self.train.maps = maps
        self.mapper = train.mapper.value
        # normalizer
        stats = self.train.get_stats()
        no_norm = [key for key, value in stats.max.items()
                   if value < 5.0]
        for var in no_norm:
            stats.mean[var] = 0.0
            stats.stddev[var] = 1.0
        self.normalizer = tdld.base.Normalizer(stats)
        # one hot
        if one_hot and multi_class:
            self.mapper.y.append(tdld.base.OneHotMap(len(self.train.labels)))


class NslKddTrain(unbalanced.DfUnbalancedDataset):
    filename = {'data': 'KDDTrain+.csv',
                'columns': 'Field Names.csv',
                'labels': 'Attack Types.csv'}

    @property
    def src_directory(self):
        return self._src_directory

    @property
    def label_groups(self):
        return self._label_groups

    @staticmethod
    def read_columns(filename, work_directory):
        columns = list()
        with open(os.path.join(work_directory, filename), 'r') as file:
            for line in file:
                columns.append(line.split(',')[0])
        return columns

    @staticmethod
    def read_labels(filename, work_directory):
        label_groups = dict()
        with open(os.path.join(work_directory, filename), 'r') as file:
            data = file.read().split('\r')
            for line in data:
                line = line.split(',')
                label_groups[line[0]] = line[1]
        return label_groups

    def _load_raw(self, work_directory):
        columns = self.read_columns(self.filename['columns'], work_directory)
        filepath = os.path.join(work_directory, self.filename['data'])
        data = pd.read_csv(filepath, names=columns + ['labels'] + ['nan'])
        return data.iloc[:, :-1]

    def _load_dataset(self, work_directory, use_groups=False,
                      multi_class=True):
        self._label_groups = self.read_labels(self.filename['labels'],
                                              work_directory)
        raw = self._load_raw(work_directory)
        if use_groups and multi_class:
            map = {'labels': {k: v for k, v in self.label_groups.items()
                              if k != v}}
            raw = raw.replace(map)
        elif not multi_class:
            map = {'labels': {k: 'attack' for k, v in self.label_groups.items()
                              if k != 'normal'}}
            raw = raw.replace(map)

        data_x = raw[[col for col in raw.columns
                      if col != 'labels']]
        data_y = raw['labels']
        datasets = self._split_data(data_x, data_y, tdld.base.DfDataset)
        return datasets

    def __init__(self, src_directory, use_groups=False, multi_class=True):
        self._src_directory = src_directory
        self._datasets = self._load_dataset(src_directory,
                                            use_groups=use_groups,
                                            multi_class=multi_class)


class NslKddTest(NslKddTrain):
    filename = {'data': 'KDDTest+.csv',
                'columns': 'Field Names.csv',
                'labels': 'Attack Types.csv'}


class NslKdd(tdld.base.Datasets):
    def __init__(self, src_directory, use_groups=True, multi_class=True,
                 train_size=0.7, one_hot=True, min_samples=900, attrs=None):
        train = NslKddTrain(
            src_directory=src_directory,
            use_groups=use_groups,
            multi_class=multi_class).filter_by_size(min_samples)
        if attrs is not None:
            train = train[attrs]
        maps = train.maps.init()
        splits = train.split([train_size], shuffle=False)
        train = splits[0]
        valid = splits[1]

        test = NslKddTest(src_directory=src_directory,
                          use_groups=use_groups,
                          multi_class=multi_class)
        if attrs is not None:
            test = test[attrs]
        test_data = {label: test.datasets[label]
                     for label in train.labels}
        test = unbalanced.DfUnbalancedDataset(test_data)
        super(NslKdd, self).__init__(train=train, valid=valid, test=test)
        self.train.maps = maps
        self.mapper = train.mapper.value
        # normalizer
        stats = self.train.get_stats()
        no_norm = [key for key, value in stats.max.items()
                   if value < 5.0]
        for var in no_norm:
            stats.mean[var] = 0.0
            stats.stddev[var] = 1.0
        self.normalizer = tdld.base.Normalizer(stats)
        # one hot
        if one_hot and multi_class and len(self.train.labels) > 2:
            self.mapper.y.append(tdld.base.OneHotMap(len(self.train.labels)))
