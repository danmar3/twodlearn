#   ***********************************************************************
#
#   This file downloads the datasets from:
#       http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/DeepVsShallowComparisonICML2007
#
#   Wrote by: Daniel L. Marino (marinodl@vcu.edu)
#    Modern Heuristics Research Group (MHRG)
#    Virginia Commonwealth University (VCU), Richmond, VA
#    http://www.people.vcu.edu/~mmanic/
#
#   ***********************************************************************


import os
import gzip
import zipfile
import numpy as np
from twodlearn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.contrib.learn.python.learn.datasets import mnist

SRC = {
    'base': 'http://www.iro.umontreal.ca/~lisa/icml2007data/mnist.zip',
    'rot': 'http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip',
    'bg-rand': 'http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_random.zip',
    'bg-img': 'http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_images.zip',
    'bg-img-rot': 'http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_back_image_new.zip',
    'rect': 'http://www.iro.umontreal.ca/~lisa/icml2007data/rectangles.zip',
    'rect-img': 'http://www.iro.umontreal.ca/~lisa/icml2007data/rectangles_images.zip'
}

FILENAME_TRAIN = {
    'base': 'mnist_train.amat',
    'rot': 'mnist_all_rotation_normalized_float_train_valid.amat',
    'bg-rand': 'mnist_background_random_train.amat',
    'bg-img': 'mnist_background_images_train.amat',
    'bg-img-rot': 'mnist_all_background_images_rotation_normalized_train_valid.amat',
    'rect': 'rectangles_train.amat',
    'rect-img': 'rectangles_im_train.amat'
}

FILENAME_TEST = {
    'base': 'mnist_test.amat',
    'rot': 'mnist_all_rotation_normalized_float_test.amat',
    'bg-rand': 'mnist_background_random_test.amat',
    'bg-img': 'mnist_background_images_test.amat',
    'bg-img-rot': 'mnist_all_background_images_rotation_normalized_test.amat',
    'rect': 'rectangles_test.amat',
    'rect-img': 'rectangles_im_test.amat'
}

DEFAULT_SOURCE_URL = SRC['base']


class DataSet(mnist.DataSet):
    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32,
                 reshape=True,
                 seed=None):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            if reshape:
                assert images.shape[3] == 1
                images = images.reshape(images.shape[0],
                                        images.shape[1] * images.shape[2])
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0


def read_datafile(data_file, one_hot=False, num_classes=10):
    mat = np.loadtxt(data_file, np.float32)
    images = mat[:, :-1]
    labels = mat[:, -1]
    labels = labels.astype(np.uint8)

    num_images = images.shape[0]
    rows = 28
    cols = 28
    assert rows * cols == images.shape[1],\
        "loaded images are not 28*28"
    images = images.reshape(num_images, rows, cols, 1)

    if one_hot:
        labels = mnist.dense_to_one_hot(labels, num_classes)
    return images, labels


def load(train_dir,
         fake_data=False,
         one_hot=False,
         reshape=True,
         validation_size=5000,
         seed=None,
         dataset_name='base'):

    if dataset_name in SRC:
        source_url = SRC[dataset_name]
    else:
        raise ValueError("dataset_type must be one of the following: {}"
                         "".format(SRC.keys()))

    if fake_data:

        def fake():
            return mnist.DataSet(
                [], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

        train = fake()
        validation = fake()
        test = fake()
        return base.Datasets(train=train, valid=validation, test=test)

    if not source_url:  # empty string check
        source_url = DEFAULT_SOURCE_URL

    compressed_file = source_url.split("/")[-1]
    print('loading file: {}'.format(compressed_file))
    local_file = base.maybe_download(compressed_file, train_dir,
                                     source_url)

    # decompress
    data_dir = os.path.join(train_dir, dataset_name)
    with zipfile.ZipFile(local_file, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    train_path = os.path.join(data_dir, FILENAME_TRAIN[dataset_name])
    train_images, train_labels = read_datafile(train_path, one_hot=one_hot)

    test_path = os.path.join(data_dir, FILENAME_TEST[dataset_name])
    test_images, test_labels = read_datafile(test_path, one_hot=one_hot)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
            .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    options = dict(reshape=reshape, seed=seed)

    train = DataSet(train_images, train_labels, **options)
    validation = DataSet(validation_images, validation_labels, **options)
    test = DataSet(test_images, test_labels, **options)

    return base.Datasets(train=train, validation=validation, test=test)


class MnistDataset(base.Datasets):
    src = 'http://yann.lecun.com/exdb/mnist/'

    filename = {'train_x': 'train-images-idx3-ubyte.gz',
                'train_y': 'train-labels-idx1-ubyte.gz',
                'test_x': 't10k-images-idx3-ubyte.gz',
                'test_y': 't10k-labels-idx1-ubyte.gz'}

    @staticmethod
    def _load_images(work_directory, filename):
        filepath = os.path.join(work_directory, filename)
        print('loading {}'.format(filepath))
        with gzip.open(filepath) as file:
            uint32 = np.dtype(np.uint32).newbyteorder('>')
            magic = np.frombuffer(file.read(4), dtype=uint32)[0]
            if magic != 2051:
                raise ValueError('Invalid magic number %d in MNIST images '
                                 'file: %s'.format(magic, filepath))
            num_images = np.frombuffer(file.read(4), dtype=uint32)[0]
            rows = np.frombuffer(file.read(4), dtype=uint32)[0]
            cols = np.frombuffer(file.read(4), dtype=uint32)[0]
            data = np.frombuffer(file.read(rows * cols * num_images),
                                 dtype=np.uint8)
            data = data.reshape(num_images, rows, cols, 1)
        return data

    @staticmethod
    def _load_labels(work_directory, filename):
        filepath = os.path.join(work_directory, filename)
        with gzip.open(filepath) as file:
            uint32 = np.dtype(np.uint32).newbyteorder('>')
            magic = np.frombuffer(file.read(4), dtype=uint32)[0]
            if magic != 2049:
                raise ValueError('Invalid magic number %d in MNIST labels '
                                 'file: %s'.format(magic, filepath))
            num_items = np.frombuffer(file.read(4), dtype=uint32)[0]
            data = np.frombuffer(file.read(num_items),
                                 dtype=np.uint8)
            data = data.reshape(num_items, 1)
        return data

    def _load_raw(self, work_directory, one_hot, dtype, reshape):
        train_x = self._load_images(work_directory, self.filename['train_x'])
        train_y = self._load_labels(work_directory, self.filename['train_y'])
        test_x = self._load_images(work_directory, self.filename['test_x'])
        test_y = self._load_labels(work_directory, self.filename['test_y'])

        if reshape:
            train_x = train_x.reshape(train_x.shape[0], -1)
            test_x = test_x.reshape(test_x.shape[0], -1)
        if dtype == np.float32:
            train_x = train_x.astype(np.float32) / 255.0
            test_x = test_x.astype(np.float32) / 255.0
        if one_hot:
            train_y = base.dense_to_one_hot(train_y)
            test_y = base.dense_to_one_hot(test_y)
        return (train_x, train_y), (test_x, test_y)

    def __init__(self, work_directory, one_hot=True, dtype=np.float32,
                 reshape=True, valid_size=0.1):
        for name, filename in self.filename.items():
            base.maybe_download(filename=filename,
                                work_directory=work_directory,
                                source_url=self.src + filename)
        train, test = self._load_raw(work_directory, one_hot, dtype, reshape)

        if not 0 <= valid_size <= 1.0:
            raise ValueError('validation size should be between 0.0 and 1.0. '
                             'Received:{}'.format(valid_size))
        valid_size = int(valid_size * train[0].shape[0])
        valid = (train[0][:valid_size],
                 train[1][:valid_size])
        train = (train[0][valid_size:],
                 train[1][valid_size:])
        super(MnistDataset, self).__init__(
            train=base.Dataset(train[0], train[1]),
            valid=base.Dataset(valid[0], valid[1]),
            test=base.Dataset(test[0], test[1]))
