import os
import time
import unittest
import twodlearn as tdl
import twodlearn.datasets.mnist
from twodlearn.templates.supervised import (
    LinearClassifier, MlpClassifier, AlexNetClassifier)

TESTS_PATH = os.path.dirname(os.path.abspath(__file__))
TMP_PATH = os.path.join(TESTS_PATH, 'tmp/')


class OptimTests(unittest.TestCase):
    def test_linear(self):
        mnist = tdl.datasets.mnist.MnistDataset(work_directory=TMP_PATH)
        model = LinearClassifier(n_inputs=28*28, n_classes=10,
                                 logger_path='tmp/loggers',
                                 options={'train/optim/max_steps': 1000})
        t1 = time.time()
        model.fit(mnist)
        t2 = time.time()
        print('training took:', t2-t1)

    def test_mlp(self):
        mnist = tdl.datasets.mnist.MnistDataset(work_directory=TMP_PATH)
        model = MlpClassifier(n_inputs=28*28, n_classes=10,
                              n_hidden=[500],
                              logger_path='tmp/loggers',
                              options={'train/optim/learning_rate': 0.002,
                                       'train/optim/max_steps': 1000})
        t1 = time.time()
        model.fit(mnist)
        t2 = time.time()
        print('training took:', t2-t1)

    def test_convnet(self):
        mnist = tdl.datasets.mnist.MnistDataset(work_directory=TMP_PATH,
                                                reshape=False)
        model = AlexNetClassifier(
            input_shape=[28, 28, 1],
            n_classes=10,
            n_filters=[32, 64],
            filter_sizes=[[5, 5], [5, 5]],
            pool_sizes=[[2, 2], [2, 2]],
            n_hidden=[1024],
            logger_path='tmp/loggers',
            options={'train/optim/learning_rate': 0.001,
                     'train/optim/max_steps': 1000})
        t1 = time.time()
        model.fit(mnist)
        t2 = time.time()
        print('training took:', t2-t1)


if __name__ == "__main__":
    unittest.main()
