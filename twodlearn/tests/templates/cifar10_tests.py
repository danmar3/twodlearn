import os
import time
import unittest
import twodlearn as tdl
import twodlearn.datasets.cifar10
from twodlearn.templates.supervised import (
    LinearClassifier, MlpClassifier, AlexNetClassifier)

TESTS_PATH = os.path.dirname(os.path.abspath(__file__))
TMP_PATH = os.path.join(TESTS_PATH, 'cifar10_data/')


class OptimTests(unittest.TestCase):
    def test_linear(self):
        mnist = tdl.datasets.cifar10.Cifar10(work_directory=TMP_PATH,
                                             reshape=True)
        model = LinearClassifier(n_inputs=32*32*3, n_classes=10,
                                 logger_path=os.path.join(TMP_PATH, 'loggers'),
                                 options={'train/optim/max_steps': 1000})
        t1 = time.time()
        model.fit(mnist)
        t2 = time.time()
        print('training took:', t2-t1)

    def test_mlp(self):
        mnist = tdl.datasets.cifar10.Cifar10(work_directory=TMP_PATH,
                                             reshape=True)
        model = MlpClassifier(n_inputs=32*32*3, n_classes=10,
                              n_hidden=[500],
                              logger_path=os.path.join(TMP_PATH, 'loggers'),
                              options={'train/optim/learning_rate': 0.002,
                                       'train/optim/max_steps': 1000})
        t1 = time.time()
        model.fit(mnist)
        t2 = time.time()
        print('training took:', t2-t1)

    def test_convnet(self):
        mnist = tdl.datasets.cifar10.Cifar10(work_directory=TMP_PATH,
                                             reshape=False)
        model = AlexNetClassifier(
            input_shape=[32, 32, 3],
            n_classes=10,
            n_filters=[32, 64],
            filter_sizes=[[5, 5], [5, 5]],
            pool_sizes=[[2, 2], [2, 2]],
            n_hidden=[1024],
            logger_path=os.path.join(TMP_PATH, 'loggers'),
            options={'train/optim/learning_rate': 0.001,
                     'train/optim/max_steps': 1000})
        t1 = time.time()
        model.fit(mnist)
        t2 = time.time()
        print('training took:', t2-t1)


if __name__ == "__main__":
    unittest.main()
