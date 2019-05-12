from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest
from twodlearn.templates.mnist.supervised import (
    MnistSupervised, SimpleMnistSupervised, MnistCustomMlp,
    MnistTestMlp, Cifar10Supervised, Cifar10TestMlp)

TESTS_PATH = os.path.dirname(os.path.abspath(__file__))
TMP_PATH = os.path.join(TESTS_PATH, 'tmp/')


class MnistTest(unittest.TestCase):
    def test_default_mlp(self):
        main = SimpleMnistSupervised(tmp_path=TMP_PATH)
        main.run_training()

    def test_mlp(self):
        options = {'model/class': MnistCustomMlp}
        main = MnistSupervised(options=options, tmp_path=TMP_PATH)
        main.run_training()


if __name__ == "__main__":
    unittest.main()
    # options = {'model/class': MnistTestMlp,
    #            'optim/train/max_steps': 5000}
    # main = MnistSupervised(options=options, tmp_path=TMP_PATH)
    # main.run_training()
    # -
    # options = {'model/class': Cifar10TestMlp,
    #            'optim/train/max_steps': 30000}
    # main = Cifar10Supervised(options=options, tmp_path=TMP_PATH)
    # main.run_training()
