import os
import unittest
import tensorflow as tf
import twodlearn as tdl
import twodlearn.templates.unsupervised
from twodlearn.datasets.mnist import MnistDataset

TESTS_PATH = os.path.dirname(os.path.abspath(__file__))
TMP_PATH = os.path.join(TESTS_PATH, 'tmp/')


class TsneTests(unittest.TestCase):
    def test_model(self):
        mnist = MnistDataset(work_directory=TMP_PATH,
                             reshape=True)
        x, labels = mnist.train.next_batch(1000)
        tsne = tdl.templates.unsupervised.Tsne(2, x, target_perplexity=20)
        tsne.perplexity_optimizer.run(1000)
        tsne.run(1000)
        loss = tf.convert_to_tensor(tsne.loss).eval()
        print('loss: {}'.format(loss))
        assert (loss < -5.0),\
            'tsne loss outside the expected value, '\
            'expected: {}, got:{}'.format('<-5.0', loss)


if __name__ == "__main__":
    unittest.main()
