import unittest
import numpy as np
import tensorflow as tf
import twodlearn as tdl


class ConstrainedTest(unittest.TestCase):
    def test_positive_variable(self):
        with tf.Session().as_default():
            test = tdl.constrained.PositiveVariableExp(
                initial_value=lambda:
                tf.exp(tf.truncated_normal_initializer()(shape=[5, 5])))
            test.initializer.run()
            x1 = test.value.eval()

            test.initializer.run()
            x2 = test.value.eval()

            with self.assertRaises(AssertionError):
                assert np.testing.assert_almost_equal(x1, x2)


if __name__ == "__main__":
    unittest.main()
