import unittest
import numpy as np
import tensorflow as tf
import twodlearn as tdl


class CommonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.InteractiveSession()

    def test_positive_variable(self):
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
