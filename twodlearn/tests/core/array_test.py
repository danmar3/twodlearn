import unittest
import numpy as np
import tensorflow as tf
import twodlearn as tdl
import twodlearn.core.array


class ArrayTests(unittest.TestCase):
    def test_np2dmesh(self):
        data = tdl.core.array.Np2dMesh([1, 2, 3, 4],
                                       [4, 5, 6])
        t1 = data.asmesh[0] * data.asmesh[1]
        t2 = data.vect2mesh(data.asarray[:, 0] * data.asarray[:, 1])
        np.testing.assert_array_equal(t1, t2)

    def test_reduce_sum_rightmost(self):
        x = tf.ones([2, 3, 4])
        with tf.Session().as_default():
            np.testing.assert_array_equal(
                tdl.core.array.reduce_sum_rightmost(x, ndims=2).eval(),
                np.array([3*4, 3*4]).astype(np.float32))
            np.testing.assert_array_equal(
                tdl.core.array.reduce_sum_rightmost(x, ndims=3).eval(),
                np.array([2*3*4]).astype(np.float32))
            np.testing.assert_array_equal(
                tdl.core.array.reduce_sum_rightmost(x).eval(),
                np.array([3*4, 3*4]).astype(np.float32))

            y1 = tdl.core.array.reduce_sum_rightmost(x, ndims=2)
            assert y1.shape.as_list() == [2]


if __name__ == "__main__":
    unittest.main()
