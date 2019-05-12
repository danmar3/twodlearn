import unittest
import numpy as np
import twodlearn as tdl
import twodlearn.core.array


class ArrayTests(unittest.TestCase):
    def test_np2dmesh(self):
        data = tdl.core.array.Np2dMesh([1, 2, 3, 4],
                                       [4, 5, 6])
        t1 = data.asmesh[0] * data.asmesh[1]
        t2 = data.vect2mesh(data.asarray[:, 0] * data.asarray[:, 1])
        np.testing.assert_array_equal(t1, t2)


if __name__ == "__main__":
    unittest.main()
