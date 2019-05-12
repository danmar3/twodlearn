import unittest
import numpy as np
import twodlearn as tdl


class xArrayTest(unittest.TestCase):
    def test_dict_to_xarray(self):
        results = {'IF': {'train': {'norm': 0.1, 'dev': 0.2, 'cr1': 0.3},
                          'valid': {'norm': 1.1, 'dev': 1.2, 'cr1': 1.3},
                          'eval':  {'norm': 2.1, 'dev': 2.2, 'cr1': 2.3}},
                   'IFG': {'train': {'norm': 3.1, 'dev': 3.2, 'cr1': 3.3},
                           'valid': {'norm': 4.1, 'dev': 4.2, 'cr1': 4.3},
                           'eval':  {'norm': 5.1, 'dev': 5.2, 'cr1': 5.3}}
                   }
        data = tdl.utils.xarray.dict_to_xarray(
            results, ['model', 'dataset', 'metrics'])
        assert (data.loc['IF', 'train'].values ==
                np.array([0.1, 0.2, 0.3])).all()
        assert (data.loc['IFG', 'valid'].values ==
                np.array([4.1, 4.2, 4.3])).all()
        with self.assertRaises(ValueError):
            data = tdl.utils.xarray.dict_to_xarray(
                results, ['model', 'dataset'])


if __name__ == "__main__":
    unittest.main()
