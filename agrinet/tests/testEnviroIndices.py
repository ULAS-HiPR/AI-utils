import os
import sys

# fixes "ModuleNotFoundError: No module named 'utils'"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# flake8: noqa
import unittest

import numpy as np
from utils.EnviroIndices import (
    get_ndvi,
    get_ndwi,
    parse_indices,
    threshold_ndvi,
    threshold_ndwi,
)


class TestEnviroIndices(unittest.TestCase):
    inp, re = np.zeros((256, 256, 3)), np.zeros((256, 256, 3))

    def test_get_ndvi(self):
        ndvi = get_ndvi(self.inp, self.re)

        self.assertIsInstance(ndvi, np.ndarray)
        self.assertEqual(ndvi.shape, (256, 256))

    def test_get_ndwi(self):
        ndwi = get_ndwi(self.inp, self.re)

        self.assertIsInstance(ndwi, np.ndarray)
        self.assertEqual(ndwi.shape, (256, 256))

    def test_parse_indices(self):
        indices = parse_indices(self.inp, self.re)

        self.assertIsInstance(indices, tuple)
        self.assertEqual(len(indices), 2)
        self.assertIsInstance(indices[0], np.ndarray)
        self.assertIsInstance(indices[1], np.ndarray)
        self.assertEqual(indices[0].shape, (256, 256))
        self.assertEqual(indices[1].shape, (256, 256))

    def test_parse_indices_regions(self):
        indices = parse_indices(self.inp, self.re, regions=True)

        self.assertIsInstance(indices, tuple)
        self.assertEqual(len(indices), 4)
        self.assertIsInstance(indices[0], np.ndarray)
        self.assertIsInstance(indices[1], np.ndarray)
        self.assertIsInstance(indices[2], np.ndarray)
        self.assertIsInstance(indices[3], np.ndarray)
        self.assertEqual(indices[0].shape, (256, 256))
        self.assertEqual(indices[1].shape, (256, 256))
        self.assertEqual(indices[2].shape, (256, 256, 3))
        self.assertEqual(indices[3].shape, (256, 256, 3))

    def test_threshold_ndvi(self):
        t_ndvi = threshold_ndvi(get_ndvi(self.inp, self.re))

        self.assertIsInstance(t_ndvi, np.ndarray)
        self.assertEqual(t_ndvi.shape, (256, 256, 3))
        self.assertEqual(t_ndvi.dtype, np.float64)

    def test_threshold_ndwi(self):
        t_ndwi = threshold_ndvi(get_ndwi(self.inp, self.re))

        self.assertIsInstance(t_ndwi, np.ndarray)
        self.assertEqual(t_ndwi.shape, (256, 256, 3))
        self.assertEqual(t_ndwi.dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
