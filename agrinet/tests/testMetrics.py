import sys
import os

# fixes "ModuleNotFoundError: No module named 'utils'"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# flake8: noqa
import unittest
import numpy as np
import pandas as pd
from utils.Metrics import CGANMetrics


class TestCGANMetrics(unittest.TestCase):
    def setUp(self):
        self.cgan_metrics = CGANMetrics()

    def tearDown(self):
        del self.cgan_metrics

    def test_psnr(self):
        x = np.random.randint(0, 255, size=(256, 256, 3)).astype(np.float32)
        y = np.random.randint(0, 255, size=(256, 256, 3)).astype(np.float32)
        psnr_score = self.cgan_metrics.psnr(x, y)
        self.assertIsInstance(psnr_score, float)

    def test_mmd(self):
        x = np.random.randn(16, 256, 256, 3).astype(np.float32)
        y = np.random.randn(16, 256, 256, 3).astype(np.float32)
        mmd_value = self.cgan_metrics.mmd(x, y)
        self.assertIsInstance(mmd_value, float)

    def test_update(self):
        disc_out = np.random.randn(16, 256, 256, 3).astype(np.float32)
        gen_out = np.random.randn(16, 256, 256, 3).astype(np.float32)
        truth_in = np.random.randn(16, 256, 256, 3).astype(np.float32)
        truth_out = np.random.randn(16, 256, 256, 3).astype(np.float32)
        self.cgan_metrics.update(disc_out, gen_out, truth_in, truth_out)
        self.assertEqual(len(self.cgan_metrics.results), 1)

    def test_get_metric(self):
        self.cgan_metrics.results = pd.DataFrame(
            {"MMD": [0.5, 0.6, 0.7], "PSNR": [20, 25, 30]}
        )
        mmd_mean = self.cgan_metrics.get_metric("MMD")
        psnr_mean = self.cgan_metrics.get_metric("PSNR")
        self.assertAlmostEqual(mmd_mean, 0.6, places=2)
        self.assertAlmostEqual(psnr_mean, 25, places=2)


if __name__ == "__main__":
    unittest.main()
