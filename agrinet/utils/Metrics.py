import numpy as np
import pandas as pd

REALITY_THRESHOLD = 0.5


class CGANMetrics:
    """Batchwise metrics for a Conditional GAN model testing"""

    results = pd.DataFrame(columns=["MMD", "PSNR"])
    results.index.name = "Batch"
    save_path = None

    def __init__(self, save_path=None):
        self.save_path = save_path

    def update(self, disc_out, gen_out, truth_in, truth_out):

        self.results.loc[len(self.results.index)] = [
            self.mmd(gen_out, truth_out),
            self.psnr(gen_out, truth_out),
        ]

    def psnr(self, x, y):
        """Compute the Peak Signal to Noise Ratio between two images"""
        x = x.numpy() if hasattr(x, "numpy") else x
        y = y.numpy() if hasattr(y, "numpy") else y
        mse = np.mean((x - y) ** 2)
        return 20 * np.log10(255) - 10 * np.log10(mse)

    def mmd(self, x, y):
        """Compute the Maximum Mean Discrepancy between two sets of samples"""
        x = x.numpy() if hasattr(x, "numpy") else x
        y = y.numpy() if hasattr(y, "numpy") else y
        x_flat = np.reshape(x, (x.shape[0], -1))
        y_flat = np.reshape(y, (y.shape[0], -1))

        # Compute squared norms
        x_sq_norm = np.sum(x_flat**2, axis=1)
        y_sq_norm = np.sum(y_flat**2, axis=1)

        # Compute pairwise distances
        dist_x = x_sq_norm + x_sq_norm[:, None] - 2 * np.dot(x_flat, x_flat.T)
        dist_y = y_sq_norm + y_sq_norm[:, None] - 2 * np.dot(y_flat, y_flat.T)
        dist_xy = x_sq_norm + y_sq_norm[:, None] - 2 * np.dot(x_flat, y_flat.T)

        # Median heuristic for bandwidth
        sigma2 = np.median(np.concatenate([dist_x, dist_y, dist_xy])) / 2
        mmd_value = (
            np.mean(np.exp(-dist_x / sigma2))
            + np.mean(np.exp(-dist_y / sigma2))
            - 2 * np.mean(np.exp(-dist_xy / sigma2))
        )

        return mmd_value

    def get_metric(self, metric):
        return self.results[metric].mean()

    def __str__(self):
        return self.results.mean().to_string()

    def __del__(self):
        if self.save_path is not None:
            import time

            today = time.strftime("%Y-%m-%d-%H-%M-%S")

            import os

            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

            self.results.to_csv(f"{self.save_path}/{today}.csv")
