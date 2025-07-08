# import matplotlib.pyplot as plt

import numpy as np
import oddball_analysis as oddball
from absl.testing import absltest


class OddballTests(absltest.TestCase):
    def test_teeger(self):
        # Test that Teeger operator is linear in power
        x = np.sin(2 * np.pi * np.arange(100) / 100)
        y = np.mean(oddball.teeger(x))
        y2 = np.mean(oddball.teeger(2 * x))
        self.assertAlmostEqual(4 * y, y2)

        x2 = np.sin(2 * np.pi * 2 * np.arange(100) / 100)
        y2 = np.mean(oddball.teeger(x2))
        self.assertAlmostEqual(4 * y, y2, delta=1e-4)


if __name__ == "__main__":
    absltest.main()
