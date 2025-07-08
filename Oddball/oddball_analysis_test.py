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

    def test_find_segment(self):
        present = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        ans = oddball.find_current_segment(present, 4, 1)
        self.assertSequenceEqual(ans, (2, 8))

    def test_label_tone_blips(self):
        a = np.asarray(
            [
                0,
                0,
                0,
                0,
                4,
                4,
                4,
                0,
                1,
                1,
            ]
        )
        res = oddball.label_tone_blips(a)
        self.assertSequenceEqual(list(res), [0, 0, 0, 0, 1, 1, 1, 0, 2, 2])


if __name__ == "__main__":
    absltest.main()
