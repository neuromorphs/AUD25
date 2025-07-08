# import matplotlib.pyplot as plt

import numpy as np
import oddball_analysis as oddball
from absl.testing import absltest
from sklearn.decomposition import FastICA


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

    def test_bandpass(self):
        x = np.zeros((1, 1024))
        x[0] = 1
        y = oddball.filter_eeg(x, 1024)
        # Just make sure it runs.. need to frequency response.

    def test_ica(self):
        data = np.random.rand(1000, 100)
        ica, ica_components = oddball.model_with_ica(data)
        self.assertIsInstance(ica, FastICA)
        self.assertIsInstance(ica_components, np.ndarray)
        self.assertEqual(ica_components.shape[1], 10)

        results = oddball.filter_ica_channels(
            ica, ica_components, bad_channels=[2, 5, 6]
        )
        self.assertEqual(data.shape, results.shape)
        # Just make sure it runs, no functional test yet.


if __name__ == "__main__":
    absltest.main()
