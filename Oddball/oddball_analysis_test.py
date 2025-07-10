import matplotlib.pyplot as plt

import numpy as np
import oddball_analysis as oddball
from absl.testing import absltest
from sklearn.decomposition import FastICA


class OddballTests(absltest.TestCase):
    def test_teeger(self):
        # Test that Teeger operator is linear in power
        x = np.sin(2 * np.pi * np.arange(100) / 100)
        y = np.mean(oddball.teeger(x))
        y2 = np.mean(oddball.teeger(2 * x))  # Double in amplitude
        self.assertAlmostEqual(4 * y, y2)

        # Test that Teeger operator is quadratic in frequency
        x2 = np.sin(2 * np.pi * 2 * np.arange(100) / 100) # Double in frequency
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
        x = np.zeros((1, 16384))
        fs = 16000
        lowcut = 500
        highcut = 2000
        order = 4
        # Put impulse in the middle since we are using filtfilt
        x[0, x.shape[1]//2] = 1
        plt.clf()
        y = oddball.filter_eeg(x, lowcut, highcut, fs, axis=1,
                               order=order, debug_spectrum=True)
        plt.savefig("test_bandpass_spectrum.png")
        spectrum = np.fft.fftshift(20 * np.log10(np.abs(np.fft.fft(y[0, :]))))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(spectrum), 1 / fs))
        print("freqs", freqs.shape, spectrum.shape)
        plt.clf()
        plt.semilogx(freqs, spectrum)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.xlim([100, 8000])
        plt.axvline(lowcut, c='r', ls=':')
        plt.axvline(highcut, c='r', ls=':')
        plt.grid('on')
        plt.savefig("test_bandpass_impulse_spectrum.png")

        # Check low-frequency corner
        fftbin = np.argmin(np.abs(freqs - lowcut))
        self.assertAlmostEqual(spectrum[fftbin], -6.02, delta=.01)

        # Check high-frequency corner (One octave above)
        fftbin = np.argmin(np.abs(freqs - highcut))
        self.assertAlmostEqual(spectrum[fftbin], -6.02, delta=.01) # Double in frequency

        # Check the middle of the passbamd
        fftbin = np.argmin(np.abs(freqs - (lowcut + highcut)/2))
        self.assertAlmostEqual(spectrum[fftbin], 0, delta=.01)

        # Check one octave below the low cutoff
        fftbin = np.argmin(np.abs(freqs - lowcut/2))
        self.assertLess(spectrum[fftbin], -60)

        # Check one octave above the high cutoff
        fftbin = np.argmin(np.abs(freqs - highcut*2))
        self.assertLess(spectrum[fftbin], -60)

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
