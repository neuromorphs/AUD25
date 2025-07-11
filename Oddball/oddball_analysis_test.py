import matplotlib.pyplot as plt

import numpy as np
import oddball_analysis as oddball
from absl.testing import absltest
from numpy.typing import NDArray
from sklearn.decomposition import FastICA


class OddballTests(absltest.TestCase):
    def test_find_most_common(self):
        res = oddball.find_most_common([1, 2, 3, 2, 1, 3, 2, 1, 2, 4, 4, 4, 4])
        self.assertEqual(res, 2)

    def test_find_segment(self):
        my_list = [0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 1, 1, 3, 1, 1, 0, 0]

        segment = oddball.find_desired_segment([i > 0 for i in my_list], 3, my_list, 1)
        self.assertEqual(segment, (2, 4))

        segment = oddball.find_desired_segment([i > 0 for i in my_list], 3, my_list, 2)
        self.assertEqual(segment, (8, 11))

        segment = oddball.find_desired_segment([i > 0 for i in my_list], 4, my_list, 1)
        self.assertEqual(segment, (15, 19))

        segment = oddball.find_desired_segment([i > 0 for i in my_list], 6, my_list, 1)
        self.assertEqual(segment, None)

    def test_label_blips(self):
        data = [0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 3, 3]
        results = oddball.label_tone_blips(np.asarray(data))
        # Lots of zeros.  5 is the next most commmon so it is label 1 (standard) and 3
        # is the least common so it is label 2 (oddball)
        self.assertSameStructure(
            results.tolist(), [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2]
        )

    def test_synthesize_oddball(self):
        num_trials = 100
        standard_freq = 250
        deviant_freq = 500
        audio, trial_sequence = oddball.create_oddball_sequence(
            n_trials=100,
            gap_duration=0.1,
            standard_freq=standard_freq,
            oddball_freq=deviant_freq,
        )
        self.assertEqual(audio.ndim, 1)
        self.assertEqual(len(trial_sequence), num_trials)

        # Make sure we get both standards and deviants
        self.assertGreater(np.sum(trial_sequence == False), 0)  # Standards
        self.assertGreater(np.sum(trial_sequence == True), 0)  # Deviants
        self.assertLess(np.sum(trial_sequence == True), np.sum(trial_sequence == False))

        sampling_rate = 25000
        standard_tone, deviant_tone = oddball.find_tone_examples(
            audio.reshape(1, -1),
        )
        oddball.plot_audio_waveforms(
            standard_tone.reshape(1, -1), deviant_tone.reshape(1, -1), sampling_rate
        )
        oddball.save_fig(plt.gcf(), ".", "test_recogniez_oddball.png")

        def freq(tone: NDArray) -> float:
            count = np.sum(np.logical_and(tone[1:] > 0, tone[:-1] < 0))
            return sampling_rate / (tone.shape[0] / count)

        self.assertAlmostEqual(freq(standard_tone), standard_freq, delta=20)
        self.assertAlmostEqual(freq(deviant_tone), deviant_freq, delta=20)

    def test_teeger(self):
        # Test that Teeger operator is linear in power
        x = np.sin(2 * np.pi * np.arange(100) / 100)
        y = np.mean(oddball.teeger(x))
        y2 = np.mean(oddball.teeger(2 * x))  # Double in amplitude
        self.assertAlmostEqual(4 * y, y2)

        # Test that Teeger operator is quadratic in frequency
        x2 = np.sin(2 * np.pi * 2 * np.arange(100) / 100)  # Double in frequency
        y2 = np.mean(oddball.teeger(x2))
        self.assertAlmostEqual(4 * y, y2, delta=1e-4)

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
        x[0, x.shape[1] // 2] = 1
        plt.clf()
        y = oddball.filter_eeg(
            x, lowcut, highcut, fs, axis=1, order=order, debug_spectrum=True
        )
        plt.savefig("test_bandpass_spectrum.png")
        spectrum = np.fft.fftshift(20 * np.log10(np.abs(np.fft.fft(y[0, :]))))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(spectrum), 1 / fs))
        print("freqs", freqs.shape, spectrum.shape)
        plt.clf()
        plt.semilogx(freqs, spectrum)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.xlim([100, 8000])
        plt.axvline(lowcut, c="r", ls=":")
        plt.axvline(highcut, c="r", ls=":")
        plt.grid("on")
        plt.savefig("test_bandpass_impulse_spectrum.png")

        # Check low-frequency corner
        fftbin = np.argmin(np.abs(freqs - lowcut))
        self.assertAlmostEqual(spectrum[fftbin], -6.02, delta=0.01)

        # Check high-frequency corner (One octave above)
        fftbin = np.argmin(np.abs(freqs - highcut))
        self.assertAlmostEqual(
            spectrum[fftbin], -6.02, delta=0.01
        )  # Double in frequency

        # Check the middle of the passbamd
        fftbin = np.argmin(np.abs(freqs - (lowcut + highcut) / 2))
        self.assertAlmostEqual(spectrum[fftbin], 0, delta=0.01)

        # Check one octave below the low cutoff
        fftbin = np.argmin(np.abs(freqs - lowcut / 2))
        self.assertLess(spectrum[fftbin], -60)

        # Check one octave above the high cutoff
        fftbin = np.argmin(np.abs(freqs - highcut * 2))
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
