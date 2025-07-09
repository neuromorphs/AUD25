from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.signal import (
    butter,
    correlate,
    sosfiltfilt,
    sosfreqz,
    find_peaks,
    hilbert,
    medfilt,
    resample,
)

from sklearn.decomposition import FastICA
from sklearn.mixture import GaussianMixture


def teeger(x: NDArray) -> NDArray:
    """Compute the Teeger energy operator over a waveform.
    The result is a signal that is the product of the instanenous freqeuncy
    and the amplitude of the sinusoid.  We use this to tell us which tone
    is which.
    """
    # https://ieeexplore.ieee.org/document/8553703/
    return x[1:-1] ** 2 - x[:-2] * x[2:]


# Helper function
def find_current_segment(present: NDArray, loc: int, skip_ahead: int = 100) -> NDArray:
    """Given a Boolean truth array and the location of a true value, find the
    start and ending locations of the true segment.

    skip_ahead is a small integer saying how many samples to skip ahead, in case
    the initial location is a bit too early.
    """
    present = np.asarray(present)
    cur_loc = loc + skip_ahead
    assert present[cur_loc]
    assert cur_loc > 0
    assert cur_loc < present.shape[0]
    start_results = np.where(present[cur_loc::-1] == 0)
    start_loc = cur_loc - start_results[0][0]
    end_results = np.where(present[cur_loc::] == 0)
    end_loc = cur_loc + end_results[0][0]
    return int(start_loc), int(end_loc)


def label_tone_blips(freqm: NDArray) -> NDArray:
    """Label each sample of the audio waveform whether it is 0 (no sound),
    standard tone (1) or deviant tone (2).  The standard is found over the deviant
    because there are more standard tones.
    """
    if freqm.ndim == 1:
        freqm = freqm.reshape(-1, 1)
    if freqm.shape[1] == 1:
        freqm2 = np.concatenate((freqm, freqm, freqm), axis=1)

    # Fit the Teeger results with 3 clusters so we can identify each tone blip
    gm = GaussianMixture(3, covariance_type="diag")
    gm.fit(freqm2)
    labels = gm.predict(freqm2)

    # Find the GMM indices corresponding to the clusters that we want.  Silence
    # will be most common, and deviants least common.
    _, counts = np.unique(labels, return_counts=True)
    zero_label = int(np.argmax(counts))
    deviant_label = int(np.argmin(counts))
    standard_label = set(range(3)).difference({zero_label, deviant_label}).pop()

    results = np.zeros(freqm.shape[0], int)
    results[labels == standard_label] = 1
    results[labels == deviant_label] = 2
    return results


def find_tone_examples(
    audio_waveform: NDArray,
    debug_plots: bool = True,
) -> Tuple[NDArray, NDArray]:
    """Given an audio waveform, find the prototypical
    standard and deviant tone arrays.
    """
    freq = teeger(audio_waveform[0, :])
    freqm = medfilt(freq, 7)  # Smooth freqs to remove glitches

    labels = label_tone_blips(freqm)
    standard_label = 1
    deviant_label = 2

    # Smoth these labels because we often see glitches (because the Teeger
    # operator is sensitive to transitions.)
    smooth_labels = medfilt(labels, 7)
    first_standard_loc = np.where(smooth_labels == standard_label)[0][0]
    first_deviant_loc = np.where(smooth_labels == deviant_label)[0][0]

    # Figure out where the tones are present, using the Hilbert transform to
    # find the envelope
    waveform_envelope = np.abs(hilbert(audio_waveform[0, :]))
    tone_present = medfilt(
        (waveform_envelope > np.max(waveform_envelope) / 2).astype(int), 7
    )

    if debug_plots:
        plt.subplot(2, 1, 1)
        plt.plot(labels, label="GMM Labels")
        plt.plot(tone_present, label="Tone Present")
        plt.plot(waveform_envelope, label="Envelope")
        plt.axvline(first_standard_loc, color="r")
        plt.axvline(first_deviant_loc, color="g")
        plt.xlim([first_standard_loc - 1000, first_standard_loc + 1000])
        plt.ylabel("Standard")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(labels, label="GMM Labels")
        plt.plot(tone_present, label="Tone Present")
        plt.plot(waveform_envelope, label="Envelope")
        plt.axvline(first_standard_loc, color="r")
        plt.axvline(first_deviant_loc, color="k")
        plt.xlim([first_deviant_loc - 1000, first_deviant_loc + 1000])
        plt.ylabel("Deviant")
        plt.legend()

    # OK, now we know where the two tones start, find their complete extant.
    standard_locs = find_current_segment(tone_present, first_standard_loc)
    deviant_locs = find_current_segment(tone_present, first_deviant_loc)

    standard_tone = audio_waveform[0, standard_locs[0] : standard_locs[1]]
    deviant_tone = audio_waveform[0, deviant_locs[0] : deviant_locs[1]]

    return standard_tone, deviant_tone


def find_all_tone_starts(
    audio_waveform: NDArray,
    pulse_template: NDArray,
    min_tone_separation=10000,
    sampling_rate: int = 25000,
) -> NDArray:
    # Perform cross-correlation to find the start of each pulse in the overall
    # audio waveform. Return the answer in sample counts.
    correlation = correlate(audio_waveform, pulse_template, mode="valid")
    hc = hilbert(correlation)
    hcm = medfilt(np.abs(hc), 15)
    plt.plot(np.arange(hcm.shape[0]) / sampling_rate, hcm)
    plt.xlabel("Time (s)")
    plt.ylabel("Hilbert of Correlation Value")
    plt.title("Hilbert of Cross-correlation of waveform and pulse template")
    peak_indices, _ = find_peaks(
        hcm, height=np.max(hcm) * 0.5, distance=min_tone_separation
    )
    return peak_indices


def butter_bandpass(lowcut: float, highcut: float, fs: float = 25000, order=2):
    b, a = butter(order, [lowcut, highcut], fs, btype="band")
    return b, a


def filter_eeg(
    data: NDArray,
    lowcut_freq: float = 500,  # Hz
    highcut_freq: float = 1500,  # Hz
    sampling_rate: int = 25000,
    order: int = 8,
    axis: int = -1,
    debug_spectrum: bool = False
) -> NDArray:
    # Example usage: Bandpass filter the eeg waveform .z
    sos = butter(order, [lowcut_freq, highcut_freq], fs=sampling_rate,
                 output='sos', btype='band')
    if debug_spectrum:
      w, h = sosfreqz(sos, fs=sampling_rate, worN=2000)
      plt.plot(w, 20*np.log10(abs(h)), label="order = %d" % order)
      plt.grid('on')
    y = sosfiltfilt(sos, data, axis=axis)
    return y

def model_with_ica(
    filtered_eeg: NDArray, # Shape n_samples x n_features
    sampling_rate: int = 25000, n_components: int = 10
) -> Tuple[FastICA, NDArray]:
    assert filtered_eeg.ndim == 2
    assert filtered_eeg.shape[0] > filtered_eeg.shape[1]
    ica = FastICA(n_components=n_components, random_state=0)
    ica_components = ica.fit_transform(filtered_eeg)

    j = [
        i + ica_components[: 10 * sampling_rate, i]
        for i in range(ica_components.shape[1])
    ]

    components = np.concatenate(
        [
            5 * i + ica_components[: 10 * sampling_rate, i : i + 1]
            for i in range(ica_components.shape[1])
        ],
        axis=1,
    )

    plt.figure(figsize=(10, 8))
    plt.plot(np.arange(components.shape[0]) / sampling_rate, components)
    for i in range(components.shape[1]):
        plt.text(0, 5 * i, f"IC #{i}")
        plt.xlabel("Time (s)")
    return ica, components


def filter_ica_channels(
    ica: FastICA,
    ica_components: NDArray,
    bad_channels: List[int] = [2, 5, 6],
) -> NDArray:
    ica_components2 = ica_components.copy()  # Num_times x num_factors
    for i in bad_channels:
        ica_components2[:, i] = 0

    cleaned_eeg = ica.inverse_transform(ica_components2)
    return cleaned_eeg

def accumulate_erp(eeg_data: NDArray,
                   locs: NDArray, # In seconds
                   sampling_rate: float = 25000,
                   num_samples: int = 12500,
                   pre_samples: int = 0,
                   remove_baseline: bool = False) -> NDArray:
  """Average the EEG response for num_samples samples starting at each
  location in the locs argument.  The locations are in seconds, so the
  sampling rate must be correct.
  """
  num_channels, num_times = eeg_data.shape
  assert num_times > num_channels
  erp = 0
  count = 0
  for loc in locs:
    loc = int(loc*sampling_rate - pre_samples)
    if loc + num_samples < eeg_data.shape[1]:
      eeg = eeg_data[:, loc:loc+num_samples]
      if remove_baseline and num_samples > 0:
        eeg -= np.mean(eeg[:, :pre_samples], axis=1, keepdims=True)
      erp += eeg
      count += 1
  return erp/count


def downsample_eeg(eeg_data: NDArray,
                 sampling_rate: float,
                 factor: int) -> NDArray:
  """Downsample the EEG data (num_channels x num_times) by an integer factor.
  Must do the anti-aliasing (low pass filter) before calling this routine.
  """
  num_channels, num_samples = eeg_data.shape
  new_sampling_rate = sampling_rate / factor
  num_new_samples = int(num_samples * new_sampling_rate / sampling_rate)

  # Resample each channel independently
  resampled_eeg_data = np.zeros((eeg_data.shape[0], num_new_samples))
  for i in range(eeg_data.shape[0]):
    resampled_eeg_data[i, :] = resample(eeg_data[i, :], num_new_samples)
  return resampled_eeg_data
