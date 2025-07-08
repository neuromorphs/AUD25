from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.signal import (
    butter,
    correlate,
    filtfilt,
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


present = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
find_current_segment(present, 4, 1)
# Should be indices 2 and 8


def find_tone_examples(
    audio_waveform: NDArray,
    debug_plots: bool = True,
) -> Tuple[NDArray, NDArray]:
    """Given an audio waveform, find the prototypical
    standard and deviant tone arrays.
    """
    freq = teeger(audio_waveform[0, :])
    freqm = medfilt(freq, 7)

    # Create 3d data appropriate for GMM code
    freqm2 = np.concatenate(
        (freqm.reshape(-1, 1), freqm.reshape(-1, 1), freqm.reshape(-1, 1)), axis=1
    )
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
