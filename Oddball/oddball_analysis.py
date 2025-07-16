import csv
import glob
import os
import sys
from collections import Counter
from itertools import groupby
from re import I
from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy.io.wavfile as wavfile

from absl import app, flags
from numpy.typing import ArrayLike, NDArray
from scipy.signal import (
    butter,
    correlate,
    find_peaks,
    hilbert,
    medfilt,
    resample,
    sosfiltfilt,
    sosfreqz,
)

from sklearn.decomposition import FastICA
from sklearn.mixture import GaussianMixture

#######################  Synthesize Audio Data   ############################


def create_oddball_sequence(
    fs=25000,  # Sampling frequency (Hz)
    duration=0.05,  # Tone duration (seconds)
    gap_duration=0.5,  # Time between tones (seconds)
    standard_freq=720,  # Frequency of standard tone (Hz)
    oddball_freq=1188,  # Frequency of oddball tone (Hz)
    n_trials=700,  # Total number of trials
    oddball_prob=0.2,  # Probability of oddball tone
) -> Tuple[NDArray, NDArray]:
    # Generate tones
    t = np.arange(0, duration, 1 / fs)  # Time vector
    standard_tone = np.sin(2 * np.pi * standard_freq * t)  # Standard tone
    oddball_tone = np.sin(2 * np.pi * oddball_freq * t)  # Oddball tone

    # Normalize tones
    standard_tone = standard_tone / np.max(np.abs(standard_tone))
    oddball_tone = oddball_tone / max(abs(oddball_tone))

    # Add silence after each tone
    silence = np.zeros(int(gap_duration * fs))
    standard_tone = np.concatenate([standard_tone, silence])
    oddball_tone = np.concatenate([oddball_tone, silence])

    # Generate trial sequence,  1 for oddball, 0 for standard
    trial_sequence = np.random.rand(n_trials) < oddball_prob
    audio = np.concatenate(
        [oddball_tone if trial else standard_tone for trial in trial_sequence]
    )
    # Add silence at the start
    audio = np.concatenate([silence, silence, silence, silence, audio])

    return audio, trial_sequence


#######################  Read Experimental Data  ############################


def read_bv_raw_data(
    data_dir: str, header_file: str
):  # Can't figure out the right type...
    if not header_file:
        files = glob.glob(os.path.join(data_dir, "*.vhdr"))
        if len(files) == 0:
            raise ValueError(f"No .vhdr files found in {data_dir}")
        elif len(files) > 1:
            raise ValueError(f"Multiple .vhdr files found in {data_dir}")
        else:
            if not data_dir.endswith("/"):
                data_dir += "/"
            header_file = files[0].replace(data_dir, "")
    # Specify the path to your BrainVision header file
    vhdr_file = os.path.join(data_dir, header_file)

    # Read the BrainVision data
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)

    # Print some basic information about the data
    print(raw.info)
    print(f"Channel names: {raw.ch_names}")
    print(f"Number of samples: {raw.n_times}")
    return raw


def find_channels(raw, channels: Union[str, List[str]]) -> List[int]:
    if isinstance(channels, str):
        channels = [channels]
    return [raw.ch_names.index(channel) for channel in channels]


def extract_waveforms(
    raw,  # mne.io.brainvision.brainvision.RawBrainVision,
) -> Tuple[NDArray, NDArray, NDArray, float]:
    """Extract the audio and EEG data from the BV file."""
    # Access the EEG data array
    eeg_data = raw.get_data()
    print(f"Shape of Raw EEG data: {eeg_data.shape}")

    sampling_rate = int(
        raw.info["sfreq"]
    )  # Getting the sampling rate from the raw object

    if raw.ch_names[31] == "Soundwave":
        # BV records the audio in the last channel (since it is Cz and
        # removed as a reference)
        full_audio_waveform = eeg_data[-1:, :].copy()
        eeg_data = eeg_data[:31, :]  # Remove the audio channel
        eeg_data = np.concatenate((eeg_data, np.zeros((1, eeg_data.shape[1]))), axis=0)
        raw.ch_names[31] = "Cz"
        audio_waveform = full_audio_waveform[:, : 30 * sampling_rate]
    elif raw.ch_names[-1] == "TRIGGER":
        # This looks like CGX data
        eeg_data = eeg_data[:32, :]
        full_audio_waveform = None
        audio_waveform = None
    else:
        raise ValueError(f"Unknown EEG data type.  Channels are {raw.ch_names}")

    return audio_waveform, full_audio_waveform, eeg_data, sampling_rate


#######################  Extract the Tone Locations ############################


def teeger(x: NDArray) -> NDArray:
    """Compute the Teeger energy operator over a waveform.
    The result is a signal that is the product of the instanenous freqeuncy
    and the amplitude of the sinusoid.  We use this to tell us which tone
    is which.
    """
    # https://ieeexplore.ieee.org/document/8553703/
    return x[1:-1] ** 2 - x[:-2] * x[2:]

    # Helper function


def tone_times_from_csv(csv_file_path: str) -> Tuple[List[float], List[float]]:
    # Obsolete, since we can read the events times from the BV file.
    standard_tone_times = []
    deviant_tone_times = []

    # T is from the stimtrack,
    # S from the Matlab label and includes standard (1) and deviant (2)
    deviant: str = "??"
    with open(csv_file_path, "r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # This skips the header line
        for time, ty in csv_reader:
            if ty[0] == "S":
                deviant = ty
            elif ty[0] == "T":
                if deviant[-1] == "2":
                    deviant_tone_times.append(float(time))
                elif deviant[-1] == "1":
                    standard_tone_times.append(float(time))
                else:
                    raise ValueError(f"Unknown tone type: {ty}")
                deviant = False
    return standard_tone_times, deviant_tone_times


def get_event_locs(
    data_dir, full_audio_waveform, standard_tone, deviant_tone, sampling_rate
) -> Tuple[List[float], List[float]]:
    """Find the start of each tone in the audio waveform using cross
    correlation.  Return two lists, one giving the starting location of each
    standard tone, and the other giving the starting location of each deviant
    tone.  The locations are in seconds.
    """
    trigger_filename = os.path.join(data_dir, "triggers.csv")
    if os.path.exists(trigger_filename):
        print("Reading triggers.csv")
        standard_locs, deviant_locs = tone_times_from_csv(
            os.path.join(data_dir, "triggers.csv")
        )
        standard_locs = [x / sampling_rate for x in standard_locs]
        deviant_locs = [x / sampling_rate for x in deviant_locs]
    else:
        print("Finding triggers by cross correlation")
        standard_locs = (
            find_all_tone_starts(full_audio_waveform[0, :], standard_tone)
            / sampling_rate
        )
        deviant_locs = (
            find_all_tone_starts(full_audio_waveform[0, :], deviant_tone)
            / sampling_rate
        )
    return standard_locs, deviant_locs


def read_cgx_events(raw) -> Tuple[List[float], List[float]]:
    trigger = raw.get_data("TRIGGER")[0, :]
    standard_locs = np.where(np.logical_and(trigger[1:] == 1, trigger[:-1] == 0))[0]
    deviant_locs = np.where(np.logical_and(trigger[1:] == 2, trigger[:-1] == 0))[0]

    assert len(standard_locs) >= len(deviant_locs)
    standard_locs = standard_locs / raw.info["sfreq"]
    deviant_locs = deviant_locs / raw.info["sfreq"]
    return standard_locs, deviant_locs


def read_bv_events(
    raw, tone_length_s: float = 0.05
) -> Tuple[List[float], List[float], List[float]]:
    """Read the BV Event data and return list of standard and deviant
    event times.  All times are in seconds.

    Returns a list of events that tell the start time (in seconds) of the
    standard and deviant tones.  And a third list of latencies between the
    label event (by serial port) and the StimTrac sound threshold event.
    """
    events, event_id = mne.events_from_annotations(raw)
    sampling_rate = float(raw.info["sfreq"])
    standards = []
    deviants = []
    latencies = []
    stimtrac_id = event_id["Trigger/T  1"]
    standard_id = event_id["Stimulus/S  1"]
    deviant_id = event_id["Stimulus/S  2"]
    next_event_is_standard = True
    id_event_time = 0
    for event in events:
        event_time = event[0] / sampling_rate
        if event[2] == stimtrac_id:
            if next_event_is_standard:
                standards.append(float(event_time))
            else:
                deviants.append(float(event_time))
            latencies.append((event[0] - id_event_time) / sampling_rate)
        if event[2] == standard_id:
            next_event_is_standard = True
            id_event_time = event[0]
        elif event[2] == deviant_id:
            next_event_is_standard = False
            id_event_time = event[0]
    return standards, deviants, latencies


def label_tone_blips(freqm: NDArray) -> NDArray:
    """Label each sample of the audio waveform whether it is 0 (no sound),
    standard tone (1) or deviant tone (2).  The standard is found over the
    deviant because there are more standard tones.
    """
    if freqm.ndim == 1:
        freqm = freqm.reshape(-1, 1)
    if freqm.shape[1] == 1:
        freqm = np.concatenate((freqm, freqm, freqm), axis=1)

    # Fit the Teeger results with 3 clusters so we can identify each tone blip
    gm = GaussianMixture(3, covariance_type="diag")
    gm.fit(freqm)
    print("Label_tone_blips means are:", gm.means_)
    labels = gm.predict(freqm)

    # Find the GMM indices corresponding to the clusters that we want.  Silence
    # will be most common, and deviants least common.
    _, counts = np.unique(labels, return_counts=True)
    print("Label_tone_blips: Counts of the 3 clusters: ", counts)
    zero_label = int(np.argmax(counts))
    deviant_label = int(np.argmin(counts))
    standard_label = set(range(3)).difference({zero_label, deviant_label}).pop()
    results = np.zeros(freqm.shape[0], int)
    results[labels == standard_label] = 1
    results[labels == deviant_label] = 2
    return results


def find_most_common(my_list: List[int]) -> int:
    # Create a Counter object from the list
    element_counts = Counter(my_list)

    # Use most_common(1) to get the single most common element and its count
    # This returns a list of tuples, so we access the first element of the list
    # and then the first element of the tuple to get just the element itself.
    most_common_element = element_counts.most_common(1)[0][0]
    return most_common_element


def find_desired_segment(
    boolean_list: List[bool], n: int, values: ArrayLike, desired_label: int
):
    """
    Finds segments of consecutive True values in a boolean list
    that are at least N elements wide using itertools.groupby.

    Args:
      boolean_list: A list of boolean values.
      n: The minimum length of the segment.

    Returns:
      A list of tuples, where each tuple represents a segment
      and contains (start_index, end_index).
    """
    current_index = 0
    for value, group in groupby(boolean_list):
        group_list = list(group)
        group_length = len(group_list)
        if value and group_length >= n:
            label = find_most_common(
                values[current_index : current_index + group_length]
            )
            if label == desired_label:
                return (current_index, current_index + group_length - 1)
        current_index += group_length

    return None


def find_tone_examples(audio_waveform: NDArray) -> Tuple[NDArray, NDArray]:
    """Given an audio waveform, find the prototypical
    standard and deviant tone arrays.
    """
    freq = teeger(audio_waveform[0, :])  # Figure out freq (and amplitude)
    freqm = medfilt(freq, 7)  # Smooth freqs to remove glitches
    freqm = medfilt(freq, 7)  # Smooth freqs to remove glitches
    freqm = medfilt(freq, 7)  # Smooth freqs to remove glitches

    labels = label_tone_blips(freqm)  # Figure our which blips are which
    standard_label = 1
    deviant_label = 2

    standard_segment = find_desired_segment(labels > 0, 500, labels, standard_label)
    if standard_segment is None:
        raise ValueError("Standard segment not found")

    deviant_segment = find_desired_segment(labels > 0, 500, labels, deviant_label)
    if not deviant_segment:
        raise ValueError("Deviant segment not found")

    standard_tone = audio_waveform[0, standard_segment[0] : standard_segment[1]]
    deviant_tone = audio_waveform[0, deviant_segment[0] : deviant_segment[1]]

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


#######################  Extract the Tone Locations ############################


def rereference_eeg(
    eeg_data: NDArray, reference_channels: Union[str, List[int]] = "Average"
) -> NDArray:
    """Re-reference the EEG data, which is supplied in shape
    num_channels x num_times
    """
    assert eeg_data.ndim == 2
    assert eeg_data.shape[1] > eeg_data.shape[0]  # More time samples than channels
    if reference_channels == "Average":
        print("Rereferencing data to the average of all channels")
        reference = np.mean(eeg_data, axis=0, keepdims=True)
        rereferenced_eeg = eeg_data - reference
    elif reference_channels == ["XXXCz"]:
        print("Rereferencing data to Cz")
        # Restore the missing Cz channel
        rereferenced_eeg = np.concatenate(
            (eeg_data, np.zeros((1, eeg_data.shape[1]))), axis=0
        )
        rereferenced_eeg[31, :] = np.mean(rereferenced_eeg[:31, :], axis=0)
    elif len(reference_channels):
        print("Rereferencing data to the mean of the listed channels")
        # Compute mean of the reference channels
        reference = np.mean(eeg_data[reference_channels, :], axis=0, keepdims=True)
        rereferenced_eeg = eeg_data - reference
    else:
        raise ValueError(f"Unknown reference_channels: {reference_channels}")

    return rereferenced_eeg


def butter_bandpass(lowcut: float, highcut: float, fs: float = 25000, order=2):
    b, a = butter(order, [lowcut, highcut], fs, btype="band")
    return b, a


def filter_eeg(
    data: NDArray,
    lowcut_freq: float = 500,  # Hz
    highcut_freq: float = 1500,  # Hz
    sampling_rate: float = 25000,
    order: int = 8,
    axis: int = -1,
    debug_spectrum: bool = False,
) -> NDArray:
    # Example usage: Bandpass filter the eeg waveform .z
    sos = butter(
        order, [lowcut_freq, highcut_freq], fs=sampling_rate, output="sos", btype="band"
    )
    if debug_spectrum:
        w, h = sosfreqz(sos, fs=sampling_rate, worN=2000)
        plt.plot(w, 20 * np.log10(abs(h)), label="order = %d" % order)
        plt.grid("on")
    y = sosfiltfilt(sos, data, axis=axis)
    return y


#######################  ICA   ############################
# See also: https://labeling.ucsd.edu/tutorial/labels


def model_with_ica(
    filtered_eeg: NDArray,  # Shape n_samples x n_features
    sampling_rate: float = 25000,
    n_components: int = 10,
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
        [5 * i + ica_components[:, i : i + 1] for i in range(ica_components.shape[1])],
        axis=1,
    )

    plt.figure(figsize=(10, 8))
    num_points_to_plot = min(components.shape[0], 10 * sampling_rate)
    plt.plot(
        np.arange(num_points_to_plot) / sampling_rate,
        components[:num_points_to_plot, :],
    )
    for i in range(components.shape[1]):
        plt.text(0, 5 * i, f"IC #{i}")
        plt.xlabel("Time (s)")
    plt.title("ICA Components")
    return ica, components


def filter_ica_channels(
    ica: FastICA,
    ica_components: NDArray,
    bad_channels: List[int] = [2, 5, 6],
) -> NDArray:
    ica_components2 = ica_components.copy()  # Num_times x num_factors
    print("Removing ICA channels: ", bad_channels)
    for i in bad_channels:
        ica_components2[:, i] = 0

    cleaned_eeg = ica.inverse_transform(ica_components2)
    return cleaned_eeg


#######################  Finding ERPs   ############################


def accumulate_erp(
    eeg_data: NDArray,
    locs: NDArray,  # In seconds
    sampling_rate: float = 25000,
    num_samples: int = 0,
    pre_samples: int = 0,
    remove_baseline: bool = False,
) -> NDArray:
    """Average the EEG response for num_samples samples starting at each
    location in the locs argument.  The locations are in seconds, so the
    sampling rate must be correct.
    """
    num_channels, num_times = eeg_data.shape
    assert (
        num_times > num_channels
    ), f"num_times {num_times} is not > num_channels {num_channels}"
    if not num_samples:
        num_samples = int(0.5 * sampling_rate)  # 0.5s
    erp = 0
    count = 0
    for loc in locs:
        loc = int(loc * sampling_rate - pre_samples)
        if loc + num_samples < eeg_data.shape[1]:
            eeg = eeg_data[:, loc : loc + num_samples]
            if remove_baseline and num_samples > 0:
                eeg -= np.mean(eeg[:, :pre_samples], axis=1, keepdims=True)
            erp += eeg
            count += 1
    if count == 0:
        raise ValueError(f"No valid locations found in {locs}")
    return erp / count


def downsample_eeg(eeg_data: NDArray, sampling_rate: float, factor: int) -> NDArray:
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


#######################  Plotting   ############################


def plot_audio_waveforms(
    standard_sound: NDArray,
    deviant_sound: NDArray,
    sampling_rate: float = 25000,
    pre_samples=0,
):
    assert standard_sound.ndim == 2
    assert deviant_sound.ndim == 2
    assert standard_sound.shape[1] > standard_sound.shape[0], standard_sound.shape
    assert deviant_sound.shape[1] > deviant_sound.shape[0], deviant_sound.shape

    plt.clf()
    num_samples = int(0.100 * sampling_rate)  # To plot
    plt.subplot(2, 1, 1)
    plt.plot(
        np.arange(min(num_samples, standard_sound.shape[1])) / sampling_rate * 1000,
        standard_sound[0, :num_samples],
    )
    plt.ylabel("Standard")
    plt.axvline(pre_samples / sampling_rate * 1000, c="r")
    plt.title("Comparing Arverage Recorded Sounds for " "Standard and Deviant Tones")
    plt.subplot(2, 1, 2)
    plt.plot(
        np.arange(min(num_samples, deviant_sound.shape[1])) / sampling_rate * 1000,
        deviant_sound[0, :num_samples],
    )
    plt.ylabel("Deviant")
    plt.xlabel("Time (ms)")
    plt.axvline(pre_samples / sampling_rate * 1000, c="r")
    # We should see two clear sinusoids at different frequencies.


def plot_erp_images(
    normal_erp: NDArray,
    deviant_erp: NDArray,
    pre_samples: int = 0,
    sampling_rate: float = 25000,
):
    plt.clf()
    time_scale = (np.arange(normal_erp.shape[1]) - pre_samples) / sampling_rate * 1000

    extent = [np.min(time_scale), np.max(time_scale), 0, normal_erp.shape[0]]
    plt.subplot(2, 1, 1)
    plt.imshow(normal_erp, extent=extent)
    plt.axis("auto")
    plt.ylabel("Standard")
    plt.colorbar()
    plt.title("Comparing ERPs for Standard and Deviant Tones")
    plt.subplot(2, 1, 2)
    plt.imshow(deviant_erp, extent=extent)
    plt.ylabel("Deviant")
    plt.axis("auto")
    plt.xlabel("Time (ms)")
    plt.colorbar()


def plot_all_erp_channels(
    normal_erp,
    deviant_erp,
    channels,
    pre_samples: int = 0,
    sampling_rate: float = 25000,
):
    plt.clf()
    time_scale = (np.arange(normal_erp.shape[1]) - pre_samples) / sampling_rate * 1000

    max = np.maximum(
        np.max(np.abs(normal_erp[channels, :])),
        np.max(np.abs(deviant_erp[channels, :])),
    )
    plt.subplot(2, 1, 1)
    plt.plot(time_scale, normal_erp[channels, :].T)
    plt.ylabel("Standard Tone")
    plt.ylim([-max, max])
    plt.title("Comparing ERPs for Standard and Deviant Tones")
    # plt.xlabel('Time (ms)');

    plt.subplot(2, 1, 2)
    plt.plot(time_scale, deviant_erp[channels, :].T)
    plt.ylabel("Deviant Tone Results")
    plt.ylim([-max, max])
    plt.xlabel("Time (ms)")


def plot_all_erp_diff(
    normal_erp,
    deviant_erp,
    channels: List[int],
    pre_samples: int = 0,
    sampling_rate: float = 25000,
    bad_channels: List[int] = [],
):
    time_scale = (np.arange(normal_erp.shape[1]) - pre_samples) / sampling_rate * 1000

    normal_average = np.mean(normal_erp[channels, :], axis=0)
    deviant_average = np.mean(deviant_erp[channels, :], axis=0)
    plt.clf()
    plt.plot(time_scale, normal_average, label="Standard")
    plt.plot(time_scale, deviant_average, label="Deviant")
    plt.plot(time_scale, deviant_average - normal_average, label="Difference")
    plt.xlabel("Time (ms)")
    plt.ylabel(r"$\mu$V")
    plt.title(
        f"Average ERPs for Channels {channels} - " f"Removing ICA #{bad_channels}"
    )
    plt.legend()


def summarize_erp_diff(
    normal_erp, deviant_erp, channels: List[int], sampling_rate: float, pre_samples: int
):
    """Summarize the difference between the standard and deviant ERPs."""

    def rms(x):
        return np.sqrt(np.mean(x * x))

    # Calculate ratio of energy in the difference (standard vs. oddball)
    # compared to the standard ERP.
    normal_average = np.mean(normal_erp[channels, :], axis=0)
    deviant_average = np.mean(deviant_erp[channels, :], axis=0)
    diff = deviant_average - normal_average

    # Bussalb's metric (based on wave 1)
    wave1rms = rms(
        normal_erp[:, int(0.050 * sampling_rate) : int(0.150 * sampling_rate)]
    )
    noise_rms = rms(normal_erp[:, :pre_samples])
    wave1noise = wave1rms / noise_rms
    return rms(normal_average), rms(deviant_average), rms(diff), wave1noise


def save_fig(fig: plt.Figure, plot_dir: str, name: str) -> None:
    """Save a matplotlib figure to a file."""
    print("Writing data to", os.path.join(plot_dir, name))
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    fig.savefig(os.path.join(plot_dir, name))


#######################  Bootstrapping ############################


def bootstrap_sample(
    standard_locs: NDArray, deviant_locs: NDArray, count: int, num_trials: int = 30
):
    for _ in range(num_trials):
        indices = list(np.arange(len(standard_locs)))
        np.random.shuffle(indices)
        standard_indices = indices[:count]
        indices = list(np.arange(len(deviant_locs)))
        np.random.shuffle(indices)
        deviant_indices = indices[:count]
        yield standard_locs[standard_indices], deviant_locs[deviant_indices]


def bootstrap_sample_erp(
    raw,
    cleaned_eeg: NDArray,
    standard_locs: ArrayLike,
    deviant_locs: ArrayLike,
    sampling_rate: float,
    pre_samples: int,
    count: int,
    num_trials: int = 30,
):
    standard_locs = np.array(standard_locs)
    deviant_locs = np.array(deviant_locs)
    metric_mean = []
    metric_std = []
    bs_sizes = [len(standard_locs) // (2**i) for i in range(10)]
    bs_sizes = [i for i in bs_sizes if i >= 32]
    for size in bs_sizes:
        diffs = []
        for slocs, dlocs in bootstrap_sample(standard_locs, deviant_locs, size):
            serp = accumulate_erp(
                cleaned_eeg,
                slocs,
                sampling_rate,
                pre_samples=pre_samples,
                remove_baseline=True,
            )
            derp = accumulate_erp(
                cleaned_eeg,
                dlocs,
                sampling_rate,
                pre_samples=pre_samples,
                remove_baseline=True,
            )
            normal_rms, deviant_rms, diff_rms, wave1noise = summarize_erp_diff(
                serp, derp, find_channels(raw, "Cz"), sampling_rate, pre_samples
            )  # Just Cz
            ratio = diff_rms / normal_rms * 100
            # diffs.append(ratio)
            diffs.append(wave1noise)
        metric_mean.append(np.mean(diffs))
        metric_std.append(np.std(diffs))
    return bs_sizes, metric_mean, metric_std


def plot_bootstrap(
    key: str, bs_sizes: List[int], metric_mean: List[float], metric_std: List[float]
):
    plt.errorbar(bs_sizes, metric_mean, yerr=metric_std, label=key)
    plt.xlabel("Number of Trials")
    plt.ylabel("ERP RMS Diff Metric (%)")
    plt.title("BV Oddball Performance vs. Experiment Size")


#######################  Experiment Data ############################


class OddballExperiment:
    def __init__(
        self,
        data_dir: str,
        header_file: str = "",
        bad_channels: List[int] = [],
        plot_dir: str = "",
    ):
        self.data_dir = data_dir
        self.header_file = header_file
        self.bad_channels = bad_channels
        self.plot_dir = plot_dir


experiments = {
    "BV_wavefile": OddballExperiment(
        "oddball_with_triggers_wavFile",
        "P300TestWithWAV_20250702_1416.vhdr",
        [0, 3, 7],
        "plots/oddball_with_wav/",
    ),
    "BV_with_triggers": OddballExperiment(
        "oddball_test_with_triggers",
        "P300Test_20250702_1352.vhdr",
        [5, 7],
        "plots/oddball_with_triggers/",
    ),
    "BV_wavefile2": OddballExperiment(
        "oddball_with_triggers_wavFile",
        "Rupesh_Trial3.vhdr",
        [0, 7],
        "plots/oddball_with_wav2/",
    ),
    "CGX_Dry": OddballExperiment(
        "oddball_with_triggers_wavFile",
        "rupesh_0711_dryEEG.vhdr",
        [2, 5, 6, 9],
        "plots/oddball_dry_eeg/",
    ),
}


def run_all_experiments(lowcut: float, highcut: float, bootstrap_trials: int = 0):
    boot_results = {}
    for name, exp in experiments.items():
        print(f"\nRunning {name}")
        res = run_one_experiment(
            exp.data_dir,
            exp.header_file,
            exp.plot_dir,
            lowcut,
            highcut,
            exp.bad_channels,
        )
        if bootstrap_trials > 0:
            (bs_sizes, metric_mean, metric_std) = bootstrap_sample_erp(
                *res, bootstrap_trials
            )
            boot_results[name] = (bs_sizes, metric_mean, metric_std)

    if bootstrap_trials > 0:
        plt.clf()
        for name in boot_results:
            bs_sizes, metric_mean, metric_std = boot_results[name]
            plot_bootstrap(name, bs_sizes, metric_mean, metric_std)
        plt.legend()
        save_fig(plt.gcf(), ".", "AllExperimentBootstrap.png")


def run_one_experiment(
    data_dir: str,
    header_file: str,
    plot_dir: str,
    lowcut: float,
    highcut: float,
    bad_channels: List[int] = [],
    save_audio_file: bool = False,
):
    raw = read_bv_raw_data(data_dir, header_file)

    (audio_waveform, full_audio_waveform, eeg_data, sampling_rate) = extract_waveforms(
        raw
    )

    if save_audio_file:
        wavfile.write(
            save_audio_file,
            sampling_rate,
            (audio_waveform / np.max(np.abs(audio_waveform)) * 32768).astype(np.int16),
        )

    use_bv_event_timing = sampling_rate <= 8000

    if use_bv_event_timing:
        if raw.ch_names[-1] == "TRIGGER":
            print("Using CGX event timing.")
            standard_locs, deviant_locs = read_cgx_events(raw)
            standard_tone = None
            deviant_tone = None
        elif raw.ch_names[-1] == "Cz":
            print("Using BV event timing.")
            standard_locs, deviant_locs, _ = read_bv_events(raw)
            standard_tone = None
            deviant_tone = None
        else:
            raise ValueError(f"Unknown data format: {raw.ch_names[-1]}")
    else:
        print("Calculating event timing from StimTrac audio.")
        standard_tone, deviant_tone = find_tone_examples(audio_waveform)
        plot_audio_waveforms(
            standard_tone.reshape(1, -1), deviant_tone.reshape(1, -1), sampling_rate
        )
        save_fig(plt.gcf(), plot_dir, "tones_found.png")

        standard_locs, deviant_locs = get_event_locs(
            data_dir,
            full_audio_waveform,
            standard_tone,
            deviant_tone,
            sampling_rate,
        )

    rereferenced_eeg = rereference_eeg(eeg_data, "Average")

    filtered_eeg = filter_eeg(
        rereferenced_eeg, lowcut, highcut, sampling_rate, axis=1, order=6
    )

    ica, ica_components = model_with_ica(filtered_eeg.T, sampling_rate)
    save_fig(plt.gcf(), plot_dir, "ICA_components.png")

    cleaned_eeg = filter_ica_channels(ica, ica_components, bad_channels).T

    pre_samples = int(0.05 * sampling_rate)
    if not use_bv_event_timing:
        # Let's first make sure we get the right answer if we use the ERP
        # function to process the audio waveforms.

        standard_average_sound = accumulate_erp(
            full_audio_waveform, standard_locs, pre_samples=pre_samples
        )
        deviant_average_sound = accumulate_erp(
            full_audio_waveform, deviant_locs, pre_samples=pre_samples
        )
        plot_audio_waveforms(
            standard_average_sound, deviant_average_sound, sampling_rate, pre_samples
        )
        save_fig(plt.gcf(), plot_dir, "ERP_audio_waveforms.png")

    # Now we can calculate the ERP for the EEG data.
    normal_erp = accumulate_erp(
        cleaned_eeg,
        standard_locs,
        sampling_rate,
        pre_samples=pre_samples,
        remove_baseline=True,
    )
    deviant_erp = accumulate_erp(
        cleaned_eeg,
        deviant_locs,
        sampling_rate,
        pre_samples=pre_samples,
        remove_baseline=True,
    )
    plot_erp_images(normal_erp, deviant_erp, pre_samples, sampling_rate)
    save_fig(plt.gcf(), plot_dir, "ERP_images.png")

    plot_all_erp_channels(
        normal_erp,
        deviant_erp,
        list(range(32)),
        pre_samples=pre_samples,
        sampling_rate=sampling_rate,
    )
    save_fig(plt.gcf(), plot_dir, "ERP_all_channels.png")

    plot_all_erp_diff(
        normal_erp,
        deviant_erp,
        find_channels(raw, "Cz"),
        sampling_rate=sampling_rate,
        pre_samples=pre_samples,
        bad_channels=bad_channels,
    )
    save_fig(plt.gcf(), plot_dir, "ERP_channel_dif.png")

    normal_rms, deviant_rms, diff_rms, wave1noise = summarize_erp_diff(
        normal_erp,
        deviant_erp,
        find_channels(raw, "Cz"),  # Just Cz
        sampling_rate,
        pre_samples,
    )

    print(f"Standard RMS: {normal_rms:3g} uV")
    print(f"Deviant RMS: {deviant_rms:.3g} uV")
    print(f"Diff RMS: {diff_rms:.3g} uV")
    print(f"Diff to Standard: {diff_rms / normal_rms * 100:.2f}%")
    print(f"Normal RMS to Noise RMS: {wave1noise}")
    return (raw, cleaned_eeg, standard_locs, deviant_locs, sampling_rate, pre_samples)


FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "/tmp", "Directory where the raw EEG BV dat is stored.")
flags.DEFINE_string(
    "header_file", "", "Which header file (and its associated files) to read."
)

flags.DEFINE_boolean("runall", False, "Whether to process all experiments.")
flags.DEFINE_integer("lowcut", 1, "Frequency for low-side of EEG bandpass filter")
flags.DEFINE_integer("highcut", 15, "Frequency for high-side of EEG bandpass filter")
flags.DEFINE_string("plot_dir", "plots", "Where to store debugging plots")
flags.DEFINE_multi_integer("bad_channels", [], "List of bad channels to remove.")
flags.DEFINE_string("save_audio_file", "", "Where to save the BV recorded audio file")
flags.DEFINE_integer("bootstrap_trials", 30, "Number of bootstrap trials to run")


def main(*argv):
    if FLAGS.runall:
        run_all_experiments(FLAGS.lowcut, FLAGS.highcut, FLAGS.bootstrap_trials)
    else:
        run_one_experiment(
            FLAGS.data_dir,
            FLAGS.header_file,
            FLAGS.plot_dir,
            FLAGS.lowcut,
            FLAGS.highcut,
            FLAGS.bad_channels,
            FLAGS.save_audio_file,
        )


if __name__ == "__main__":
    app.run(main)
