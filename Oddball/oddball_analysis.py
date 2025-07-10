import csv
import os
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np

from absl import app, flags
from numpy.typing import NDArray
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


def read_bv_raw_data(
    data_dir: str, header_file: str):  # Can't figure out the right type...
    # Specify the path to your BrainVision header file
    vhdr_file = os.path.join(data_dir, header_file)

    # Read the BrainVision data
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)

    # Print some basic information about the data
    print(raw.info)
    print(f"Channel names: {raw.ch_names}")
    print(f"Number of samples: {raw.n_times}")
    return raw


def extract_waveforms(
    raw,  # mne.io.brainvision.brainvision.RawBrainVision,
) -> Tuple[NDArray, NDArray, NDArray, float]:
    """Extract the audio and EEG data from the BV file."""
    # Access the EEG data array
    eeg_data = raw.get_data()
    print(f"Shape of EEG data: {eeg_data.shape}")

    full_audio_waveform = eeg_data[-1:, :].copy()
    sampling_rate = int(
        raw.info["sfreq"]
    )  # Getting the sampling rate from the raw object

    eeg_data = eeg_data[:31, :]  # Remove the audio channel
    eeg_data = np.concatenate((eeg_data, np.zeros((1, eeg_data.shape[1]))),
                              axis=0)
    raw.ch_names[31] = "Cz"

    audio_waveform = full_audio_waveform[:, : 30 * sampling_rate]
    return audio_waveform, full_audio_waveform, eeg_data, sampling_rate


def teeger(x: NDArray) -> NDArray:
    """Compute the Teeger energy operator over a waveform.
    The result is a signal that is the product of the instanenous freqeuncy
    and the amplitude of the sinusoid.  We use this to tell us which tone
    is which.
    """
    # https://ieeexplore.ieee.org/document/8553703/
    return x[1:-1] ** 2 - x[:-2] * x[2:]

    # Helper function


def rereference_eeg(
    eeg_data: NDArray, reference_channels: List[str] = ["Cz"]
) -> NDArray:
    """Re-reference the EEG data, which is supplied in shape
    num_channels x num_times
    """
    assert eeg_data.ndim == 2
    assert eeg_data.shape[1] > eeg_data.shape[0]
    if reference_channels == ["Cz"]:
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
        print("Rereferencing data to the mean of *all* the channels")
        # Compute the mean of *all* the channels
        reference = np.mean(eeg_data, axis=0, keepdims=True)
        rereferenced_eeg = eeg_data - reference

    print("Re-referenced eeg data has shape", rereferenced_eeg.shape)
    return rereferenced_eeg


def tone_times_from_csv(csv_file_path: str) -> Tuple[List[float], List[float]]:
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
        plt.title('Finding Standard and Deviant Tone Examples')

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
    plt.title('ICA Components')
    return ica, components


def filter_ica_channels(
    ica: FastICA,
    ica_components: NDArray,
    bad_channels: List[int] = [2, 5, 6],
) -> NDArray:
    ica_components2 = ica_components.copy()  # Num_times x num_factors
    print('Removing ICA channels: ', bad_channels)
    for i in bad_channels:
        ica_components2[:, i] = 0

    cleaned_eeg = ica.inverse_transform(ica_components2)
    return cleaned_eeg


def accumulate_erp(
    eeg_data: NDArray,
    locs: NDArray,  # In seconds
    sampling_rate: float = 25000,
    num_samples: int = 12500,
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


def plot_audio_waveforms(
    standard_average_sound: NDArray,
    deviant_average_sound: NDArray,
    sampling_rate: float = 25000,
    pre_samples=0,
):
    plt.clf()
    num_samples = int(0.100 * sampling_rate)  # To plot
    plt.subplot(2, 1, 1)
    plt.plot(
        np.arange(num_samples) / sampling_rate * 1000,
        standard_average_sound[0, :num_samples],
    )
    plt.ylabel("Standard")
    plt.axvline(pre_samples / sampling_rate * 1000, c="r")
    plt.title('Comparing Arverage Recorded Sounds for '
              'Standard and Deviant Tones')
    plt.subplot(2, 1, 2)
    plt.plot(
        np.arange(num_samples) / sampling_rate * 1000,
        deviant_average_sound[0, :num_samples],
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
    plt.title('Comparing ERPs for Standard and Deviant Tones')
    plt.subplot(2, 1, 2)
    plt.imshow(deviant_erp, extent=extent)
    plt.ylabel("Deviant")
    plt.axis("auto")
    plt.xlabel("Time (ms)")
    plt.colorbar()

def plot_all_erp_channels(normal_erp, deviant_erp, channels,
                          pre_samples: int = 0, sampling_rate: float = 25000):
  plt.clf()
  time_scale = (np.arange(normal_erp.shape[1]) - pre_samples)/sampling_rate*1000

  max = np.maximum(np.max(np.abs(normal_erp[channels, :])),
                   np.max(np.abs(deviant_erp[channels, :])))
  plt.subplot(2, 1, 1)
  plt.plot(time_scale, normal_erp[channels, :].T)
  plt.ylabel('Standard Tone')
  plt.ylim([-max, max])
  plt.title('Comparing ERPs for Standard and Deviant Tones')
  # plt.xlabel('Time (ms)');

  plt.subplot(2, 1, 2)
  plt.plot(time_scale, deviant_erp[channels, :].T)
  plt.ylabel('Deviant Tone Results')
  plt.ylim([-max, max])
  plt.xlabel('Time (ms)');


def plot_all_erp_diff(normal_erp, deviant_erp, channels: List[int],
                          pre_samples: int = 0, sampling_rate: float = 25000,
                          bad_channels: List[int] = []):
  time_scale = (np.arange(normal_erp.shape[1]) - pre_samples)/sampling_rate*1000

  normal_average = np.mean(normal_erp[channels, :], axis=0)
  deviant_average = np.mean(deviant_erp[channels, :], axis=0)
  plt.clf()
  plt.plot(time_scale, normal_average,
          label='Standard')
  plt.plot(time_scale, deviant_average,
          label='Deviant')
  plt.plot(time_scale, deviant_average - normal_average,
          label='Difference')
  plt.xlabel('Time (ms)')
  plt.ylabel(r'$\mu$V')
  plt.title(f'Average ERPs for Channels {channels} - Removing ICA #{bad_channels}')
  plt.legend()


FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "/tmp", "Directory where the raw EEG BV dat is stored.")
flags.DEFINE_string(
    "header_file", "/tmp",
    "Which header file (and its associated files) to read."
)

flags.DEFINE_integer("lowcut", 1,
                     "Frequency for low-side of EEG bandpass filter")
flags.DEFINE_integer("highcut", 15,
                     "Frequency for high-side of EEG bandpass filter")
flags.DEFINE_string("plot_dir", "plots",
                    "Where to store debugging plots")
flags.DEFINE_multi_integer("bad_channels", [],
                           "List of bad channels to remove.")


def save_fig(fig: plt.Figure, plot_dir: str, name: str) -> None:
    """Save a matplotlib figure to a file."""
    print('Writing data to', os.path.join(plot_dir, name))
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    fig.savefig(os.path.join(plot_dir, name))


def main(*argv):
    raw = read_bv_raw_data(FLAGS.data_dir, FLAGS.header_file)

    (audio_waveform, full_audio_waveform,
     eeg_data, sampling_rate) = extract_waveforms(raw)

    standard_tone, deviant_tone = find_tone_examples(audio_waveform,
                                                     debug_plots=True)
    save_fig(plt.gcf(), FLAGS.plot_dir, "Tone_examples.png")

    standard_locs, deviant_locs = get_event_locs(
        FLAGS.data_dir, full_audio_waveform, standard_tone, deviant_tone,
        sampling_rate
    )

    rereferenced_eeg = rereference_eeg(eeg_data, ["Cz"])

    filtered_eeg = filter_eeg(
        rereferenced_eeg, FLAGS.lowcut, FLAGS.highcut,
        sampling_rate, axis=1, order=6
    )

    ica, ica_components = model_with_ica(filtered_eeg.T, sampling_rate)
    save_fig(plt.gcf(), FLAGS.plot_dir, "ICA_components.png")

    cleaned_eeg = filter_ica_channels(ica, ica_components,
                                      FLAGS.bad_channels).T

    # Let's first make sure we get the right answer if we use the ERP o
    # function to process the audio waveforms.

    pre_samples = int(0.05 * sampling_rate)
    standard_average_sound = accumulate_erp(
        full_audio_waveform, standard_locs, pre_samples=pre_samples
    )
    deviant_average_sound = accumulate_erp(
        full_audio_waveform, deviant_locs, pre_samples=pre_samples
    )
    plot_audio_waveforms(
        standard_average_sound, deviant_average_sound,
        sampling_rate, pre_samples
    )
    save_fig(plt.gcf(), FLAGS.plot_dir, "ERP_audio_waveforms.png")

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
    save_fig(plt.gcf(), FLAGS.plot_dir, "ERP_images.png")

    plot_all_erp_channels(normal_erp, deviant_erp, list(range(32)),
                          pre_samples=pre_samples,
                          sampling_rate=sampling_rate)
    save_fig(plt.gcf(), FLAGS.plot_dir, "ERP_all_channels.png")

    plot_all_erp_diff(normal_erp, deviant_erp, [31],
                      pre_samples=pre_samples,
                      bad_channels=FLAGS.bad_channels)
    save_fig(plt.gcf(), FLAGS.plot_dir, "ERP_channel_dif.png")


if __name__ == "__main__":
    app.run(main)
