import time
import numpy as np
import mne               # EEG processing
import librosa           # audio loading and resampling
import matplotlib.pyplot as plt
import sounddevice as sd  # audio playback
from scipy.signal import resample, correlate, hilbert  # signal processing
from scipy.stats import pearsonr  # compute correlation

# suppress detailed MNE logs
mne.set_log_level('CRITICAL')

def lag_generator_new(r, lags):
    '''
    Args:
        r: array [time, neurons]
        lags: tuple (min_lag, max_lag)
    Returns:
        out: array [time, neurons * num_lags]
    '''
    # create list of lag offsets
    lags = list(range(lags[0], lags[1] + 1))
    out = np.zeros([r.shape[0], r.shape[1] * len(lags)])
    # pad to accommodate positive lags
    r = np.pad(r, ((0, len(lags)), (0, 0)), 'constant')

    r_lag_list = []
    for lag in lags:
        # shift the signal
        t1 = np.roll(r, lag, axis=0)
        # zero out wrapped values
        if lag < 0:
            t1[lag - 1 :, :] = 0
        else:
            t1[:lag, :] = 0
        # keep valid portion for this lag
        r_lag_list.append(t1[:out.shape[0], :])
    # concatenate all lagged data
    out = np.concatenate(r_lag_list, axis=1)
    return out


def eeg_cca_from_chunk_combination(chunk, fs, long_audio1, long_audio2, long_sr):
    # CCA-based correlation for speech and music stimuli

    # filter settings
    freq_low, freq_high = 0.1, 15
    ftype, order = 'butter', 3
    downfreq = 128

    # lag windows
    lags_neuro = [-40, 10]
    lags_stim = [-10, 10]

    # EEG + stim channel names
    ch_names = ['Fp1', 'Fz', 'F3', 'F7', 'F9', 'FC5', 'FC1', 'C3', 'T7', 'CP5',
                'CP1', 'Pz', 'P3', 'P7', 'P9', 'O1', 'Oz', 'O2', 'P10', 'P8',
                'P4', 'CP2', 'CP6', 'T8', 'C4', 'FC2', 'FC6', 'F10', 'F8',
                'F4', 'Fp2', 'AUX_1', 'AUX_2']

    # load pretrained CCA weights for both conditions
    x_weights_speech = np.load('../Analysis/CCA/4_epochs/speech_only_x_weights.npy')
    y_weights_speech = np.load('../Analysis/CCA/4_epochs/speech_only_y_weights.npy')
    x_weights_music = np.load('../Analysis/CCA/4_epochs/music_only_x_weights.npy')
    y_weights_music = np.load('../Analysis/CCA/4_epochs/music_only_y_weights.npy')

    # prepare data array
    data_array = np.array(chunk)  # shape: (samples, channels)

    # create MNE Raw object including stim channels
    info = mne.create_info(ch_names=ch_names,
                            ch_types=['eeg']*31 + ['stim']*2,
                            sfreq=fs)
    stim = data_array[:, -2:]
    stim1, stim2 = stim[:, 0], stim[:, 1]
    eeg = mne.io.RawArray(data_array.T, info)
    eeg = eeg.pick_channels(ch_names[:31])  # retain only EEG channels

    # apply bandpass filter
    iir_params = dict(order=order, ftype=ftype)
    mne.filter.create_filter(eeg.get_data(), eeg.info['sfreq'],
                             l_freq=freq_low, h_freq=freq_high,
                             method='iir', iir_params=iir_params)

    # downsample for CCA
    eeg = eeg.resample(sfreq=downfreq).get_data()

    # normalize short stimulus segment (z-score)
    short_audio = np.squeeze(stim2)
    short_audio = (short_audio - np.mean(short_audio)) / np.std(short_audio)

    # align stimulus to long reference via cross-correlation
    corr = correlate(long_audio1, short_audio, mode='valid')
    best = np.argmax(corr)
    end = best + len(short_audio)
    print(f"Best match at {best}-{end}, corr={np.max(corr):.2f}")
    stim1 = long_audio1[best:end]
    stim2 = long_audio2[best:end]

    # compute envelopes and resample to EEG rate
    def make_env(sig):
        env = np.abs(hilbert(sig))
        n_samples = int(len(env)/fs * downfreq)
        return np.expand_dims(resample(env, n_samples), axis=1)

    env1, env2 = make_env(stim1), make_env(stim2)

    # load means/stds and apply z-score normalization
    stats = np.load('../Analysis/CCA/4_epochs/speech_all_means_stds.npy', allow_pickle=True).item()
    eeg = ((eeg.T - stats['eeg_mean']) / stats['eeg_std']).T
    env1 = (env1 - stats['stim_mean']) / stats['stim_std']
    env2 = (env2 - stats['stim_mean']) / stats['stim_std']

    # apply time lags
    eeg = lag_generator_new(eeg.T, lags_neuro)
    env1 = lag_generator_new(env1, lags_stim)
    env2 = lag_generator_new(env2, lags_stim)

    # trim to equal length
    n = min(eeg.shape[0], env1.shape[0])
    eeg, env1, env2 = eeg[:n], env1[:n], env2[:n]

    # CCA projections for speech
    Xc_s = eeg @ x_weights_speech
    Y1_s = env1 @ y_weights_speech
    Y2_s = env2 @ y_weights_speech
    r1_s = pearsonr(Xc_s.flatten(), Y1_s.flatten()).statistic
    r2_s = pearsonr(Xc_s.flatten(), Y2_s.flatten()).statistic

    # CCA projections for music
    Xc_m = eeg @ x_weights_music
    Y1_m = env1 @ y_weights_music
    Y2_m = env2 @ y_weights_music
    r1_m = pearsonr(Xc_m.flatten(), Y1_m.flatten()).statistic
    r2_m = pearsonr(Xc_m.flatten(), Y2_m.flatten()).statistic

    return r1_s, r2_s, r1_m, r2_m