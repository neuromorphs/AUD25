import time
import numpy as np
import mne               # EEG handling
import librosa           # audio processing
import matplotlib.pyplot as plt
import sounddevice as sd  # For playback
from scipy.signal import resample, correlate, hilbert  # signal ops
from scipy.stats import pearsonr  # correlation metric

# suppress verbose MNE logs
mne.set_log_level('CRITICAL')

def lag_generator_new(r, lags):
    '''
    Create time-lagged versions of neural data.
    r: array [time, neurons]
    lags: tuple (min_lag, max_lag)
    returns: array [time, neurons * num_lags]
    '''
    # expand lag range and prepare output array
    lags = list(range(lags[0], lags[1]+1))
    out = np.zeros([r.shape[0], r.shape[1]*len(lags)])
    # pad to handle shifts
    r = np.pad(r, ((0, len(lags)), (0, 0)), 'constant')

    r_lag_list = []
    for lag in lags:
        t1 = np.roll(r, lag, axis=0)
        # zero out wrapped samples
        if lag < 0:
            t1[lag-1:, :] = 0
        else:
            t1[:lag, :] = 0
        # collect aligned segment
        r_lag_list.append(t1[:out.shape[0], :])
    # concatenate all lagged versions
    out = np.concatenate(r_lag_list, axis=1)
    return out


def eeg_cca_from_chunk(chunk, fs, long_audio1, long_audio2, long_sr):
    # CCA-based correlation of EEG to two stimuli

    # filtering params
    bpf_applied = True
    freq_low, freq_high = 0.1, 15
    ftype, order = 'butter', 3
    downfreq = 128  # target EEG rate

    # lags for neural and stimulus signals
    lags_neuro = [-40, 10]
    lags_stim = [-10, 10]

    # channel naming
    ch_names = ['Fp1','Fz','F3','F7','F9','FC5','FC1','C3','T7','CP5','CP1','Pz',
                'P3','P7','P9','O1','Oz','O2','P10','P8','P4','CP2','CP6','T8',
                'C4','FC2','FC6','F10','F8','F4','Fp2','AUX_1','AUX_2']

    # load pretrained CCA weights
    x_weights = np.load('../Analysis/CCA/Weights/RealTime/speech_only_x_weights.npy')
    y_weights = np.load('../Analysis/CCA/Weights/RealTime/speech_only_y_weights.npy')

    # convert incoming chunk to array
    data_array = np.array(chunk)

    # create MNE Raw object for EEG + stim channels
    info = mne.create_info(ch_names=ch_names,
                            ch_types=['eeg']*31 + ['stim']*2,
                            sfreq=fs)
    stim = data_array[:, -2:]
    stim1, stim2 = stim[:, 0], stim[:, 1]
    eeg = mne.io.RawArray(data_array.T, info)
    eeg = eeg.pick_channels(ch_names[:31])  # drop stim channels

    # bandpass filter EEG
    iir_params = dict(order=order, ftype=ftype)
    mne.filter.create_filter(eeg.get_data(), eeg.info['sfreq'],
                             l_freq=freq_low, h_freq=freq_high,
                             method='iir', iir_params=iir_params)

    # downsample for CCA
    eeg = eeg.resample(sfreq=downfreq).get_data()

    # normalize and align stimulus segments
    short_audio = (stim2 - np.mean(stim2)) / np.std(stim2)
    correlation = correlate(long_audio1, short_audio, mode='valid')
    best = np.argmax(correlation)
    end = best + len(short_audio)
    print(f"Best match at {best}-{end}, corr={np.max(correlation):.2f}")
    stim1 = long_audio1[best:end]
    stim2 = long_audio2[best:end]

    # extract envelopes and resample to EEG rate
    def make_env(sig):
        env = np.abs(hilbert(sig))
        n_samples = int(len(env)/fs * downfreq)
        return np.expand_dims(resample(env, n_samples), axis=1)

    env1, env2 = make_env(stim1), make_env(stim2)

    # load normalization stats and apply
    stats = np.load('../Analysis/CCA/speech_all_means_stds.npy', allow_pickle=True).item()
    eeg = ((eeg.T - stats['eeg_mean'])/stats['eeg_std']).T
    env1 = (env1 - stats['stim_mean'])/stats['stim_std']
    env2 = (env2 - stats['stim_mean'])/stats['stim_std']

    # apply time lags
    eeg = lag_generator_new(eeg.T, lags_neuro)
    env1 = lag_generator_new(env1, lags_stim)
    env2 = lag_generator_new(env2, lags_stim)

    # trim to equal length
    n = min(eeg.shape[0], env1.shape[0])
    eeg, env1 = eeg[:n], env1[:n]

    # project into CCA space
    Xc = eeg @ x_weights
    Y1c = env1 @ y_weights
    Y2c = env2 @ y_weights

    # compute correlations
    r1 = pearsonr(Xc.flatten(), Y1c.flatten()).statistic
    r2 = pearsonr(Xc.flatten(), Y2c.flatten()).statistic
    return r1, r2
