% =========================================================================
% Example: Compute the Modulation Power Spectrum (MPS) / Modulation Response Function (MRF)
%
% Author: Moïra-Phoebé Huet (MP Huet)
% Email:  mphuet@jhu.edu
% Date:   2025-07-16
%
% Description:
% This script computes the MPS from an acoustic waveform or the MRF from a 
% reconstructed spectrogram. 
%
% If you use this code, please cite the following paper:
%
% Huet, M.-P., & Elhilali, M. (2025). 
% The shape of attention: How cognitive goals sculpt cortical representation of speech.
% bioRxiv. https://doi.org/10.1101/2025.05.22.655464
% =========================================================================

%% 1. Parameters

fmin = 125;                % Minimum frequency for the spectrogram
fmax = 8000;               % Maximum frequency
channelsPerOct = 16;       % Channels per octave
target_fs = 100;           % Target sampling rate for modulation
omegas_t = 2.^(-3:1:2);    % Temporal modulation frequencies (rate)
omegas_f = 2.^(-2:1:2);    % Spectral modulation frequencies (scale)

%% 2. Load audio

audio_file = 'audio.wav';

[x, fs] = audioread(audio_file);
x = mean(x, 2);                          % Mono mix
x = (x - mean(x)) / std(x);             % Normalize
x = x / max(abs(x)) * 0.98;             % Avoid clipping

%% 3. Compute and plot spectrogram

[y, time, freqs] = GetSpectrogram(x, fs, fmin, fmax, channelsPerOct);
figure;
PlotSpectrogram(y, time, freqs);
title('Spectrogram');

% Resample spectrogram along time axis
y = resample(y, target_fs, fs);  
time = (0:length(y)-1) / target_fs;

%% 4. Compute and plot MPS/MRF

MRF = GetRS(y, target_fs, freqs, omegas_t, omegas_f);
MRF = PermuteFold(MRF, omegas_t, omegas_f);

figure;
PlotRS(MRF, omegas_t, omegas_f, 'MPS for audio');