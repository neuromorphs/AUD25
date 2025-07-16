% =========================================================================%
% Author: MP Huet (mphuet@jhu.edu)
% If used, please cite:
% Huet & Elhilali (2025), bioRxiv, https://doi.org/10.1101/2025.05.22.655464
% =========================================================================
function [spectrogram_matrix, time, freqs] = GetSpectrogram(signal, fs, fmin, fmax, numChannelsPerOctave)
% GETSPECTROGRAM Computes a cochlear spectrogram of an input signal using a gammatone filterbank.
% COMMENT: YOU need to have the amtoolbox on
%
% Inputs:
%   signal               - Input signal
%   fs                   - Sampling frequency of the input signal
%   fmin                 - Minimum frequency for the filterbank
%   fmax                 - Maximum frequency for the filterbank
%   numChannelsPerOctave - Number of channels per octave in the filterbank
%
% Outputs:
%   spectrogram_matrix   - Spectrogram of the input signal
%   time                 - Time vector corresponding to the spectrogram
%   freqs                - Frequencies band of the filterbank channels
%
% Created by MP Huet: mphuet@jhu.edu
% Date: 03/01/2024

    % Check if the sampling frequency meets the Nyquist condition
    if fs < 2*fmax
        error('The sampling frequency does not meet the Nyquist condition. Increase the sampling frequency.');
    end

    % Check the dimension of the signal
    [rows, cols] = size(signal);
    if rows == 1 && cols > 1
        signal = signal';
    end

    % Calculate the number of octaves based on the frequency range
    numOctaves = log2(fmax/fmin);
    totalChannels = floor(numOctaves * numChannelsPerOctave);
    octave_ratio = 2^(1/numChannelsPerOctave);
    freqs = fmin * octave_ratio.^(0:totalChannels);
    time = (0:length(signal)-1)/fs;
    totalChannels = length(freqs);

    % Initialize the output matrix
    spectrogram_matrix = zeros(length(signal), totalChannels-1);

    % Create gammatone filterbank with myedge set to 0.6
    [b, a] = gammatone(freqs, fs, 4, 0.6, 'complex');
    outsig = 2 * real(ufilterbankz(b, a, signal));

    % Process the last channel (highest frequency)
    y2_h = outsig(:, end);

    % Parameters for the cochlear filterbank
    alph = exp(-1/(8*2^(4-1)));	

    % Process all other channels
    for ch = (totalChannels-1):-1:1
        y2 = outsig(:, ch);
        y3 = y2 - y2_h;
        y2_h = y2;
        y4 = max(y3, 0);
        y5 = filter(1, [1 -alph], y4);
        spectrogram_matrix(:, ch) = y5(1:length(signal));
    end;

% Logarythmic compression
spectrogram_matrix = log1p(abs(spectrogram_matrix));

end
