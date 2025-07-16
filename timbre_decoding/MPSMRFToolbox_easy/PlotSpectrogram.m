% =========================================================================%
% Author: MP Huet (mphuet@jhu.edu)
% If used, please cite:
% Huet & Elhilali (2025), bioRxiv, https://doi.org/10.1101/2025.05.22.655464
% =========================================================================
function PlotSpectrogram(spectrogram_matrix, time, freqs, scale)
% PlotSpectrogram Plots the spectrogram of a signal with the desired frequency scale.
%
% INPUTS:
%   spectrogram_matrix  - Matrix representing the spectrogram of the signal.
%   time                - Time scale corresponding to the spectrogram.
%   center_freqs        - Center frequencies of the gammatone filters.
%   scale               - Desired frequency scale for the y-axis ('Hz' or
%                         'cycles_per_octave'). (Default = 'Hz')
%
% OUTPUTS:
%   None. The function directly plots the spectrogram.
%
% Created by MP Huet: mphuet@jhu.edu
% Date: 03/01/2024

% Set default values if necessary
if nargin < 4 || isempty(scale)
    scale = 'Hz'; % Default value
end

numChannelsPerOctave = floor(length(freqs)/(log2(freqs(end) / freqs(1))));

% Check the desired scale and adjust y-axis accordingly
switch scale
    case 'Hz'
        y_values = freqs;
        ylabel_str = 'Frequency (Hz)';
        y_scale_type = 'log';
        
    case 'cycles_per_octave'
        f_reference = freqs(1); % using the lowest frequency as reference
        y_values = log2(freqs / f_reference);
        ylabel_str = 'Cycles per Octave';
        y_scale_type = 'linear'; % The y-scale is inherently logarithmic for cycles per octave
        
    otherwise
        error('Invalid scale specified. Choose "Hz" or "cycles_per_octave".');
end

% Get fmin and fmax
fmin = y_values(1);
fmax = y_values(end);

% Ensure fmin and fmax are valid for logarithmic scale
if strcmp(y_scale_type, 'log') && (fmin <= 0 || fmax <= 0)
    error('Frequency values must be positive for logarithmic scale.');
end

% Plot the spectrogram
imagesc(time, 1:length(y_values), spectrogram_matrix');
set(gca, 'YDir', 'normal'); % Ensure the y-direction is correct

% Set Y-ticks and Y-tick labels to reflect the frequencies
set(gca, 'YTick', 1:numChannelsPerOctave:length(y_values), 'YTickLabel', arrayfun(@num2str, y_values(1:numChannelsPerOctave:length(y_values)), 'UniformOutput', false));
xlabel('Time (s)');
ylabel(ylabel_str);
colormap(viridis);

end
