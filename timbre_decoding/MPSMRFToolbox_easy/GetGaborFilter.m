% =========================================================================%
% Author: MP Huet (mphuet@jhu.edu)
% If used, please cite:
% Huet & Elhilali (2025), bioRxiv, https://doi.org/10.1101/2025.05.22.655464
% =========================================================================
function gaborFilter = GetGaborFilter(omega_t, omega_f, fs, f_step)
% GetGaborFilter Generates a 2D Gabor filter using specified parameters.
%
% INPUTS:
%   omega_t     - Temporal frequency.
%   omega_f     - Spectral frequency.
%   fs          - Sampling frequency.
%   freqs       - Array of frequencies.
%
% OUTPUT:
%   gaborFilter - 2D Gabor filter.
%
% Created by MP Huet: mphuet@jhu.edu
% Date: 04/01/2024

% Adjust sigma values based on omega values
sigma_t = 1 / (0.5 * abs(omega_t));
sigma_f = 1 / (0.5 * omega_f);

% Define the window for time values
t_small = 1/fs:1/fs:abs(sigma_t);
t_small = t_small - mean(t_small);

f_small = f_step:f_step:abs(sigma_f);
f_small = f_small - mean(f_small);

% Create a 2D meshgrid of time and frequency values
[t, f] = meshgrid(t_small, f_small);

% Rotate t and f_grid coordinates based on the orientation angle of 0
t_prime = t * cos(0) + f * sin(0);
f_prime = -t * sin(0) + f * cos(0);

% Compute the adjustment factor based on the given parameters
adjustment = 1 / (2 * pi * sigma_t * sigma_f);

% Compute the 2D Gaussian component of the Gabor filter
gaussian = exp(-0.5 * (t_prime.^2 / sigma_t^2 + f_prime.^2 / sigma_f^2));

% Generate the complex sinusoidal component of the Gabor filter
complex_sinusoid = exp(1i * 2 * pi * (omega_t * t + omega_f * f));
%complex_sinusoid = cos(2*pi*(omega_t*t_prime + omega_f*f_prime));

% Compute the 2D Gabor filter
gaborFilter = gaussian .* complex_sinusoid;
gaborFilter = gaborFilter * adjustment;
gaborFilter = gaborFilter';

% Visualization (commented out)
% figure;
% imagesc(t_small,f_small,real(gaborFilter));
% colormap('jet');
% colorbar;
% axis image;
% title('2D Gabor Filter');


end
