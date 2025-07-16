% =========================================================================%
% Author: MP Huet (mphuet@jhu.edu)
% If used, please cite:
% Huet & Elhilali (2025), bioRxiv, https://doi.org/10.1101/2025.05.22.655464
% =========================================================================
function f_step = GetFstep(freqs)

% Set the reference frequency to the lowest frequency in the freqs array
f_reference = freqs(1); 
f_octaves = log2(freqs / f_reference);
f_step = f_octaves(2)-f_octaves(1);

end