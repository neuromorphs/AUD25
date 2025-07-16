% =========================================================================%
% Author: MP Huet (mphuet@jhu.edu)
% If used, please cite:
% Huet & Elhilali (2025), bioRxiv, https://doi.org/10.1101/2025.05.22.655464
% =========================================================================
function R = GetRS(y, fs, freqs, omegas_t, omegas_f)

% Define the steps for frequency axis
f_step = GetFstep(freqs);

% Define center frequencies in Hz and cycles/octaves
omegas_t = [-fliplr(omegas_t), omegas_t];

% Initialize the result matrix
R = zeros(length(omegas_t), length(omegas_f));

% Apply Gabor filters to the spectrogram
for omega_t = 1:length(omegas_t)
    for omega_f = 1:length(omegas_f)
        % Get the Gabor filter for the current frequencies
        target_rate = log2(abs(omegas_t(omega_t)));
        target_scale = log2(abs(omegas_f(omega_f)));
        target_ratesign = sign(omegas_t(omega_t));
        gaborFilter = GetGaborFilter(omegas_t(omega_t), omegas_f(omega_f),fs, f_step);
        conv_result = mean(abs(conv2(y, gaborFilter, 'same')), 'all');
        R(omega_t, omega_f) = conv_result;
    end
end


R = R';

    function str = roundvalue(value)
        if value == round(value)
            str = sprintf('%.0f', value);  % Affiche comme un entier
        elseif value == round(value, 1)
            str = sprintf('%.1f', value);  % Un seul chiffre après la virgule si nécessaire
        else
            str = sprintf('%.2f', value);  % Deux chiffres après la virgule si nécessaire
        end
    end

    function new_folder(folder_in)
        if ~exist(folder_in,'dir')
            mkdir (folder_in)
        end
    end
end