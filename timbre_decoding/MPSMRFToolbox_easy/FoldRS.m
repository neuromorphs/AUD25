% =========================================================================%
% Author: MP Huet (mphuet@jhu.edu)
% If used, please cite:
% Huet & Elhilali (2025), bioRxiv, https://doi.org/10.1101/2025.05.22.655464
% =========================================================================

function R_combined = FoldRS(R, omegas_t, omegas_f)
    % Define scale and rate vectors
    rate = omegas_t;
    scale = omegas_f;

    % Check if the size of R along the appropriate dimensions matches the rate and scale vectors
    if size(R, 1) ~= length(scale) || size(R, 2) ~= 2 * length(rate)
        error('The input matrix R does not have the correct size. Ensure it matches the fixed rate and scale vectors.');
        return;
    end

    % Generalize folding process for any matrix with dimensions greater than or equal to 2
    num_dims = length(size(R));
    
    % Dynamically slice the left and right parts along the last dimension (rate)
    if num_dims == 3
        % For a 3D matrix [scale, 2*rate, other]
        R_left = R(:, 1:length(rate), :);
        R_right = R(:, length(rate)+1:end, : );
    elseif num_dims == 4
        % For a 4D matrix [scale, 2*rate, other1, other2]
        R_left = R(:, 1:length(rate), :, :);
        R_right = R(:, length(rate)+1:end, :, :);
    elseif num_dims == 2
        % For a 4D matrix [scale, 2*rate, other1, other2]
        R_left = R(:, 1:length(rate));
        R_right = R(:, length(rate)+1:end);    
    else
        error('The function supports only 2D, 3D or 4D matrices.');
        return;
    end

    % Flip the left part along the rate dimension (last dimension for rate)
    R_left = flip(R_left, 2);  % Flip along the rate dimension (second dimension)

    % Combine left and right parts by averaging
    R_combined = (R_left + R_right) / 2;

end
