% =========================================================================%
% Author: MP Huet (mphuet@jhu.edu)
% If used, please cite:
% Huet & Elhilali (2025), bioRxiv, https://doi.org/10.1101/2025.05.22.655464
% =========================================================================
function R_fold = PermuteFold(R, omegas_t, omegas_f)
if ndims(R) == 3
    R_permuted = permute(R, [2, 3, 1]);
    R_fold = FoldRS(R_permuted, omegas_t, omegas_f);
elseif ndims(R) == 4
    R_permuted = permute(R, [3, 4, 1, 2]);
    R_fold = FoldRS(R_permuted, omegas_t, omegas_f);
elseif ndims(R) == 2
R_fold = FoldRS(R, omegas_t, omegas_f);
end
end