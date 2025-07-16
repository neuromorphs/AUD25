% =========================================================================%
% Author: MP Huet (mphuet@jhu.edu)
% If used, please cite:
% Huet & Elhilali (2025), bioRxiv, https://doi.org/10.1101/2025.05.22.655464
% =========================================================================
function PlotRS(R, omegas_t, omegas_f, mytitle, pos,value_threshold, value_range, mycolormap)
% PlotRS - Display a rate-scale plot with optional value range for color scaling.
%
%
% Inputs:
%   R - Scale-rate matrix. Ensure that R has the correct size.
%   omegas_t - Vector of temporal frequencies.
%   omegas_f - Vector of spatial frequencies.
%   value_range - Optional two-element vector specifying the min and max
%                 values for color scaling. If empty, scaling is automatic.
%
% Outputs:
%   A visual representation of the rate-scale plot.
%
% Created by MP Huet: mphuet@jhu.edu
% Date: 06/25/2024

% Check if value_range is provided
if nargin < 8
    mycolormap = viridis;
end

if nargin < 7
    value_range = [];
end

if nargin < 6
    value_threshold = [0.95, 0.9, 0.85];
end
if nargin < 5
    pos=[0.1, 0.1, 1, 1];
end
fontsi = 8 + 2*pos(3) + 2*pos(4);

% Define scale and rate vectors
rate = omegas_t;
scale = omegas_f;

% Check if R has the correct size
if size(R, 1) ~= length(scale) || size(R, 2) ~= length(rate)
    error('The input matrix R does not have the correct size. Ensure it matches the fixed rate and scale vectors.');
    return;
end

% Interpolate and construct the matrix for visualization
rs_combined = interp2(R, 2, 'cubic');
rate_int_indices = mod(rate, 0.25) == 0;  % Indices des éléments entiers
rate_int = rate(rate_int_indices);  % Les valeurs entières
xticks = [find(rate_int_indices)];  % Positions ajustées pour l'affichage
xticklabels = arrayfun(@num2str, rate_int, 'UniformOutput', false);  % Labels pour xticks
scale_int_indices = mod(scale, 0.25) == 0;  % Indices des éléments entiers
yticks = scale(scale_int_indices);  % Les valeurs entières

% Affichage de la matrice principale
axes('Position', [pos(1) pos(2) 0.72*pos(3) 0.72*pos(4)]);
if isempty(value_range)
    imagesc([1:length(rate)], scale, rs_combined);
else
    imagesc([1:length(rate)], scale, rs_combined, value_range);
end
colormap(mycolormap);
axis xy;
set(gca, ...
    'xtick', xticks, ...
    'xticklabel', xticklabels, ...
    'ytick', yticks, ...
    'yscale', 'log', ...
    'fontsi', fontsi, ...
    'ylim', [min(scale), max(scale)]);

% Tracer les lignes de contour pour le seuil de 95
hold on;
x_axis = interp2(meshgrid(1:length(rate)), 2, 'cubic');
x_axis = x_axis(1,:);
y_axis = interp2(meshgrid(scale), 2, 'cubic');
y_axis = y_axis(1,:);
threshold = max(rs_combined, [], 'all') * value_threshold(1);
[C, h] = contour(x_axis, y_axis, rs_combined, [threshold, threshold], 'Color', 'w', 'LineWidth', 2);
threshold = max(rs_combined, [], 'all') * value_threshold(2);
[C, h] = contour(x_axis, y_axis, rs_combined, [threshold, threshold], 'Color', 'w', 'LineWidth', 1);
threshold = max(rs_combined, [], 'all') * value_threshold(3);
[C, h] = contour(x_axis, y_axis, rs_combined, [threshold, threshold], 'Color', 'w', 'LineWidth', 0.3, 'LineStyle', '--');


hold off;
drawnow;
xlabel('Temporal modulation (Hz)');
ylabel('Spectral modulation (cyc/oct)');

% Ajouter la colorbar
colorbar('Position', [pos(1)+(0.81*pos(3)) pos(2) 0.02*pos(3) 0.72*pos(4)]);  % Ajuster la position de la colorbar

% Créer la colormap et sélectionner la couleur moyenne
cmap = mycolormap;  % Charger la colormap 
mid_color = cmap(round(size(cmap, 1) / 2), :);  % Prendre la couleur au milieu
%mid_color = [0.2, 0.2, 0.2];

% Affichage du tracé de la moyenne en bas, avec inversion de l'axe Y
axes('Position', [pos(1) pos(2)+0.73*pos(4) 0.72*pos(3) 0.07*pos(4)]);
hold on;
x = x_axis;
y = mean(rs_combined, 1);
y = y-min(y);
y = y.^2.5;
fill([x, fliplr(x)], [y, zeros(1, length(y))], mid_color , 'FaceAlpha', 0.2, 'EdgeColor', 'none');  % Ombre grisée
plot(x, y, 'Color', mid_color);  % Tracé de la courbe
hold off;
set(gca, ...
    'xtick', [], ...  % Masquer les graduations de l'axe X
    'xticklabel', [], ...  % Masquer les étiquettes de l'axe X
    'ytick', [], ...  % Masquer les graduations de l'axe Y
    'YColor', 'none', ...  % Masquer la couleur de l'axe Y (rend l'axe Y invisible)
    'yticklabel', [], ...  % Masquer les étiquettes de l'axe Y
    'fontsi', fontsi);  % Inverser l'axe Y
xlim([min(xticks), max(xticks)]);  % Définir les mêmes limites pour l'axe X
title(mytitle);
% if nargin > 6
%     ylim(value_range); 
% end

% Affichage d'un tracé supplémentaire à gauche avec rotation de 90 degrés dans l'autre sens
axes('Position', [pos(1)+0.73*pos(3) pos(2) 0.07*pos(3) 0.72*pos(4)]);  % Ajuster la position pour le tracé
hold on;
x = mean(rs_combined, 2);
x = x';
x = x-min(x);
x = x.^2.5;
y = y_axis;

% Remplacer les zéros par une petite valeur positive compatible avec l'échelle logarithmique
small_value = min(y(y > 0)) * 0.1;  % Une petite valeur positive basée sur le minimum de y
fill([x, fliplr(x)], [y, small_value * ones(1, length(y))], mid_color, 'FaceAlpha', 0.2, 'EdgeColor', 'none');  % Ombre grisée
plot(x, y, 'Color', mid_color);  % Tracé de la courbe
hold off;
set(gca, ...
    'xtick', [], ...  % Masquer les graduations de l'axe X
    'xticklabel', [], ...  % Masquer les étiquettes de l'axe X
    'yscale', 'log', ...  % Utiliser l'échelle logarithmique pour l'axe Y
    'XColor', 'none', ...  % Masquer la couleur de l'axe Y (rend l'axe Y invisible)
    'ylim', [min(scale), max(scale)],...
    'fontsi', fontsi, ...
    'ytick', []);  % Afficher les graduations de l'axe Y
ylim([min(scale), max(scale)]);  % Définir les mêmes limites pour l'axe Y
% if nargin > 6
%     xlim(value_range); 
% end

end
