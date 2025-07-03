function channelCorr = calculateChannelCorrelation(eegData)
    % Calculate average correlation between adjacent channels    
    % Values near +1: strong positive correlation
    % Values near -1: strong negative correlation
    % Values near 0: little to no linear correlation
    % Logic: High correlations between distant electrodes might indicate 
    % noise issues, while correlations between nearby electrodes 
    % suggest good signal quality.

    data = eegData.data;
    nChans = size(data, 1);
    corrMatrix = corrcoef(data');

    % Remove diagonal and calculate mean correlation
    corrMatrix(logical(eye(nChans))) = NaN;
    channelCorr = nanmean(corrMatrix, 'all');
end
