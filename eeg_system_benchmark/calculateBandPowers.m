function bandPowers = calculateBandPowers(eegData)
    % Calculate power (var) in standard frequency bands
    data = eegData.data;
    fs = eegData.fs;
    
    bands = struct();
    bands.delta = [0.5 4];
    bands.theta = [4 8];
    bands.alpha = [8 12];
    bands.beta = [12 30];
    bands.gamma = [30 100];
    
    bandNames = fieldnames(bands);
    bandPowers = struct();
    
    for i = 1:length(bandNames)
        band = bands.(bandNames{i});
        [b, a] = butter(4, band/(fs/2), 'bandpass');
        filteredData = filtfilt(b, a, data')';
        bandPowers.(bandNames{i}) = mean(var(filteredData, [], 2)); % V = var(A,w,dim)
    end
end