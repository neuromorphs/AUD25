function overallLevel = calculateOVeralLevel(eegData)
    % Calculate RMS noise level in microvolts
    data = eegData.data;
    
    % Filter 1-100 frequency components
    fs = eegData.fs;
    [b, a] = butter(4, [1 100]/(fs/2), 'bandpass');
    filteredData = filtfilt(b, a, data')';
    
    % Calculate RMS across all channels
    noiseLevel = sqrt(mean(filteredData.^2, 'all'));
end