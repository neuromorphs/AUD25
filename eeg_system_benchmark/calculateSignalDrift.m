function drift = calculateSignalDrift(eegData)
    % Calculate signal drift over time
    data = eegData.data;
    fs = eegData.fs;
    
    % Calculate trend in each channel
    timeVec = (0:size(data, 2)-1) / fs / 60; % in minutes
    
    channelDrifts = zeros(size(data, 1), 1);
    for ch = 1:size(data, 1)
        p = polyfit(timeVec, data(ch, :), 1);       % Do polynomial curve fitting
        channelDrifts(ch) = p(1); % slope in μV/min
    end
    
    drift = mean(abs(channelDrifts));
end