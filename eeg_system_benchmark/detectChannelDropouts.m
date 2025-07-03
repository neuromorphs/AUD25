function dropouts = detectChannelDropouts(eegData)
    % Detect channel dropouts (flat lines or extreme values)
    data = eegData.data;
    
    dropouts = struct();
    
    % Flat line detection
    flatThreshold = 0.1; % microvolts
    flatChannels = std(data, [], 2) < flatThreshold;
    dropouts.flatChannels = find(flatChannels);
    
    % Extreme value detection
    extremeThreshold = 500; % microvolts
    extremeChannels = any(abs(data) > extremeThreshold, 2);
    dropouts.extremeChannels = find(extremeChannels);
    
    dropouts.totalDropouts = sum(flatChannels) + sum(extremeChannels);
end