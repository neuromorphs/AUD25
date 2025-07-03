function artifactMetrics = assessArtifactRejection(eegData)
    % Assess artifact rejection capabilities
    % Use ICA with ICAlabel instead??
    data = eegData.data;
    
    % Detect various types of artifacts
    % Eye blinks (high amplitude, frontal channels)
    eyeBlinkThreshold = 100; % microvolts
    eyeBlinks = sum(abs(data(1:4, :)) > eyeBlinkThreshold, 'all');
    
    % Muscle artifacts (high frequency, high amplitude)
    fs = eegData.fs;
    [b, a] = butter(4, 30/(fs/2), 'high');
    muscleData = filtfilt(b, a, data')';
    muscleArtifacts = sum(abs(muscleData) > 50, 'all');
    
    % Channel jumps/dropouts
    channelJumps = sum(abs(diff(data, 1, 2)) > 200, 'all');
    
    artifactMetrics = struct();
    artifactMetrics.eyeBlinks = eyeBlinks;
    artifactMetrics.muscleArtifacts = muscleArtifacts;
    artifactMetrics.channelJumps = channelJumps;
    artifactMetrics.totalArtifacts = eyeBlinks + muscleArtifacts + channelJumps;
end