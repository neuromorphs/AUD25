function jitter = calculateTimingJitter(eegData)
    % Calculate timing jitter in triggers
    triggers = eegData.triggers;
    
    % Calculate inter-trigger intervals
    iti = diff(triggers);
    
    % Expected ITI (assuming regular presentation)
    expectedITI = median(iti);
    
    % Jitter as standard deviation of ITI
    jitter = std(iti - expectedITI) / eegData.fs * 1000; % in milliseconds
end