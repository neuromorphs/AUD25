function lineNoise = assessLineNoise(eegData)
    % Assess 50 (EU)/60 (US) Hz line noise
    data = eegData.data;
    fs = eegData.fs;
    
    % Calculate PSD
    [psd, freqs] = pwelch(data', [], [], [], fs);
    psd = mean(psd, 2);
    
    % Find 50 Hz and 60 Hz components
    [~, idx50] = min(abs(freqs - 50));
    [~, idx60] = min(abs(freqs - 60));
    
    % Compare to nearby frequencies - how many?
    baselineIdx = [idx50-2:idx50-1, idx50+1:idx50+2, idx60-2:idx60-1, idx60+1:idx60+2];
    baselinePower = mean(psd(baselineIdx));
    
    linePower = max(psd(idx50), psd(idx60));
    lineNoise = 10 * log10(linePower / baselinePower);
end