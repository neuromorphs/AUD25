function [psd, freqs] = calculatePSD(eegData)
    % Calculate power spectral density
    data = eegData.data;
    fs = eegData.fs;
    
    % Average PSD across all channels
    [psd, freqs] = pwelch(data', [], [], [], fs);   % pxx = pwelch(x,window,noverlap,nfft)
    psd = mean(psd, 2);
end