function snrResults = calculateSNR(eegData, varargin)
% CALCULATE_SNR_IMPROVED - Multiple methods for EEG SNR calculation
%
% USAGE:
%   snr = calculateSNR(eegData)
%   snr = calculateSNR(eegData, 'method', 'baseline')
%
% METHODS:
%   'spectral'  - Signal vs noise bands (default)
%   'baseline'  - Pre-stimulus vs post-stimulus (requires events)
%   'erp'       - ERP signal vs residual noise (requires events)
%   'all'       - Calculate all methods

% Parse inputs
p = inputParser;
addParameter(p, 'method', 'spectral', @ischar);
addParameter(p, 'signalBand', [1 30], @isnumeric);       % Hz
addParameter(p, 'noiseBand', [50 100], @isnumeric);      % Hz
addParameter(p, 'baselineWindow', [-0.2 0], @isnumeric); % seconds
addParameter(p, 'signalWindow', [0.1 0.5], @isnumeric);  % seconds
parse(p, varargin{:});

data = eegData.data;
fs = eegData.fs;
snrResults = struct();

%% SPECTRAL SNR (Signal vs Noise Frequency Bands)
if strcmp(p.Results.method, 'spectral') || strcmp(p.Results.method, 'all')
    fprintf('Calculating spectral SNR...\n');
    
    % Filter signal band (e.g., 1-30 Hz for EEG signal)
    signalBand = p.Results.signalBand;
    [b_sig, a_sig] = butter(4, signalBand/(fs/2), 'bandpass');
    signalData = filtfilt(b_sig, a_sig, data')';
    signalPower = mean(var(signalData, [], 2));
    
    % Filter noise band (e.g., 50-100 Hz for electrical noise signal)
    noiseBand = p.Results.noiseBand;
    [b_noise, a_noise] = butter(4, noiseBand/(fs/2), 'bandpass');
    noiseData = filtfilt(b_noise, a_noise, data')';
    noisePower = mean(var(noiseData, [], 2));
    
    snrResults.spectral_SNR_dB = 10 * log10(signalPower / noisePower);
    snrResults.spectral_signalPower = signalPower;
    snrResults.spectral_noisePower = noisePower;
end

%% BASELINE SNR (Pre-stimulus vs Post-stimulus)
if (strcmp(p.Results.method, 'baseline') || strcmp(p.Results.method, 'all')) && ...
   isfield(eegData, 'events')
    fprintf('Calculating baseline SNR...\n');
    
    events = eegData.events;
    if isstruct(events)
        eventSamples = events.sample;
    else
        eventSamples = events;
    end
    
    baselineWindow = p.Results.baselineWindow;
    signalWindow = p.Results.signalWindow;
    
    baselineSamples = round(baselineWindow * fs);
    signalSamples = round(signalWindow * fs);
    
    baselinePower = [];
    signalPower = [];
    
    for i = 1:length(eventSamples)
        eventIdx = eventSamples(i);
        
        % Extract baseline period
        baselineStart = eventIdx + baselineSamples(1);
        baselineEnd = eventIdx + baselineSamples(2);
        
        % Extract signal period
        signalStart = eventIdx + signalSamples(1);
        signalEnd = eventIdx + signalSamples(2);
        
        % Check bounds
        if baselineStart > 0 && signalEnd <= size(data, 2)
            baselineData = data(:, baselineStart:baselineEnd);
            signalData = data(:, signalStart:signalEnd);
            
            baselinePower = [baselinePower; var(baselineData, [], 2)];
            signalPower = [signalPower; var(signalData, [], 2)];
        end
    end
    
    avgBaselinePower = mean(baselinePower, 'all');
    avgSignalPower = mean(signalPower, 'all');
    
    snrResults.baseline_SNR_dB = 10 * log10(avgSignalPower / avgBaselinePower);
    snrResults.baseline_signalPower = avgSignalPower;
    snrResults.baseline_noisePower = avgBaselinePower;
end

%% ERP SNR (Event-Related Potential vs Residual Noise)
if (strcmp(p.Results.method, 'erp') || strcmp(p.Results.method, 'all')) && ...
   isfield(eegData, 'events')
    fprintf('Calculating ERP SNR...\n');
    
    events = eegData.events;
    if isstruct(events)
        eventSamples = events.sample;
    else
        eventSamples = events;
    end
    
    % Extract epochs
    epochTime = [-0.2 0.8]; % -200ms to 800ms
    epochSamples = round(epochTime * fs);
    
    epochs = [];
    for i = 1:length(eventSamples)
        eventIdx = eventSamples(i);
        startIdx = eventIdx + epochSamples(1);
        endIdx = eventIdx + epochSamples(2);
        
        if startIdx > 0 && endIdx <= size(data, 2)
            epochs(:, :, end+1) = data(:, startIdx:endIdx);
        end
    end
    
    if size(epochs, 3) > 1
        % Calculate average ERP
        avgERP = mean(epochs, 3);
        
        % Calculate residual noise (individual trials - average)
        residualNoise = epochs - repmat(avgERP, [1, 1, size(epochs, 3)]);
        
        % Signal power = variance of average ERP
        erpSignalPower = var(avgERP, [], 2);
        
        % Noise power = variance of residual
        erpNoisePower = var(residualNoise, [], [2, 3]);
        
        snrResults.erp_SNR_dB = 10 * log10(mean(erpSignalPower) / mean(erpNoisePower));
        snrResults.erp_signalPower = mean(erpSignalPower);
        snrResults.erp_noisePower = mean(erpNoisePower);
    end
end


%% Summary
if strcmp(p.Results.method, 'all')
    % Calculate overall SNR score
    snrValues = [];
    if isfield(snrResults, 'spectral_SNR_dB')
        snrValues(end+1) = snrResults.spectral_SNR_dB;
    end
    if isfield(snrResults, 'baseline_SNR_dB')
        snrValues(end+1) = snrResults.baseline_SNR_dB;
    end
    if isfield(snrResults, 'erp_SNR_dB')
        snrValues(end+1) = snrResults.erp_SNR_dB;
    end

    snrResults.overall_SNR_dB = mean(snrValues);
    snrResults.snr_consistency = std(snrValues);
    
    % Display results
    fprintf('\n=== SNR Results ===\n');
    if isfield(snrResults, 'spectral_SNR_dB')
        fprintf('Spectral SNR: %.2f dB\n', snrResults.spectral_SNR_dB);
    end
    if isfield(snrResults, 'baseline_SNR_dB')
        fprintf('Baseline SNR: %.2f dB\n', snrResults.baseline_SNR_dB);
    end
    if isfield(snrResults, 'erp_SNR_dB')
        fprintf('ERP SNR: %.2f dB\n', snrResults.erp_SNR_dB);
    end

    fprintf('Overall SNR: %.2f ± %.2f dB\n', snrResults.overall_SNR_dB, snrResults.snr_consistency);
end

end
