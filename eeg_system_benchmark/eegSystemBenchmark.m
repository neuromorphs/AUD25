function benchmarkResults = eegSystemBenchmark(eegData, systemInfo, varargin)
% Benchmarking test battery for different EEG systems
%
% Use:
%   results = eegSystemBenchmark(eegData, systemInfo)
%   results = eegSystemBenchmark(eegData, systemInfo, 'param', value, ...)
%
% Inputs:
%   eegData    - Structure with fields:
%                .data:     [channels x samples/time x trials] EEG data
%                .fs:       sampling frequency (Hz)
%                .events:   [1 x trials] event markers/triggers
%                .chanlocs: channel location structure (optional)
%                .impedances (over time?) if available
%   
%   systemInfo - Structure with fields:
%                .name:     system name string
%                .type:     'wet', 'dry' or 'semidry' electrodes
%                .channels: number of channels
%                .wireless: true/false
%
% parameters for defining the tests:
%   'plotResults'     - Generate plots          (default: true)
%   'saveResults'     - Save results to file    (default: true)
%   'testSignal'      - Include test signal analysis (default: true)
%   'auditoryERP'     - Include auditory ERP analysis (default: true)
%   'artifactTest'    - Include artifact rejection test (default: true)
%   'frequencyTest'   - Include frequency response test (default: true)
%
% OUTPUT:
%   benchmarkResults - Structure containing all benchmark metrics

% Parse input arguments
p = inputParser;
addParameter(p, 'plotResults', true, @islogical);
addParameter(p, 'saveResults', true, @islogical);
addParameter(p, 'testSignal', true, @islogical);
addParameter(p, 'auditoryERP', true, @islogical);
addParameter(p, 'artifactTest', true, @islogical);
addParameter(p, 'frequencyTest', true, @islogical);
addParameter(p, 'refSystem', [], @isstruct);
parse(p, varargin{:});

% Initialize results structure
benchmarkResults = struct();
benchmarkResults.systemInfo = systemInfo;
benchmarkResults.timestamp = datetime('now');

fprintf('=== EEG System Benchmark: %s ===\n', systemInfo.name);

%% 1. METRIC FOR BASIC SIGNAL QUALITY
fprintf('1. Analyzing basic signal quality...\n');

% STILL TODO:
% Signal-to-noise ratio - 3 way to calculate in function
snr = calculateSNR(eegData,'method','spectral');   % 'spectral' (default),'baseline', 'erp','all'
benchmarkResults.signalQuality.SNR_dB = snr;

% Baseline activity level --> Should maybe do noise instead/additionally? How to define noise?
overallLevel = calculateOverallLevel(eegData);
benchmarkResults.signalQuality.noiseLevel_uV = overallLevel;

% Channel correlation (higher correlation may indicate cross-talk)
channelCorr = calculateChannelCorrelation(eegData);
benchmarkResults.signalQuality.channelCorrelation = channelCorr;

% Impedance stability (if impedance data available) 
% - can we implement to data over time?
if isfield(eegData, 'impedance')
    impStability = std(eegData.impedance, [], 2);
    benchmarkResults.signalQuality.impedanceStability_kOhm = impStability;
end

%% 2. FREQUENCY RESPONSE ANALYSIS
if p.Results.frequencyTest
    fprintf('2. Testing frequency response...\n');
    
    % Power spectral density for correlation of the systems (?)
    [psd, freqs] = calculatePSD(eegData);
    benchmarkResults.frequencyResponse.PSD = psd;
    benchmarkResults.frequencyResponse.frequencies = freqs;
    
    % Frequency band power
    bandPowers = calculateBandPowers(eegData);
    benchmarkResults.frequencyResponse.bandPowers = bandPowers;
    
    % 50 (EU)/60(US) Hz noise assessment
    % Logic: minimal power line interference peaks
    lineNoise = assessLineNoise(eegData);
    benchmarkResults.frequencyResponse.lineNoise_dB = lineNoise;
end

%% 3. TEMPORAL PRECISION TESTS
fprintf('3. Evaluating temporal precision...\n');

% Event-related potential analysis (if events available)
if isfield(eegData, 'events') && ~isempty(eegData.events)
    erpMetrics = analyzeERP(eegData);
    benchmarkResults.temporalPrecision.ERP = erpMetrics;
end

% Timing jitter analysis
if isfield(eegData, 'triggers')
    jitter = calculateTimingJitter(eegData);
    benchmarkResults.temporalPrecision.jitter_ms = jitter;
end

%% 5. ARTIFACT REJECTION PERFORMANCE
if p.Results.artifactTest
    fprintf('5. Testing artifact rejection...\n');
    % ToDo: Change to ICA?
    artifactMetrics = assessArtifactRejection(eegData);
    benchmarkResults.artifactRejection = artifactMetrics;
end

%% 6. SYSTEM STABILITY TESTS
fprintf('6. Evaluating system stability...\n');

% Signal drift over time
drift = calculateSignalDrift(eegData);
benchmarkResults.stability.drift_uV_per_min = drift;

% Channel dropout detection - ToDo: usage of eeglab for rejection instead?
dropouts = detectChannelDropouts(eegData);
benchmarkResults.stability.channelDropouts = dropouts;

%% 7. COMPARATIVE ANALYSIS - TODO: Reference system? Cross-somparison metrics?
if ~isempty(p.Results.refSystem)
    fprintf('7. Comparing with reference system...\n');
    
    comparison = compareWithReference(benchmarkResults, p.Results.refSystem);
    benchmarkResults.comparison = comparison;
end

%% 8. GENERATE OVERALL SCORE
benchmarkResults.overallScore = calculateOverallScore(benchmarkResults);

%% VISUALIZATION
if p.Results.plotResults
    generateBenchmarkPlots(benchmarkResults, eegData);
end

%% SAVE RESULTS
if p.Results.saveResults
    filename = sprintf('EEG_Benchmark_%s_%s.mat', ...
        systemInfo.name, datestr(now, 'yyyymmdd_HHMMSS'));
    save(filename, 'benchmarkResults');
    fprintf('Results saved to: %s\n', filename);
end

fprintf('=== Benchmark Complete ===\n');
fprintf('Overall Score: %.2f/100\n', benchmarkResults.overallScore);

end
