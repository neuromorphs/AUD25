% Prepare the data structure
eegData.data = curr_EEGMatrix;  % [channels x samples x trials]
eegData.fs = 500;  % sampling frequency
eegData.events = eventMarkers;  % trigger positions

% System information
systemInfo.name = 'Cognionics CGX-32';
systemInfo.type = 'dry';
systemInfo.channels = 32;
systemInfo.wireless = true;

% Run benchmark
results = eegSystemBenchmark(eegData, systemInfo);