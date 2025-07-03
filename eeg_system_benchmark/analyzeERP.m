function erpMetrics = analyzeERP(eegData)
    % Analyze event-related potentials
    % Currentl N1 & P2, depending on paradigm add P3?

    data = eegData.data;
    events = eegData.events;
    fs = eegData.fs;
    
    % Extract epochs around events
    epochTime = [-0.2 0.8]; % -200ms to 800ms (takane from oddball)
    epochSamples = round(epochTime * fs);
    
    epochs = [];
    for i = 1:length(events)
        startIdx = events(i) + epochSamples(1);
        endIdx = events(i) + epochSamples(2);
        
        if startIdx > 0 && endIdx <= size(data, 2)
            epochs(:, :, end+1) = data(:, startIdx:endIdx);
        end
    end
    
    % Calculate average ERP
    avgERP = mean(epochs, 3);
    
    % Find key components (simplified)
    timeVec = linspace(epochTime(1), epochTime(2), size(avgERP, 2));
    
    % N100 (80-120ms)
    n100Window = timeVec >= 0.08 & timeVec <= 0.12;
    [n100Amp, n100Idx] = min(mean(avgERP(:, n100Window), 1));
    n100Lat = timeVec(n100Window);
    n100Lat = n100Lat(n100Idx);
    
    % P200 (150-250ms)
    p200Window = timeVec >= 0.15 & timeVec <= 0.25;
    [p200Amp, p200Idx] = max(mean(avgERP(:, p200Window), 1));
    p200Lat = timeVec(p200Window);
    p200Lat = p200Lat(p200Idx);
    
    erpMetrics = struct();
    erpMetrics.N100_amplitude = n100Amp;
    erpMetrics.N100_latency = n100Lat;
    erpMetrics.P200_amplitude = p200Amp;
    erpMetrics.P200_latency = p200Lat;
    erpMetrics.avgERP = avgERP;
    erpMetrics.timeVector = timeVec;
end
