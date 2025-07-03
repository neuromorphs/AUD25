function generateBenchmarkPlots(results, eegData)
    % Generate comprehensive benchmark plots
    
    figure('Position', [100, 100, 1200, 800]);
    
    % Plot 1: Signal Quality Overview
    subplot(2, 3, 1);
    bar([results.signalQuality.SNR_dB, results.signalQuality.noiseLevel_uV]);
    set(gca, 'XTickLabel', {'SNR (dB)', 'Noise (μV)'});
    title('Signal Quality');
    
    % Plot 2: Frequency Response
    if isfield(results, 'frequencyResponse')
        subplot(2, 3, 2);
        semilogx(results.frequencyResponse.frequencies, ...
                 10*log10(results.frequencyResponse.PSD));
        xlabel('Frequency (Hz)');
        ylabel('Power (dB)');
        title('Power Spectral Density');
        grid on;
    end
    
    % Plot 3: ERP (if available)
    if isfield(results, 'temporalPrecision') && isfield(results.temporalPrecision, 'ERP')
        subplot(2, 3, 3);
        erp = results.temporalPrecision.ERP;
        plot(erp.timeVector, mean(erp.avgERP, 1));
        xlabel('Time (s)');
        ylabel('Amplitude (μV)');
        title('Average ERP');
        grid on;
    end
    
    % Plot 4: Artifact Assessment
    subplot(2, 3, 4);
    artifacts = results.artifactRejection;
    bar([artifacts.eyeBlinks, artifacts.muscleArtifacts, artifacts.channelJumps]);
    set(gca, 'XTickLabel', {'Eye Blinks', 'Muscle', 'Jumps'});
    title('Artifact Count');
    
    % Plot 5: Channel Quality
    subplot(2, 3, 5);
    if isfield(results.signalQuality, 'impedanceStability_kOhm')
        bar(results.signalQuality.impedanceStability_kOhm);
        xlabel('Channel');
        ylabel('Impedance Stability (kΩ)');
        title('Channel Impedance Stability');
    else
        text(0.5, 0.5, 'No impedance data', 'HorizontalAlignment', 'center');
        title('Channel Quality');
    end
    
    % Plot 6: Overall Score
    subplot(2, 3, 6);
    pie([results.overallScore, 100-results.overallScore], ...
        {sprintf('Score: %.1f', results.overallScore), ''});
    title('Overall Benchmark Score');
    
    sgtitle(sprintf('EEG System Benchmark: %s', results.systemInfo.name));
end