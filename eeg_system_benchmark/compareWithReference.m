function comparison = compareWithReference(results, refResults)
    % Compare results with reference system
    % Do we have a reference system? How do we want that?
    comparison = struct();
    
    % SNR comparison
    comparison.SNR_improvement = results.signalQuality.SNR_dB - ...
        refResults.signalQuality.SNR_dB;
    
    % Noise level comparison
    comparison.noise_reduction = refResults.signalQuality.noiseLevel_uV - ...
        results.signalQuality.noiseLevel_uV;
    
    % Overall score comparison
    comparison.score_difference = results.overallScore - refResults.overallScore;
end