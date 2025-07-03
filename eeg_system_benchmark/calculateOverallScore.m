function overallScore = calculateOverallScore(results)
    % Calculate overall benchmark score (0-100)
    
    % Weighted scoring
    weights = struct();
    weights.SNR = 0.25;
    weights.noise = 0.20;
    weights.stability = 0.20;
    weights.artifacts = 0.15;
    weights.temporal = 0.20;
    
    % Normalize metrics to 0-100 scale
    snrScore = min(100, max(0, (results.signalQuality.SNR_dB + 10) * 5));
    noiseScore = min(100, max(0, 100 - results.signalQuality.noiseLevel_uV * 2));
    stabilityScore = min(100, max(0, 100 - results.stability.drift_uV_per_min * 10));
    artifactScore = min(100, max(0, 100 - results.artifactRejection.totalArtifacts / 1000));
    
    % Temporal score (based on ERP quality if available)
    if isfield(results, 'temporalPrecision') && isfield(results.temporalPrecision, 'ERP')
        temporalScore = 75; % Placeholder - would be based on ERP quality
    else
        temporalScore = 50;
    end
    
    overallScore = weights.SNR * snrScore + ...
                   weights.noise * noiseScore + ...
                   weights.stability * stabilityScore + ...
                   weights.artifacts * artifactScore + ...
                   weights.temporal * temporalScore;
end