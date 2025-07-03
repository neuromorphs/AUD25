%% Multi-Trial Two-Audio + Mid-Trial Arrow Flip with Logging
clc; clear; close all;

%% Change to script directory
tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));

%% List of trials to run
trials     = [16];
numTrials  = numel(trials);

%% Build file lists
fileList1 = arrayfun(@(tr) ...
    sprintf("Stimuli/Trial_%d_Conv_1.wav", tr), ...
    trials, 'UniformOutput', false);
fileList2 = arrayfun(@(tr) ...
    sprintf("Stimuli/Trial_%d_Conv_2.wav", tr), ...
    trials, 'UniformOutput', false);

%% Preallocate log vectors
initialDir = strings(numTrials,1);
secondDir  = strings(numTrials,1);

%% Audio I/O setup
frameLength  = 1024;
deviceWriter = audioDeviceWriter('SampleRate', 44100);
% deviceWriter.Device = "Philosopher’s Stone";
deviceWriter.Device = "WH-1000XM4";
% deviceWriter.Device = "MacBook Pro Speakers";


%% UIFigure for the arrow (reuse across trials)
promptFig = uifigure( ...
  'Name',    "Direction Cue", ...
  'Color',   'white', ...
  'Position',[100 100 600 600], ...
  'Visible', 'off' ...
);

for idx = 1:numTrials
    %% File paths for this trial
    file1 = fileList1{idx};
    file2 = fileList2{idx};

    %% Read full audio to compute midpoint
    [a1, fs1] = audioread(file1);
    [a2, fs2] = audioread(file2);
    assert(fs1==fs2, "Sampling rates must match");
    nSamples  = min(numel(a1), numel(a2));
    midSample = floor(nSamples/2);

    %% Set up readers
    reader1 = dsp.AudioFileReader(file1, 'SamplesPerFrame', frameLength);
    reader2 = dsp.AudioFileReader(file2, 'SamplesPerFrame', frameLength);

    %% Pick random initial direction (1=Left, 2=Right)
    dir     = randi(2);
    swapped = false;

    %% Log the initial direction
    if dir==1
        initialDir(idx) = "Left";
    else
        initialDir(idx) = "Right";
    end

    %% Draw the initial arrow
    clf(promptFig)
    ax = uiaxes(promptFig, 'Position',[50 50 500 500]);
    ax.XAxis.Visible = 'off';  ax.YAxis.Visible = 'off';
    img = uiimage(promptFig, 'Position',ax.Position, 'ScaleMethod','fit');
    img.ImageSource = "ArrowDesign/" + lower(initialDir(idx)) + "Arrow.jpeg";

    %% Show & force render
    promptFig.Visible = "on";
    drawnow;  pause(0.05);

    %% Playback loop with midpoint flip
    sampleCount = 0;
    while ~(isDone(reader1) || isDone(reader2))
        y1 = reader1();
        y2 = reader2();
        sampleCount = sampleCount + size(y1,1);

        % Flip arrow at midpoint
        if ~swapped && sampleCount >= midSample
            swapped = true;
            % Determine second direction (the opposite)
            if initialDir(idx)=="Left"
                secondDir(idx) = "Right";
            else
                secondDir(idx) = "Left";
            end
            img.ImageSource = "ArrowDesign/" + lower(secondDir(idx)) + "Arrow.jpeg";
            drawnow;
        end

        % Route streams based on original dir
        if dir==1
            out = [y1, y2];
        else
            out = [y2, y1];
        end
        deviceWriter(out);

        drawnow limitrate;
    end

    %% Hide arrow and release readers
    promptFig.Visible = "off";  drawnow;
    release(reader1);
    release(reader2);

    pause(1);  % inter-trial interval
end

%% Save log to CSV
logTable = table( ...
    trials(:), ...
    string(fileList1(:)), ...
    string(fileList2(:)), ...
    initialDir, ...
    secondDir, ...
    'VariableNames', ...
    {'Trial','File1','File2','FirstHalfArrow','SecondHalfArrow'} ...
);
writetable(logTable, "trial_arrow_log.csv");

%% Final cleanup
release(deviceWriter);
close(promptFig);
