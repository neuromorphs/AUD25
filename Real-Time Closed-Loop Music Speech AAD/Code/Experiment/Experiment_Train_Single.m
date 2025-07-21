%% Sequential Speech & Music Playback with Logo Display
clc; clear; close all;

%% Change to script directory
tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));

%% List of speech and music files (all ~10 min each)
speechFiles = { ...
    "Stimuli/speech_only_long_22kHz/jane_eyre_05_part1.wav", ...
    "Stimuli/speech_only_long_22kHz/jane_eyre_05_part2.wav", ...
};
musicFiles = { ...
    "Stimuli/piano_only_long_22kHz/piano_4_1_22050Hz.wav", ...
    "Stimuli/piano_only_long_22kHz/piano_4_2_22050Hz.wav", ...
};

%% Interleave speech and music into one sequence
numPairs = min(numel(speechFiles), numel(musicFiles));
allFiles  = cell(1, numPairs*2);
fileTypes = strings(1, numPairs*2);
for k = 1:numPairs
    allFiles{2*k-1}   = speechFiles{k};
    fileTypes(2*k-1)  = "Speech";
    allFiles{2*k}     = musicFiles{k};
    fileTypes(2*k)    = "Music";
end

%% Audio I/O setup
frameLength  = 1024;
deviceWriter = audioDeviceWriter('SampleRate',22050);
% deviceWriter.Device = "Philosopher’s Stone";
% deviceWriter.Device = "WH-1000XM4";
deviceWriter.Device = "MacBook Pro Speakers";
% deviceWriter.Device = "Primary Sound Driver";

%% Compute centered position for a 800×400 window
screenSize = get(groot,'ScreenSize');    % [left bottom width height]
figW       = 400;
figH       = 400;
figX       = screenSize(1) + (screenSize(3)-figW)/2;
figY       = screenSize(2) + (screenSize(4)-figH)/2;

%% Create centered UIFigure
promptFig = uifigure( ...
  'Name',     "Now Playing", ...
  'Color',    'white', ...
  'Position', [figX figY figW figH], ...
  'Visible',  'off' ...
);

% PulseWidth = 0.1;
% port = serialport('COM5', 2000000);
% write(port, 0, 'uint8');
% pause(PulseWidth);

%% Prepare axes for logo display
ax = uiaxes(promptFig, ...
    'Units','normalized', ...
    'Position',[0.1 0.1 0.8 0.8]);
ax.XAxis.Visible = 'off';
ax.YAxis.Visible = 'off';
ax.Box          = 'off';
ax.XLim         = [0 1];
ax.YLim         = [0 1];
ax.YDir         = 'reverse';
hold(ax,'on');

%% Load logo images
speechLogo = imread("Images/speech.jpg");
musicLogo  = imread("Images/music.jpg");
colormap(ax, gray);

%% Loop through all files sequentially
for i = 1:numel(allFiles)
    thisFile = allFiles{i};
    thisType = fileTypes(i);
    
    % display correct logo
    cla(ax);
    if thisType=="Speech"
        image(ax, [0 1], [0 1], speechLogo);
    else
        image(ax, [0 1], [0 1], musicLogo);
    end
    promptFig.Visible = "on";
    drawnow; pause(0.1);
    
    % play the file
    % if thisType=="Speech"
    %     write(port,2,'uint8');
    % else
    %     write(port,3,'uint8');
    % end
    reader = dsp.AudioFileReader(thisFile, 'SamplesPerFrame',frameLength);
    % write(port,0,'uint8');

    while ~isDone(reader)
        y = reader();
        if size(y,2)==1
            out = [y,y];
        else
            out = y;
        end
        deviceWriter(out);
    end
    release(reader);
    
    % hide logo between files
    promptFig.Visible = "off";
    drawnow;
    pause(1);  % inter-file interval

    % Prompt to continue
    cont = input('Do you want to continue? (1 = Yes, 0 = No): ');
    if cont ~= 1
        break;
    end

end

%% Log the playback order
order = (1:numel(allFiles))';
logTable = table( ...
    order, ...
    string(allFiles(:)), ...
    fileTypes(:), ...
    'VariableNames', {'Order','File','Type'} ...
);
writetable(logTable, "play_order_log.csv");

%% Final cleanup
release(deviceWriter);
close(promptFig);