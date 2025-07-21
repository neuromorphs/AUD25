%% Multi-Trial Two-Audio + Mid-Trial Arrow Flip with Top/Bottom Fade and Logging
clc; clear; close all;
rng(0);

%% Change to script directory
tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));

%% List of trials to run
trials     = 1:19;
numTrials  = numel(trials);

% pre-generate the left/right assignment so it stays the same each pass
dirSequence = randi(2, numTrials, 1);

%% Build file lists
fileList1 = arrayfun(@(tr) ...
    sprintf("Stimuli/speech_only_short_22kHz/jane_eyre_05_part%d.wav", tr), ...
    trials, 'UniformOutput', false);
fileList2 = arrayfun(@(tr) ...
    sprintf("Stimuli/piano_only_long_cropped_22kHz/piano_4_1_22050Hz_part%d.wav", tr), ...
    trials, 'UniformOutput', false);

%% Preallocate log vectors
initialDir = strings(numTrials,1);
secondDir  = strings(numTrials,1);

%% Audio I/O setup
frameLength  = 1024;
deviceWriter = audioDeviceWriter('SampleRate', 22050);
% deviceWriter.Device = "Philosopher’s Stone";
% deviceWriter.Device = "WH-1000XM4";
deviceWriter.Device = "MacBook Pro Speakers";
% deviceWriter.Device = "Primary Sound Driver";

%% Compute centered position for a 400×800 window
screenSize = get(groot, 'ScreenSize');    % [x0 y0 width height]
figW       = 400;
figH       = 800;
figX       = screenSize(1) + (screenSize(3) - figW)/2;
figY       = screenSize(2) + (screenSize(4) - figH)/2;

%% Create centered UIFigure
promptFig = uifigure( ...
  'Name',     "Direction Cue", ...
  'Color',    'white', ...
  'Position', [figX figY figW figH], ...
  'Visible',  'off' ...
);

% PulseWidth = 0.1;
% port = serialport('COM5', 2000000);
% write(port, 0, 'uint8');
% pause(PulseWidth);

% run two passes
for rep = 1:2

  for idx = 1:numTrials
      %% File paths for this trial
      file1 = fileList1{idx};
      file2 = fileList2{idx};

      %% Read full audio to compute midpoint & RMS for equalization
      [a1, fs1] = audioread(file1);
      [a2, fs2] = audioread(file2);
      assert(fs1==fs2, "Sampling rates must match");
      % convert to mono if necessary
      if size(a1,2) > 1, a1 = mean(a1,2); end
      if size(a2,2) > 1, a2 = mean(a2,2); end

      nSamples  = min(numel(a1), numel(a2));
      midSample = floor(nSamples/2);

      %--- Equalize overall power (RMS) between the two streams
      rms1 = sqrt(mean(a1.^2));
      rms2 = sqrt(mean(a2.^2));
      targetRMS = (rms1 + rms2)/2;
      gain1 = targetRMS / rms1;
      gain2 = targetRMS / rms2;

      %% Set up readers
      reader1 = dsp.AudioFileReader(file1, 'SamplesPerFrame', frameLength);
      reader2 = dsp.AudioFileReader(file2, 'SamplesPerFrame', frameLength);

      %% Use the fixed sequence for left/right
      dir     = dirSequence(idx);
      swapped = false;

      %% Log the initial direction on first pass only
      if rep == 1
          if dir==1
              initialDir(idx) = "Speech";
          else
              initialDir(idx) = "Music";
          end
      end

      %% Draw the initial logos top/bottom with fade
      % clf(promptFig)
      ax = uiaxes(promptFig, ...
          'Units', 'normalized', ...
          'Position', [0.1 0.1 0.8 0.8] ...  
      );
      ax.XAxis.Visible = 'off';  ax.YAxis.Visible = 'off';
      ax.Box = 'off';

      leftImgData  = imread("Images/speech.jpg");
      rightImgData = imread("Images/music.jpg");
      colormap(ax, gray);

      % Define alpha values, invert on second pass
      if rep == 1
          if dir==1
              alphaLeft  = 1.0; 
              alphaRight = 0.3;
          else
              alphaLeft  = 0.3; 
              alphaRight = 1.0;
          end
      else
          if dir==1
              alphaLeft  = 0.3; 
              alphaRight = 1.0;
          else
              alphaLeft  = 1.0; 
              alphaRight = 0.3;
          end
      end

      ax.XLim = [0 1];
      ax.YLim = [0 2];
      ax.YDir = 'reverse';

      hold(ax, 'on'); 
      hLeft  = image(ax, [0,1], [0,1], leftImgData,  'AlphaData', alphaLeft);
      hRight = image(ax, [0,1], [1,2], rightImgData, 'AlphaData', alphaRight);

      %% Show & force render
      promptFig.Visible = "on";
      drawnow;  pause(0.05);

      %% Playback loop with midpoint flip & fade
      % write(port,1,'uint8')
      sampleCount = 0;
      % write(port,0,'uint8')
      while ~(isDone(reader1) || isDone(reader2))
          y1 = reader1();
          y2 = reader2();
          if size(y1,2)>1, y1=mean(y1,2); end
          if size(y2,2)>1, y2=mean(y2,2); end

          y1 = y1 * gain1;
          y2 = y2 * gain2;

          sampleCount = sampleCount + size(y1,1);

          if ~swapped && sampleCount >= midSample
              swapped = true;
              if rep == 1
                  % first pass flip (and log secondDir)
                  if dir==1
                      % write(port,2,'uint8')
                      secondDir(idx) = "Music";
                      hLeft.AlphaData  = 0.3;
                      hRight.AlphaData = 1.0;
                  else
                      % write(port,3,'uint8')
                      secondDir(idx) = "Speech";
                      hLeft.AlphaData  = 1.0;
                      hRight.AlphaData = 0.3;
                  end
              else
                  % second pass flip (reverse highlight)
                  if dir==1
                      % write(port,3,'uint8')
                      hLeft.AlphaData  = 1.0;
                      hRight.AlphaData = 0.3;
                  else
                      % write(port,2,'uint8')
                      hLeft.AlphaData  = 0.3;
                      hRight.AlphaData = 1.0;
                  end
              end
              drawnow;
              % write(port,0,'uint8')
          end

          if dir==1
              out = [y1, y2];
          else
              out = [y2, y1];
          end
          deviceWriter(out);
          drawnow limitrate;
      end

      %% Hide logos and release readers
      promptFig.Visible = "off";  drawnow;
      release(reader1);
      release(reader2);
      pause(1);

      if ismember(idx, [10, 19, 29])
          cont = input('Do you want to continue? (1 = Yes, 0 = No): ');
          if cont ~= 1
              return;   % exit script immediately
          end
      end
  end

end

%% Save log to CSV (including filenames, rounds, left/right files, and both halves)
% Build vectors for two rounds
rounds     = [ones(numTrials,1); 2*ones(numTrials,1)];
trialCol   = [trials(:);        trials(:)];
file1Col   = [string(fileList1(:)); string(fileList1(:))];
file2Col   = [string(fileList2(:)); string(fileList2(:))];

% Determine left/right assignment based on dirSequence
leftFiles  = strings(numTrials,1);
rightFiles = strings(numTrials,1);
for i = 1:numTrials
    if dirSequence(i)==1
        leftFiles(i)  = fileList1{i};  % speech on left
        rightFiles(i) = fileList2{i};  % music on right
    else
        leftFiles(i)  = fileList2{i};  % music on left
        rightFiles(i) = fileList1{i};  % speech on right
    end
end
% Duplicate for two rounds
leftFileCol  = [leftFiles;  leftFiles];
rightFileCol = [rightFiles; rightFiles];

% First‐ and second‐half highlights
firstHalf    = [initialDir;       secondDir];
secondHalf   = [secondDir;        initialDir];

% Assemble table
logTable = table( ...
    trialCol, ...
    rounds, ...
    file1Col, ...
    file2Col, ...
    leftFileCol, ...
    rightFileCol, ...
    firstHalf, ...
    secondHalf, ...
    'VariableNames', ...
    {'Trial','Round','SpeechFile','MusicFile','LeftFile','RightFile','FirstHalfAttend','SecondHalfAttend'} ...
);

writetable(logTable, "trial_arrow_log.csv");

%% Final cleanup
release(deviceWriter);
close(promptFig);