clear

addpath lib\utilsGui\
addpath lib\utilsGui\miditoolbox\

stimFolder = './stimuli/trials';

wavstims = dir([stimFolder,'/*.wav']);
[~, reindex] = sort( str2double( regexp( {wavstims.name}, '\d+', 'match', 'once' )));
wavstims = wavstims(reindex);
wavstims = {wavstims(:).name}';
wavstims = cellfun(@(x)[stimFolder,'/',x],wavstims,'UniformOutput',false);

midistims = dir([stimFolder,'/*.mid']);
midistims = midistims(reindex);
midistims = {midistims(:).name}';
midistims = cellfun(@(x)[stimFolder,'/',x],midistims,'UniformOutput',false);

switchtimes = dir([stimFolder,'/info/*.csv']);
switchtimes = switchtimes(reindex);
switchtimes = {switchtimes(:).name}';
switchtimes = cellfun(@(x)[stimFolder,'/info/',x],switchtimes,'UniformOutput',false);

nTrl = length(midistims);

stim.names = {'Envelopes','Note Onsets','Note Surprise','Note Entropy',...
    'Switch Times','Participant Ratings','Ratings Derivative'};

stim.condIdxs = ones(1,nTrl);
stim.fs = 512;

stim.stimIdxs = 1:nTrl;



surpsStimIdx = [2 3];

iSurprises = load("stimuli\trials\MT_surprises.mat");
% surps = struct2cell(iSurprises.surprises);
% surps = cellfun(@(x)x(surpsStimIdx,:),surps,UniformOutput=false);
% ents = struct2cell(iSurprises.entropies);
% ents = cellfun(@(x)x(surpsStimIdx,:),ents,UniformOutput=false);

% boths = {};
% 
% for i = 1:length(ents)
%     e = ents{i};
%     s = surps{i};
%     boths{i} = [s; e];
% end

% intervalSurprises = cell2struct(boths',fieldnames(iSurprises.entropies));
[12 19];
for iTrl = 1:nTrl%1:nTrl
    [audio,fs] = audioread(wavstims{iTrl});

    [Onsets,pitches] = getOnsetfromMidi(midistims{iTrl},stim.fs,0.01);

     %%%%%%  FEATURE SET 1: ENVELOPE
    % Feature row (e.g., envelope = 1st row; note onset = 2nd row)
    featIdx = 1;
    % Take only channel one (mono not stereo) and extract envelope
    env = abs(hilbert(audio(:,1)));
    % Downsample envelope
    envDown = resample(env,stim.fs,fs,0);
    stim.data{featIdx,iTrl} = envDown;

    %%%%%%  FEATURE SET 2: NOTE ONSET
    featIdx = 2;
    % Onsets need to be padded as getOnsetFromMidi ends vector as soon
    % as the last onset happens
    onsetsPadLenght = length(envDown) - length(Onsets);
    if onsetsPadLenght < 0
        1+1;
    end
    Onsets = [Onsets; zeros(onsetsPadLenght,1)];

    stim.data{featIdx,iTrl} = Onsets;

    %%%%%% FEATURE SET 3: NOTE SURPRISE
    featIdx = 3;

    songName = wavstims{iTrl};
    songName = strsplit(songName,'/');
    songName = songName{4};
    songName = strsplit(songName,'.');
    songName = songName{1};
    songName = ['generated_',songName];
    surprises = iSurprises.surprises.(songName);
    entropies = iSurprises.entropies.(songName);

    % surps = Onsets;
    surps = Onsets;
   
    surps(surps == 1) = geomean(surprises(surpsStimIdx,:));
    disp(sum(surps))
    stim.data{featIdx,iTrl} = surps;

    %%%%%% FEATURE SET 4: NOTE ENTROPY
    featIdx = 4;

    entrops = Onsets;
    try
    entrops(entrops~=0) = entropies(surpsStimIdx,:)';
    catch ME
        disp(iTrl)
    end
    stim.data{featIdx,iTrl} = entrops;

    %%%%%% FEATURE SET 5: SWITCH TIMES
    featIdx = 5;

    trlSwitches = zeros(length(Onsets),1);
    trlSwitchTimes = readmatrix(switchtimes{iTrl});
    trlSwitchTimes = round(trlSwitchTimes(2:end,2)*stim.fs);
    trlSwitches(trlSwitchTimes) = 1;

    stim.data{featIdx,iTrl} = trlSwitches;
end

save("dataCND\pre_dataStim.mat","stim")
save("dataCND\dataStim.mat","stim")
