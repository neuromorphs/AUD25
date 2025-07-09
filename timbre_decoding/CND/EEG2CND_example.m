clear
addpath lib\eeglab2024.0\
eeglab nogui
addpath lib\utilsJohn\

raw_folder = './raw/';

origFs = 1000;
downFs = 100;

folderNames = dir(raw_folder);
folderNames = folderNames(3:end);
[~, reindex] = sort( str2double( regexp( {folderNames.name}, '\d+', 'match', 'once' )));
folderNames = folderNames(reindex);

% First get all the wavs
stimFolder = './stimuli/trials';
stims = dir([stimFolder,'/*.wav']);
 [~, reindex] = sort( str2double( regexp( {stims.name}, '\d+', 'match', 'once' )));
stims= stims(reindex) ;
stims = {stims(:).name}';
stims = cellfun(@(x)[stimFolder,'/',x],stims,'UniformOutput',false);
stims = cellfun(@(x)round((length(audioread(x))/44100)*512),stims,'UniformOutput',true);
%stims = stims(participantTrialPerm);

stimTimesFolder = './stimuli/trials/info';
stimTimes = dir([stimTimesFolder,'/*.csv']);
 [~, reindex] = sort( str2double( regexp( {stimTimes.name}, '\d+', 'match', 'once' )));
stimTimes= stimTimes(reindex) ;
stimTimes = {stimTimes(:).name}';
stimTimes = cellfun(@(x)[stimTimesFolder,'/',x],stimTimes,'UniformOutput',false);
stimTimes = cellfun(@(x) readtable(x),stimTimes,'UniformOutput',false);
stimTimes = cellfun(@(x) round(table2array(x(:,2))*512),stimTimes,'UniformOutput',false);

load("scripts_mat\chanlocs64.mat")
 
%%
for i = 11
    folder = [folderNames(i).folder,'\',folderNames(i).name,'\'];
    eegFileName = dir([folder,'*.eeg']);
    vhdiFileName = dir([folder,'*.vhdi']);
    eegFileName = eegFileName(1).name;
    
    disp(eegFileName)
    load([folder,'info.mat']);

    % First do the EEG
    [eeg_raw,trigs] = pop_readbv([folder,eegFileName]);

    eeg_raw = double(eeg_raw(1:66,:));

    trigs=trigs-min(trigs);
    trigs(trigs>256) = trigs(trigs>256)-min(trigs(trigs>256));

    trialStarts = find(diff(trigs)==1);

    if i == 2
        trialStarts = trialStarts(repmat([false,true(1,6)],1,5));
    elseif i == 11
        trialStarts = trialStarts(1:24);
    end

    buttonPresses = find(diff(trigs)==3);
    buttonPresses(buttonPresses < 0) = 0;

    eeg.dataType = 'eeg';
    eeg.deviceName = 'Biosemi';
    eeg.data = {};
    eeg.fs = 512;
    eeg.origTrialPosition = info.trialOrder;
    eeg.extChan = {};
    eeg.cndVersion = 1;
    eeg.chanlocs = chanlocs;

    EEG_Sections = {};
    mastoid_sections = {};
    all_buttonPresses = {};
    all_pieceSwitches = {};

    nTrials = length(trialStarts);

    for j = 1:nTrials
        start = trialStarts(j)+512;
        fin = start + stims(info.trialOrder(j));

        EEG_Sections = [EEG_Sections,transpose(eeg_raw(1:64,start:fin))];
        mastoid_sections = [mastoid_sections,transpose(eeg_raw(65:66,start:fin))];
        
        
        trial_buttonPresses = zeros(1,fin-start+1);
        trial_buttonPresses(buttonPresses((buttonPresses>start & buttonPresses<fin))-start) = 1;
        all_buttonPresses = [all_buttonPresses,trial_buttonPresses];

        trial_pieceSwitches = zeros(1,fin-start+1);
        trial_pieceSwitches(stimTimes{info.trialOrder(j)}(2:end)) = 1;

        all_pieceSwitches = [all_pieceSwitches,trial_pieceSwitches'];
    end

    eeg.data = EEG_Sections;
    eeg.extChan{1} = struct();
    eeg.extChan{1}.data = mastoid_sections;
    eeg.extChan{1}.description = 'mastoids';

    eeg.extChan{2} = struct();
    eeg.extChan{2}.data = all_buttonPresses;
    eeg.extChan{2}.description = 'Button Presses';

    eeg.extChan{3} = struct();
    eeg.extChan{3}.data = all_pieceSwitches;
    eeg.extChan{3}.description = 'Piece Switches';
    %%

    save(['dataCND\dataSub',int2str(i),'.mat'],"eeg")
end
