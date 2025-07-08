% Cognition and Natural Sensory Processing (CNSP) Workshop
% Example 1 - Forward TRF
%
% This example script loads and preprocesses a publicly available dataset
% (you can use any of the dataset in the CNSP resources). Then, the script
% runs a typical forward TRF analysis.
%
% Note:
% This code was written with the assumption that all subjects were
% presented with the same set of stimuli. Hence, we use a single stimulus
% file (dataStim.mat) that applies to all subjects. This is compatible
% with scenarios with randomise presentation orders. In that case, the
% EEG/MEG trials should be sorted to match the single stimulus file. 
% The original order is preserved in a specific CND variable. If distinct
% subjects were presented with different stimuli, it is necessary to
% include a stimulus file per participant.
%
% CNSP-Workshop 2022
% https://cnspworkshop.net
% Author: Giovanni M. Di Liberto
% Copyright 2021 - Giovanni Di Liberto
%                  Nathaniel Zuk
%                  Michael Crosse
%                  Aaron Nidiffer
%                  Giorgia Cantisani
%                  (see license file for details)
% Last update: 24 June 2022
% Slightly Edited to fit dataset for John O'Doherty

clearvars -except eeg stim
close all

%%
addpath ./lib/cnsp_utils
addpath ./lib/cnsp_utils/cnd
addpath ./lib/mTRF-Toolbox_v2/mtrf
addpath ./lib/NoiseTools
addpath ./lib/eeglab2024.0/

eeglab
%% Parameters - Natural speech listening experiment
dataMainFolder = './';
% dataMainFolder = '../datasets/LalorNatSpeechReverse/';
% dataMainFolder = '../datasets/AliceSpeech/';
dataCNDSubfolder = 'dataCND/';

reRefType = 'mastoids'; % or 'Mastoids'
bandpassFilterRange = [0.1,42]; % Hz (indicate 0 to avoid running the low-pass
                          % or high-pass filters or both)
                          % e.g., [0,8] will apply only a low-pass filter
                          % at 8 Hz
downFs = 512; % Hz. *** fs/downFs must be an integer value ***

eegFilenames = dir([dataMainFolder,dataCNDSubfolder,'dataSub*.mat']);
[~, reindex] = sort( str2double( regexp( {eegFilenames.name}, '\d+', 'match', 'once' )));
eegFilenames = eegFilenames(reindex);

stimFilenames = dir([dataMainFolder,dataCNDSubfolder,'dataStim*.mat']);
nSubs = length(eegFilenames);

if downFs < bandpassFilterRange(2)*2
    disp('Warning: Be careful. The low-pass filter should use a cut-off frequency smaller than downFs/2')
end

%% Preprocess EEG - Natural speech listening experiment
for sub = 1:12
    % Loading EEG data
    eegFilename = [dataMainFolder,dataCNDSubfolder,eegFilenames(sub).name];
    disp(['Loading EEG data: ',eegFilenames(sub).name])
    load(eegFilename,'eeg')
    
    eeg = cndNewOp(eeg,'Load'); % Saving the processing pipeline in the eeg struct

    % Handling button presses and switches first as the downsample messes
    % them up
    behaviouralData = eeg.extChan(2:3);
    for iExt = 1:2
        data = behaviouralData{iExt}.data;

        for iTrl = 1:length(data)
            times = round((find(data{iTrl})/eeg.fs)*downFs);
            timesVector = zeros(1,round(length(data{iTrl})/eeg.fs*downFs));
            timesVector(times) = 1;
            data{iTrl} = timesVector;
        end
        behaviouralData{iExt}.data = data;
    end

    % Filtering - LPF (low-pass filter)
    if bandpassFilterRange(2) > 0
        hd = getLPFilt(eeg.fs,bandpassFilterRange(2));
        
        % Filtering each trial/run with a cellfun statement
        eeg.data = cellfun(@(x) filtfilthd(hd,x),eeg.data,'UniformOutput',false);
        
        % Filtering external channels
        if isfield(eeg,'extChan')
            for extIdx = 1
                eeg.extChan{extIdx}.data = cellfun(@(x) filtfilthd(hd,x),eeg.extChan{extIdx}.data,'UniformOutput',false);
            end
        end
        
        eeg = cndNewOp(eeg,['LPF ',num2str(bandpassFilterRange(2))]);
    end
    
    % Downsampling EEG and external channels
    if downFs < eeg.fs
        eeg = cndDownsample(eeg,downFs);
    end
    
    % Filtering - HPF (high-pass filter)
    if bandpassFilterRange(1) > 0 
        hd = getHPFilt(eeg.fs,bandpassFilterRange(1));
        
        % Filtering EEG data
        eeg.data = cellfun(@(x) filtfilthd(hd,x),eeg.data,'UniformOutput',false);
        
        % Filtering external channels
        if isfield(eeg,'extChan')
            for extIdx = 1
                eeg.extChan{extIdx}.data = cellfun(@(x) filtfilthd(hd,x),eeg.extChan{extIdx}.data,'UniformOutput',false);
            end  
        end
        
        eeg = cndNewOp(eeg,['HPF ',num2str(bandpassFilterRange(1))]);
    end
    
    % Replacing bad channels
    if isfield(eeg,'chanlocs')
        for tr = 1:length(eeg.data)
            eeg.data{tr} = removeBadChannels(eeg.data{tr}, eeg.chanlocs);
        end
    end

    eeg.data = cellfun(@double,eeg.data,UniformOutput=false);
    
    % Re-referencing EEG data
    eeg = cndReref(eeg,reRefType);

    eeg.extChan(2:3) = behaviouralData;
    
    % Saving preprocessed data
    eegPreFilename = [dataMainFolder,dataCNDSubfolder,'pre_beta_',eegFilenames(sub).name];
    disp(['Saving preprocessed EEG data: pre_',eegFilenames(sub).name])
    save(eegPreFilename,'eeg')
end
disp('Done')

function result = do_horzcat(data,indices)
    result = data(1);
    for i = 2:length(indices)
        result = cellfun(@horzcat,result,data(i),'uni',0);
    end
end
