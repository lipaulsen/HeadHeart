% function [] = epoch_timeLock(subjects, data_dir, results_dir)

%% Epoching and Time Locking Data for HeadHeart

% Author: Lisa Paulsen
% Contact: lisaspaulsen[at]web.de
% Created on: 1 October 2024
% Last update: 15 October 2024

%% REQUIRED TOOLBOXES
% Image Processing Toolbox
% Signal Processing Toolbox
% Statistics and Machine Learning Toolbox

% Extract Features through Time Frequencz Decomposition from EEG and ECG data
%
% Inputs:
% Preprocessed data (EEG, LFP, ECG) from .mat file
%
% Outputs:    Features extracted from the data in .mat files
% - ECG: IBI(sub, :), HRV, HF-HRV, LF-HRV
% - EEG & LFP: Power of delta, theta, alpha, beta, gamma bands for all electrodes

% Steps:
% 1. LOAD DATA
% 2. FEATURE EXTRACTION ECG
%   2a. Calculate HRV features from IBI(sub, :) data
%   2b. Save HRV features in a tsv file
%   2c. Plot HRV features
% 3. FEATURE EXTRACTION EEG
%   3a. Calculate the power of all EEG frequency bands
%   3b. Calculate the mean power of the EEG frequency bands for a region of interest (ROI)
%   3c. Save EEG power features in a tsv file
%   3d. Plot EEG power features
% 4. AVERAGE SELECTED FEATURES ACROSS PARTICIPANTS

%% ============= SET GLOABAL VARIABLES AND PATHS =========================
%clear all
%close all

subfnames = fieldnames(subjects);

% Define if plots are to be shown
show_plots = false;

%If we use left and right STN as seperate subjects put this as true
%(increases subjectsize by 2)
seperateSTN = true;

%flag if baseline is on or off
baseline = false; %currently no baseline but if needed can be added

% Define feature extraction steps to perform
steps = {'Load Data', 'Epoch and Timelock Data'};

% Define folder variables
tfr_name = 'tfr';  % averaged data folder (inside preprocessed)
epoch_name = 'epoch';  % feature extraction folder (inside derivatives)

channels = {'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'Pz', 'L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4'};
% channels = {'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'Pz', 'STN'};
if seperateSTN
channels = {'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'Pz', 'STNl', 'STNr'};
end

LfpElec.SG041 = {'L3', 'R3'};
LfpElec.SG043  = {'L4', 'R1'}; 
LfpElec.SG046  = {'L4', 'R1'};
LfpElec.SG047  = {'L3', 'R4'};
LfpElec.SG050 = {'L3', 'R3'};
LfpElec.SG052  = {'L4', 'R2'};
LfpElec.SG056  = {'L4', 'R1'};
LfpElec.STNl = {'L1', 'L2', 'L3', 'L4'};
LfpElec.STNr = {'R1', 'R2', 'R3', 'R4'};


% Define Time Window
tWidth   = 1.5;
tOffset  = 0.4;

% Define Struct
EPOCH = [];

%% ============================ 1. LOAD DATA =============================
disp("************* STARTING EPOCH AND TIMELOCKING *************");

for fn = 2 % MedOn
    subfname = subfnames{fn};
    for c = 1:numel(channels)
        for sub = 1:numel(subjects.goodHeartMOff) % BE AWARE THAT THIS EXCLUDES PATIENTS WITH ARRITHYMIAS
            
            % Extract the subject
            subject = subjects.goodHeartMOff{sub};
            nSub = numel(subjects.goodHeartMOff);
            %I should count my seperate STNs as subjects but I dont
            %understand yet why or if i should up my sub count
            % if seperateSTN
            %     nSub = numel(subjects.goodHeartMOff)*2;
            % end

            if seperateSTN
                channels{8} = LfpElec.(subject){1};
                channels{9} = LfpElec.(subject){2};
            end
            channel = channels{c};

            if ismember('Load Data', steps)

                fprintf('Loading Data of subject %s number %i of %i\n', subject, sub, numel(subjects.goodHeartMOff));

                pattern = fullfile(data_dir, tfr_name, [subject, '_TFR_', subfname, '*']);
                files = dir(pattern);
                filename = fullfile(files(1).folder, files(1).name);
                load(filename, 'TFR', '-mat');

                % Load the the cleaned ECG R Peaks Data
                pattern = fullfile(data_dir, 'preproc', 'all', [subject, '_preprocessed_', subfname, '*']);
                files = dir(pattern);
                filename = fullfile(files(1).folder, files(1).name);
                load(filename, 'SmrData');

            end

            SR = TFR.SR;
            dtTime=1/SR;
            freqs = TFR.freqs;

            EventTms = SmrData.EvData.EvECGP_Cl;



            % ???????????? CALCULATE BASELINE PER TRIAL ?????????????????????

            % I will treat left and right STN as seperate subjects 
            % I will use the LFP Electorde with the highest beta as my STN 





            %% ============================ 2. EPOCH and TIMELOCK DATA =============================
            disp("************* STARTING EPOCH AND TIMELOCKING *************");
            if ismember('Epoch and Timelock Data', steps)

                fprintf('****************** Extract Time for %s %s...****************\n', subject, channel);
                % Get Channel Power and Phase Data
                ChDta_phs = squeeze(TFR.(channel).phase(:,:));
                ChDta_pow = squeeze(TFR.(channel).pow(:,:));

                % Extract the Time Windows around Event
                nEvent=length(EventTms);
                nDataAll=length(ChDta_pow);

                nWidth=int32(tWidth/dtTime)+1;
                nOffset=int32(tOffset/dtTime)+1;

                % Define The Epoch Event Matrix
                epochChDtaPhs = zeros(nEvent,numel(freqs), nWidth);
                epochChDtaPow = zeros(nEvent,numel(freqs), nWidth);
                epochChDtaPhsall = zeros(nEvent, nSub, numel(freqs), nWidth);
                epochChDtaPowall = zeros(nEvent, nSub, numel(freqs), nWidth);

                % Ectract the epoch times around the r peak  
                for i = 1:nEvent
                    currtime=int32(EventTms(i)/dtTime);
                    n1=currtime-nOffset;
                    n2=n1+nWidth-1;
                    if n1 > 0 && n2 < nDataAll % Check that the time windows is int he data
                        % extract the Data around the time window
                        epochChDtaPhs(i,:,:) = ChDta_phs(:,n1:n2); %(TrialsxFreqxPhase)
                        epochChDtaPow(i,:,:) = ChDta_pow(:,n1:n2); %(TrialsxFreqxPower)
                        epochChDtaPhsall(i,sub,:,:) = ChDta_phs(:,n1:n2); %(TrialsxSubxFreqxPhase)
                        epochChDtaPowall(i,sub,:,:) = ChDta_pow(:,n1:n2); %(TrialsxSubxFreqxPower)
                    end
                end
                % Extract the Time Axis for future use
                TmAxis=zeros(1,nWidth);
                s=-1*tOffset;  for i=1:nWidth;  TmAxis(i)=s;  s=s+dtTime;  end
                
                % EPOCH.(subject).(channel).pow = epochChDtaPow;
                % EPOCH.(subject).(channel).meanpow = squeeze(mean(epochChDtaPow, 1, 'omitnan'));
                % EPOCH.(subject).(channel).medianpow = squeeze(median(epochChDtaPow, 1, 'omitnan'));
                % EPOCH.(subject).(channel).phs = epochChDtaPhs;
                % EPOCH.(subject).(channel).meanphs = squeeze(mean(epochChDtaPhs, 1, 'omitnan'));
                % EPOCH.(subject).(channel).medianphs = squeeze(median(epochChDtaPhs, 1, 'omitnan'));
                % EPOCH.SR = SR;
                % EPOCH.freqs = freqs;
                % EPOCH.times = TmAxis;
                % 
                % 
                % save_path = fullfile(data_dir, epoch_name, [subject,  '_EPOCH_', subfname ,'_Rest_Hilbert_Freq=', num2str(freqs(1)),'-', num2str(freqs(end)),'Hz_bin=', num2str(mean(diff(freqs))),'HZ-Epoch=', num2str(TmAxis(1)),'to',num2str(TmAxis(end)),'s.mat']);
                % save(save_path, 'EPOCH', '-v7.3');

            end
        end
        
        % rename the channels to STNl and STNr so that it can be averaged and saved
        % effectively in this struct 
        channels = {'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'Pz', 'STNl', 'STNr'};
        channel = channels{c};

        POWER.(channel).mean = squeeze(mean(mean(epochChDtaPowall, 1, 'omitnan'), 2, 'omitnan')); % Mean over all trials and all subjects
        %POWER.(channel).median = squeeze(median(median(epochChDtaPowall, 1, 'omitnan'), 2, 'omitnan'));

        PHASE.(channel).mean = squeeze(mean(mean(epochChDtaPhsall, 1, 'omitnan'), 2, 'omitnan'));
        %PHASE.(channel).median = squeeze(median(median(epochChDtaPhsall, 1, 'omitnan'), 2, 'omitnan'));

        Params.SR = SR;
        Params.freqs = freqs; 
        Params.times = TmAxis;

       


    end

    save_path = fullfile(data_dir, epoch_name, 'avg', ['POW-PHS-AVG_', subfname ,'n=', num2str(nSub),...
        '_Rest_Hilbert_Freq=', num2str(freqs(1)),'-', num2str(freqs(end)),'Hz_bin=', num2str(mean(diff(freqs))),...
        'HZ-Epoch=', num2str(TmAxis(1)),'to',num2str(TmAxis(end)),'s_BSL=', num2str(baseline),'.mat']);
    save(save_path, 'POWER', 'PHASE', 'Params', '-v7.3');


end

%% ============================ 3. AVERAGE DATA OVER ALL SUBS =============================
disp("************* AVERAGE the TFR DATA  *************");



