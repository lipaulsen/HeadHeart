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
% 2. EPOCH and TIMELOCK DATA
% 3. SAVE DATA

%% ============= SET GLOABAL VARIABLES AND PATHS =========================
%clear all
%close all

% ========================== SUBJECT FLAGS ================================

% MEDICATION
% only one can be true at all times
MedOn = true;
MedOff = false;

% SUBJECT STATUS
% only one can be true at all times
newsubs = false;
oldsubs = false;
allsubs = true;

if MedOn == true & newsubs == true % Only New Subs that are MedOn
    subjects = string({subjects([subjects.new] == 1 & [subjects.MedOn] == 1).ID});
elseif MedOff == true & newsubs == true  % Only New Subs that are MedOff
    subjects = string({subjects([subjects.new] == 1 & [subjects.MedOff] == 1).ID});
elseif MedOn == true & oldsubs == true  % Only Old Subs that are MedOn
    subjects = string({subjects([subjects.new] == 0 & [subjects.MedOn] == 1).ID});
elseif MedOff == true & oldsubs == true  % Only Old Subs that are MedOff
    subjects = string({subjects([subjects.new] == 0 & [subjects.MedOff] == 1).ID});
elseif MedOn == true & allsubs == true  % All Subs that are MedOn
    subjects = string({subject_info([subjects.MedOn] == 1).ID});
elseif MedOff == true & allsubs == true % All Subs that are MedOff
    subjects = string({subject_info([subjects.MedOff] == 1).ID});
end


% Convert 'channels_raw' (string) to cell array of strings
channels_raw = cellfun(@(x) strsplit(x, ', '), {subjects.channels_raw}, 'UniformOutput', false);
channels_stn = cellfun(@(x) strsplit(x, ', '), {subjects.channels}, 'UniformOutput', false);

channel = channels_stn([subjects.new] == 1 & [subjects.MedOn] == 1);
chans = string(channel{1,3})

%subfnames = fieldnames(subjects);

%=========================================================================

% Define if plots are to be shown
show_plots = false;

%If we use left and right STN as seperate subjects put this as true
%(increases subjectsize by 2)
seperateSTN = true;

%flag if baseline is on or off
baseline = false; %currently no baseline but if needed can be added

% Define feature extraction steps to perform
steps = {'ECG Data'}; %'Load Data', 'Epoch and Timelock Data',

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
tWidth   = 0.9;
tOffset  = 0.3;

% Define Struct
EPOCH = [];
EP = []; 

TfrAllPhs = [];

nSub = numel(subjects.goodHeartMOff);

epochChDtaPhsAll = [];
epochChDtaPowAll = [];


%% ============================ 1. LOAD DATA =============================
disp("************* STARTING EPOCH AND TIMELOCKING *************");
if ismember('Epoch and Timelock Data', steps)
    for fn = 1%:2 % MedOn
        for c = 1:numel(channels)
            subfname = subfnames{fn};

            for sub = 1:numel(subjects.goodHeartMOff) % BE AWARE THAT THIS EXCLUDES PATIENTS WITH ARRITHYMIAS

                % Extract the subject
                subject = subjects.goodHeartMOff{sub};

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

                %% ============================ 2. EPOCH and TIMELOCK DATA =============================
                disp("************* STARTING EPOCH AND TIMELOCKING *************");


                fprintf('****************** Extract Time for %s %s...****************\n', subject, channel);
                % Get Channel Power and Phase Data
                ChDta_phs = TFR.(channel).phase(:,:);
                ChDta_pow = TFR.(channel).pow(:,:);

                % Extract the Time Windows around Event
                nEvent=length(EventTms);
                nDataAll=length(ChDta_pow);

                nWidth=int32(tWidth/dtTime)+1;
                nOffset=int32(tOffset/dtTime)+1;

                % % Define The Epoch Event Matrix
                % epochChDtaPhs = NaN(nEvent,numel(freqs), nWidth);
                % epochChDtaPow = NaN(nEvent,numel(freqs), nWidth);
                % % epochChDtaPhsall = zeros(nEvent, nSub, numel(freqs), nWidth); % Maybe rather with NAN and then omitnan
                % % epochChDtaPowall = zeros(nEvent, nSub, numel(freqs), nWidth);
                if ismember(channel, LfpElec.STNl)
                    channel = 'STNl';
                elseif ismember(channel, LfpElec.STNr)
                    channel = 'STNr';
                end
                

                MasEnableEvents=int32(zeros(nEvent,1));
                nPosibleEvent=int32(0);
                for i = 1:nEvent
                    currtime=int32(EventTms(i)/dtTime);
                    n1=currtime-nOffset;
                    n2=n1+nWidth-1;
                    if n1 > 0 && n2 < nDataAll
                        nPosibleEvent=nPosibleEvent+1;
                        MasEnableEvents(nPosibleEvent)=i;
                    end
                end
                nEvent=nPosibleEvent;
                % Save All events  in case for later analysis, could be delete;
                EvTime=zeros(1,nEvent);
                for i = 1:nEvent
                    EvTime(i)=EventTms(MasEnableEvents(i));
                    currtime=int32(EvTime(i)/dtTime);
                    n1=currtime-nOffset;
                    n2=n1+nWidth-1;
                    EPOCH.(channel).(subject).phase(i,:,:) = ChDta_phs(:,n1:n2); %(TrialsxFreqxPhase)
                    EPOCH.(channel).(subject).power(i,:,:) = ChDta_pow(:,n1:n2); %(TrialsxFreqxPower)
                end
                EventTms = EvTime;
                % Ectract the epoch times around the r peak
                % for i = 1:nEvent
                %     currtime=int32(EventTms(i)/dtTime);
                %     n1=currtime-nOffset;
                %     n2=n1+nWidth-1;
                %     if n1 > 0 && n2 < nDataAll % Check that the time windows is int he data
                %         % extract the Data around the time window
                %         EPOCH.(channel).(subject).phase(i,:,:) = ChDta_phs(:,n1:n2); %(TrialsxFreqxPhase)
                %         EPOCH.(channel).(subject).power(i,:,:) = ChDta_pow(:,n1:n2); %(TrialsxFreqxPower)
                % 
                %     end
                % end
                [nEvent, Frqs, nTm] = size(EPOCH.(channel).(subject).phase);
                TmAxis=zeros(nTm,1)';
                s=-1*tOffset;  for i=1:nTm;  TmAxis(i)=s;  s=s+dtTime;  end
            end
        end

        % Create and save a structure for each subject
        for sub = 1:numel(subjects.goodHeartMOff)
            subject = subjects.goodHeartMOff{sub};

            % Initialize a subject-specific structure
            EP = struct();
            EP.subject = subject;
            EP.channels = channels;  % Include channel names for reference
            EP.SR = SR;
            EP.freqs = freqs;
            EP.time = TmAxis;
            channels = {'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'Pz', 'STNl', 'STNr'};

            % Copy data from EPOCH to EP
            for ch = 1:numel(channels)
                channel = channels{ch};
                if isfield(EPOCH, channel) && isfield(EPOCH.(channel), subject)
                    EP.(channel).phase = EPOCH.(channel).(subject).phase;
                    EP.(channel).power = EPOCH.(channel).(subject).power;
                end
            end

            % Save the per-subject data
            save_filename = fullfile(data_dir, epoch_name, [subject,'_', subfname ,...
            '_Rest_Hilbert_Freq=', num2str(freqs(1)),'-', num2str(freqs(end)),'Hz_bin=', num2str(mean(diff(freqs))),...
            'HZ-Epoch=', num2str(TmAxis(1)),'to',num2str(TmAxis(end)),'s_BSL=', num2str(baseline),'.mat']);
            save(save_filename, 'EP', '-v7.3');  % Use -v7.3 for large files
            fprintf('Saved epoched data for subject %s to %s\n', subject, save_filename);
        end
        %% ============================ 3. SAVE DATA =============================
        channels = {'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'Pz', 'STNl', 'STNr'};
        % Loop over each channel to calculate mean power and phase across subjects
        for c = 1:numel(channels)

            channel = channels{c}; % Get the current channel
            % Initialize variables to store power and phase data across subjects
            % These will hold the power and phase data for each subject
            power_all_subs = {};
            phase_all_subs = {};

            % Loop over each subject
            for sub = 1:numel(subjects.goodHeartMOff)
                subject = subjects.goodHeartMOff{sub};

                % Check if the current subject's data exists in the EPOCH struct
                if isfield(EPOCH, channel) && isfield(EPOCH.(channel), subject)
                    % Get the power and phase data for the current subject and channel
                    power_data = EPOCH.(channel).(subject).power; % (Events x Freqs x Time)
                    phase_data = EPOCH.(channel).(subject).phase; % (Events x Freqs x Time)

                    % Store the current subject's data in the cell arrays
                    power_all_subs{sub} = power_data; % Store power data for the subject
                    phase_all_subs{sub} = phase_data; % Store phase data for the subject
                end
            end

            % Now calculate the mean across subjects, preserving the event structure
            % For each event, we'll calculate the mean across subjects
            max_events = max(cellfun(@(x) size(x, 1), power_all_subs)); % Find the max number of events across all subjects
            mean_power = NaN(max_events, numel(freqs), nWidth); % Initialize matrix for mean power (Events x Freqs x Time)
            mean_phase = NaN(max_events, numel(freqs), nWidth); % Initialize matrix for mean phase (Events x Freqs x Time)

            % Loop over each event (subject-wise) and calculate the mean across subjects
            for e = 1:max_events
                temp_power = [];
                temp_phase = [];

                % Loop through subjects and extract data for event e
                for sub = 1:numel(subjects.goodHeartMOff)
                    if size(power_all_subs{sub}, 1) >= e % Check if the subject has this event
                        temp_power = cat(1, temp_power, power_all_subs{sub}(e, :, :)); % Concatenate data across subjects
                        temp_phase = cat(1, temp_phase, phase_all_subs{sub}(e, :, :)); % Concatenate data across subjects
                    end
                end

                % Calculate the mean for each event across subjects
                mean_power(e, :, :) = mean(temp_power, 1, 'omitnan'); % Mean across subjects for event e
                mean_phase(e, :, :) = mean(temp_phase, 1, 'omitnan'); % Mean across subjects for event e
            end

            % Store the results in the EPOCH struct for later use
            POWER.(channel).submean = mean_power; % (Events x Freqs x Time)
            POWER.(channel).mean = squeeze(mean(mean_power,1)); % (Freqs x Time)
            PHASE.(channel).submean = mean_phase; % (Events x Freqs x Time)
            POWER.(channel).mean = squeeze(mean(mean_phase,1)); % (Freqs x Time)

            Params.SR = SR;
            Params.freqs = freqs;
            Params.times = TmAxis;

            % Optionally, display the mean power and phase for the current channel
            fprintf('Mean Power and Phase calculated for channel: %s\n', channel);
        end

        save_path = fullfile(data_dir, epoch_name, 'avg', ['POW_AVG_', subfname ,'n=', num2str(nSub),...
            '_Rest_Hilbert_Freq=', num2str(freqs(1)),'-', num2str(freqs(end)),'Hz_bin=', num2str(mean(diff(freqs))),...
            'HZ-Epoch=', num2str(TmAxis(1)),'to',num2str(TmAxis(end)),'s_BSL=', num2str(baseline),'.mat']);
        save(save_path, 'POWER', 'Params', '-v7.3');

        save_path = fullfile(data_dir, epoch_name, 'avg', ['PHS-AVG_', subfname ,'n=', num2str(nSub),...
            '_Rest_Hilbert_Freq=', num2str(freqs(1)),'-', num2str(freqs(end)),'Hz_bin=', num2str(mean(diff(freqs))),...
            'HZ-Epoch=', num2str(TmAxis(1)),'to',num2str(TmAxis(end)),'s_BSL=', num2str(baseline),'.mat']);
        save(save_path, 'PHASE', 'Params', '-v7.3');

    end
end


%% ============================ 3. AVERAGE DATA OVER ALL SUBS =============================


% WARNNING IF YOU EVER USE THIS CODE HTERE WERE HEAVY ADJUSTMENTS MADE SO
% PLEASE BE AWARE AND TAKE CARE - I TRIED TO WRITE IN THE COMMENT WHERE A
% CRITICAL ADJUSTMENT WAS MADE 



if ismember('ECG Data', steps)
    disp("************* AVERAGE the ECG DATA  *************");

    NewSR = 300;

    epochECGall = cell(numel(subjects.goodHeartMOff), 1)

    for f = 1:2 % MedOn
        subfname = subfnames{f};
        for s = 1:numel(subjects.goodHeartMOff) % BE AWARE THAT THIS EXCLUDES PATIENTS WITH ARRITHYMIAS

            % Extract the subject
            subject = subjects.goodHeartMOff{s};


            fprintf('Loading Data of subject %s number %i of %i\n', subject, s, numel(subjects.goodHeartMOff));


             % Load the the cleaned ECG R Peaks Data
            pattern = fullfile(data_dir, 'itc', 'evecg' ,[subject, '_', subfname, '*']);
            files = dir(pattern);
            filename = fullfile(files(1).folder, files(1).name);
            load(filename, 'EvEcgData');

            epochECGall{s} = EvEcgData;

        end

        % 
        %     % Load the the cleaned ECG R Peaks Data
        %     pattern = fullfile(data_dir, 'preproc', 'all', [subject, '_preprocessed_', subfname, '*']);
        %     files = dir(pattern);
        %     filename = fullfile(files(1).folder, files(1).name);
        %     load(filename, 'SmrData');
        % 
        % 
        %     EventTms = SmrData.EvData.EvECGP_Cl;
        %     ECG = SmrData.WvDataCleaned(21,:);
        %     SR = SmrData.SR;
        % 
        %     %Downsample auf 300
        %     % DOWNSASMPLE
        %     if NewSR > 0
        %         FsOrigin=SR;
        %         if  FsOrigin ~=  NewSR
        %             [fsorig, fsres] = rat(FsOrigin/NewSR);
        %             ECG=resample(ECG,fsres,fsorig);
        %             dtTime=1/NewSR;
        %         end
        %         NewSR=1.0/dtTime;
        %     end
        % 
        %     % Extract the Time Windows around Event
        %     nEvent=length(EventTms);
        %     nDataAll=length(ECG);
        % 
        %     nWidth=int32(tWidth/dtTime)+1;
        %     nOffset=int32(tOffset/dtTime)+1;
        % 
        %     % Define The Epoch Event Matrix
        %     % epochECG = zeros(nEvent,numel(freqs), nWidth);
        % 
        %     % Preallocate matrix for current subject's epochs (nEvent x nWidth)
        %     subjectEpochs = NaN(nEvent, nWidth); % NaN helps spot missing trials
        % 
        %     % Ectract the epoch times around the r peak
        %     for e = 1:nEvent
        %         currtime=int32(EventTms(e)/dtTime);
        %         n1=currtime-nOffset;
        %         n2=n1+nWidth-1;
        %         if n1 > 0 && n2 < nDataAll % Check that the time windows is int he data
        %             % extract the Data around the time window
        %             %epochECG(i,:,:) = ECG(n1:n2); %(TrialsxTime)
        %             %epochECGall(e,s,:) = ECG(n1:n2); %(TrialsxSubxTime)
        %             subjectEpochs(e, :) = ECG(n1:n2);
        %         end
        %     end
        %     [nEvent, nTm] = size(subjectEpochs);
        %         TmAxis=zeros(nTm,1)';
        %         st=-1*tOffset;  for i=1:nTm;  TmAxis(i)=st;  st=st+dtTime;  end
        %     % Store the subject's epochs in the cell array
        %     epochECGall{s} = subjectEpochs;
        % end
        % 
        % 
        % % Initialize a cell array to store averaged data
         epochECGall_avg = cell(1, 6);

        % Loop through each cell in epochECGall
        for i = 1:length(epochECGall)
            epochECGall_avg{i} = mean(epochECGall{i}, 1, 'omitnan'); % Average over trials (dimension 1)
            figure;
            plot(TmAxis, epochECGall_avg{i})
            title(sprintf('AVG ECG for %s'), subjects.goodHeartMOff{i})
        end

        % Combine all averages into a single matrix
        all_avg_matrix = cell2mat(epochECGall_avg'); % Resulting size: 6 x 451

        % Compute the grand average across all datasets
        grand_avg = mean(all_avg_matrix, 1); % Resulting size: 1 x 451
        figure;
        plot(TmAxis, epochECGall_avg{i})
        title(sprintf('Grand AVG ECG over all Subjects'))

      
        AVGECG.allsubs = all_avg_matrix;
        AVGECG.mean = grand_avg;
        AVGECG.SR = 300; % TAKE CARE
        AVGECG.times = TmAxis;


        save_path = fullfile(data_dir, epoch_name, 'avg', ['ECG-AVG_', subfname ,'n=', num2str(nSub),...
            '_Rest_Hilbert_Freq=', num2str(freqs(1)),'-', num2str(freqs(end)),'Hz_bin=', num2str(mean(diff(freqs))),...
            'HZ-Epoch=', num2str(TmAxis(1)),'to',num2str(TmAxis(end)),'s_BSL=', num2str(baseline),'.mat']);
        save(save_path, 'AVGECG', '-v7.3');

    end
end