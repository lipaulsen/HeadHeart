% function [] = erp_calc_stats(subjects, data_dir, results_dir)

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

subfnames = fieldnames(subjects);
preprocessed_name = 'preprocessed'; 
epoch_name = 'epoch';

% Define if plots are to be shown
show_plots = false;

% Channel Parameters
channels = {'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'Pz', 'STNl', 'STNr'};
LfpElec.SG041 = {'L3', 'R3'};
LfpElec.SG043  = {'L4', 'R1'};
LfpElec.SG046  = {'L4', 'R1'};
LfpElec.SG047  = {'L3', 'R4'};
LfpElec.SG050 = {'L3', 'R3'};
LfpElec.SG052  = {'L4', 'R2'};
LfpElec.SG056  = {'L4', 'R1'};

nSub = numel(subjects.goodHeartMOff);

seperateSTN = true

NewSR=300;

% Filter Parameters
Fhp = 4;
FltPassDir='twopass'; % onepass  twopass

steps = {'Plot SS ERP'}; 

% Define Time Window
tWidth   = 0.9;
tOffset  = 0.3;

% Baseline parameters
baseline = true;
baseline_win = [-0.3 -0.1];


for fn = 1:2 % MedOn
    subfname = subfnames{fn};

    fprintf('Loading AVG ECG Data\n');
    pattern = fullfile(data_dir, epoch_name, 'avg', ['ECG-AVG_', subfname, 'n=', num2str(nSub), '*']);
    files = dir(pattern);
    filename = fullfile(files(1).folder, files(1).name);
    load(filename, 'AVGECG');

    for sub = 1:numel(subjects.goodHeartMOff) % BE AWARE THAT THIS EXCLUDES PATIENTS WITH ARRITHYMIAS
        % Extract the subject
        subject = subjects.goodHeartMOff{sub};
        
        fprintf('Loading Data of  subject %s number %i of %i\n', subject, sub, numel(subjects.goodHeartMOff));

        pattern = fullfile(data_dir, 'preproc', 'all', [subject, '_', preprocessed_name, '_', subfname, '*']);
        files = dir(pattern);
        filename = fullfile(files(1).folder, files(1).name);
        load(filename, 'SmrData');
        % Load subject data
        % subject_data = fullfile(data_dir, preprocessed_name, subfname, ['sub-', subject], [subject, '_preprocessed_', subfname, '_Rest.mat']);
        % load(subject_data, 'SmrData');

        pattern = fullfile(data_dir, 'itc', 'evecg' ,[subject, '_', subfname, '*']);
        files = dir(pattern);
        filename = fullfile(files(1).folder, files(1).name);
        load(filename, 'EvEcgData');

        SR = SmrData.SR;
        EventTms = SmrData.EvData.EvECGP_Cl;

        for c = 1:numel(channels)
            if seperateSTN
                channels{8} = LfpElec.(subject){1};
                channels{9} = LfpElec.(subject){2};
            end
            channel = channels{c};

            %% ==================== EPOCH DATA ==========================
            fprintf('****************** EPOCH for %s %s...****************\n', subject, channel);

            ChDta = SmrData.WvDataCleaned(c, :);

            % HIGH PASS FILTER
            if Fhp > 0
                ChDta=ft_preproc_highpassfilter(ChDta,SR,Fhp,4,'but',FltPassDir); % twopass
            end

            % DOWNSASMPLE
            if NewSR > 0
                FsOrigin=SR;
                if  FsOrigin ~=  NewSR
                    [fsorig, fsres] = rat(FsOrigin/NewSR);
                    ChDta=resample(ChDta,fsres,fsorig);
                    dtTime=1/NewSR;
                end
                NewSR=1.0/dtTime;
            end

            [EventTms,EvData,TmAxis]=GetEvTimeAndData(EventTms,ChDta,dtTime,tWidth,tOffset);
            [nEvs,nData]=size(EvData);

            %% ============== BASELINE CORRECTION ======================

            if baseline

                fprintf('****************** Baseline Correction for %s %s med: %s ...****************\n', subject, channel, subfname);
                % My baseline is the time window -0.3 to -0.1 s
                % before my Rpeak of every trial

                % Find the the indices for the basseline window
                bidx = find(TmAxis' >= baseline_win(1) & TmAxis' <= baseline_win(2));

                % for every trial calc the mean of the baseline win
                % and subtract that from the entire epoch
                for t = 1:nEvs
                    baseline_mean = mean(EvData(t, bidx(1):bidx(end)),2);
                    EvData(t,:) = EvData(t,:)-baseline_mean;
                end

            end
            
            EvDataAllAvgTrs(sub,c,:) = squeeze(mean(EvData,1));

        end
    end

    %% ============== Plotting ERPs SUBJECT LEVEL=========================
 % 
 % f2 = figure;
 %         set(f2,'Position', [1949 123 1023 785]);
 %         row = ceil(c / 3); % Calculate the row number
 %         col = mod(c - 1, 3) + 1; % Calculate the column number
 %         subplot(3, 5, (row - 1) * 5 + col)






    %% ============== Plotting ERPs GROUP LEVEL=========================
    
    for c = 1:numel(channels)
         channels = {'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'Pz', 'STNl', 'STNr'};
         channel = channels{c};
    
         EvDataAll_chanavg = squeeze(mean(squeeze(EvDataAllAvgTrs(:,c,:)),1));
            
         f2 = figure;
         set(f2,'Position', [1949 123 1023 785]);
         
         subplot(2,1,1)
         plot(TmAxis, AVGECG.mean', 'Color', 'k'); hold on
         set(gca,'Position',[0.1300 0.5838 0.77 0.3])
         xline(0, "--k", 'LineWidth', 2);
         title(sprintf('Average ECG in %s, medication: %s', channel, subfname))
         axis("tight");
         ylabel('Amplitude (μV)')
         hold off

         subplot(2,1,2)
         plot(TmAxis, EvDataAll_chanavg, 'Color', 'b');hold on
         xline(0, "--k", 'LineWidth', 2);
         xlabel('Time (s)') % Add x-label
         ylabel('Amplitude (μV)') % Add y-label
         title(sprintf('Average ERP in %s, medication: %s', channel, subfname))
         axis("tight");

         gr2 = fullfile('F:\HeadHeart\2_results\erp' , ['AvgERP_', channel, '_', subfname,  '.png']);
         exportgraphics(f2,gr2, 'Resolution', 300)

    end

end