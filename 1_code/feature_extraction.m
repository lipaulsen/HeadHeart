% function [] = feature_extraction(subjects, data_dir, results_dir)

%% Time Frequency Decomposition for HeadHeart

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
% - ECG: IBI, HR, HRV (avg_features_folder, ['Averages_HRV-IBI-HR_Rest_nsub=', num2str(numel(subjects.goodHeartMOff)),'.mat'])
% - EEG & LFP: Epoched TFR (data_dir, 'tfr', Hz_dir, [subject,  '_TFR-EPOCH_', subfname ,'_Rest_Hilbert_Freq=',
%                           num2str(stfr),'-', num2str(enfr),'_bin=', num2str(dfr),'Hz_DS=', num2str(NewSR),'_HP=', 
%                           num2str(Fhp),'Hz_EP=-',num2str(tOffset), 'to', num2str(tWidth-tOffset) ,'_BSL=', 
%                           num2str(baseline_win(1)),'to', num2str(baseline_win(2)),'s.mat']);

% Steps:
% 1. LOAD DATA
% 2. FEATURE EXTRACTION ECG
%   2a. Calculate HRV features from IBI(sub, :) data
%   2b. Save HRV features in a tsv file
%   2c. Plot HRV features
% 3. FEATURE EXTRACTION EEG
%   3a. HP Filter, Downsample, Epoch, Baseline Correction
%   3b. Time Freq Decomp
% 4. AVERAGE SELECTED FEATURES ACROSS PARTICIPANTS

%% ============= SET GLOABAL VARIABLES AND PATHS =========================
%clear all
%close all

subfnames = fieldnames(subjects);

% Define if plots are to be shown
show_plots = false;

% Define feature extraction steps to perform
steps = {'Load Data','Feature Extraction EEG'}; %'Feature Extraction ECG',

% Define folder variables
preprocessed_name = 'preprocessed';  % preprocessed folder (inside derivatives)
averaged_name = 'avg';  % averaged data folder (inside preprocessed)
feature_name = 'features';  % feature extraction folder (inside derivatives)



% Create the features data folder if it does not exist
for fn = 1:2 % MedOn and MedOff
    subfname = subfnames{fn};
    for i = 1:length(subjects.(subfname))
        subject = subjects.(subfname){i};
        subject_features_folder = fullfile(data_dir, feature_name, subfname, sprintf('sub-%s', subject));
        if ~exist(subject_features_folder, 'dir')
            mkdir(subject_features_folder);
        end

        % results folder
        subject_features_results_folder = fullfile(results_dir, feature_name, subfname, sprintf('sub-%s', subject));
        if ~exist(subject_features_results_folder, 'dir')
            mkdir(subject_features_results_folder);
        end
    end
end

avg_features_folder = fullfile(data_dir, feature_name, averaged_name);
if ~exist(avg_features_folder, 'dir')
    mkdir(avg_features_folder);
end

% Define parameters for time-frequency analysis of both ECG and EEG data
window_length_hrv = 10;  % 10 samples window  % Length of the window for smoothing

% Define parameters for time-frequency analysis of both ECG and EEG data
window_length_tfa = 2;  % 2s window  % Length of the window for smoothing
overlap = 0.5;  % 50% overlap    % Overlap of the windows for smoothing
mirror_length = 180;  % Length of the mirror extension for symmetric padding

% Define parameters for time-frequency analysis of ECG data
sampling_frequency_ibi = 1;  % Hz ibi and HR data are already sampled at 1 Hz from the preprocessing

% Define low and high frequency bands for HRV analysis
lf_band = [0.04, 0.15];
hf_band = [0.15, 0.4];

% % Define parameters for time-frequency analysis of EEG data
% sampling_frequency_eeg = 200;  % Hz EEG data will be downsampled to 100 Hz
% frequencies = 0.5:0.5:50;  % Resolution 0.5 Hz
% cycles = 6;


% Create color palette for plots
colors.ECG.IBI = [0.9569,    0.9451,    0.8706];    % Creme
colors.ECG.HRV = [0.8784,    0.4784,    0.3725];   % Pink
colors.ECG.LF_HRV = [0.2392    0.2510    0.3569];  % Light Orange
colors.ECG.HF_HRV = [0.5059    0.6980    0.6039];   % Dark Orange
%
colors.EEG.delta = [ 0.9490    0.8000    0.5608];  % Yellow
% colors.EEG.theta = "#D55E00";    % Dark Orange
% colors.EEG.alpha = "#CC79A7"; % Pink
% colors.EEG.beta = "#56B4E9";   % Light Blue
% colors.EEG.gamma = "#009E73";   % Green

%If we use left and right STN as seperate subjects put this as true
%(increases subjectsize by 2)
seperateSTN = true;

% Suppress excessive logging if using FieldTrip
ft_defaults; % If using FieldTrip

NewSR=300;
stfr=0.5;   enfr=30; dfr=0.2;
Frqs=stfr:dfr:enfr;
if NewSR < 2*enfr; NewSR=(enfr+2)*2; end

Fhp = 0.5;
Hz_dir = '0.5Hz';

BandWidth=2; % BandWidth in Hz;
Qfac     =2; % Attenuation in db(-Qfac)
WaveletnCyc=6;
WaveletgWidth=3;
FltPassDir='twopass'; % onepass  twopass

% Define Time Window
tWidth   = 0.9;
tOffset  = 0.3;

channels = {'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'Pz', 'L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4'};
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

%% ============================== FUNCTIONS ==============================



%% ============================ 1. LOAD DATA =============================
disp("************* STARTING Feature Extraction  *************");

%subjects = {'SG041', 'SG043',  'SG047', 'SG050', 'SG052', 'SG056'}; % this is wihtout the 2 patients with arythmia

% Initialize the matrix for all HRV measures
HRV.rmssd_avg = []; %
HRV.rmssd_tits{1} = {'HRV Average of subj'};

IBI = [];

HR = [];

ChsCmxEvFrTm =[];
epoch = true;
baseline = true;
baseline_win = [-0.3 -0.1];



for fn = 2%:2 % MedOn
    subfname = subfnames{fn};
    for sub = 1:numel(subjects.new) %numel(subjects.goodHeartMOff) % BE AWARE THAT THIS EXCLUDES PATIENTS WITH ARRITHYMIAS
        % Extract the subject
        subject = subjects.goodHeartMOff{sub};

        if ismember('Load Data', steps)

            fprintf('Loading Data of  subject %s number %i of %i\n', subject, sub, numel(subjects.goodHeartMOff));

            pattern = fullfile(data_dir, 'preproc', 'all', [subject, '_', preprocessed_name, '_', subfname, '*']);
            files = dir(pattern);
            filename = fullfile(files(1).folder, files(1).name);
            load(filename, 'SmrData');
            % Load subject data
            % subject_data = fullfile(data_dir, preprocessed_name, subfname, ['sub-', subject], [subject, '_preprocessed_', subfname, '_Rest.mat']);
            % load(subject_data, 'SmrData');

            SR = SmrData.SR;
            EventTms = SmrData.EvData.EvECGP_Cl;
        end

        % Define the path to subject feature folder
        subject_feature_folder = fullfile(data_dir, feature_name,  subfname, sprintf('sub-%s', subject));

        % Define the path to subject results folder
        subject_results_folder = fullfile(results_dir, feature_name,  subfname, sprintf('sub-%s', subject));



        %% ================================== HRV ================================
        if ismember('Feature Extraction ECG', steps)
            % Calculate the HRV (Heart-Rate Variability from filtered ECG signal
            disp('Calculating HRV...');

            % For HRV we will use the Raw ECG Signal (not the cleaned one)
            IBI.(subfname){sub, :} = SmrData.EvData.ECGcomp(1,:);
            HR.(subfname){sub, :} = SmrData.EvData.ECGcomp(2,:);


            % Calculate RMSSD HRV (Time-Range)
            HRV.rmssd_avg.(subfname)(1,sub) = sqrt(mean(diff(IBI.(subfname){sub, :}).^2));% average HRV rmssd over all IBI


            % Calculate Rolling RMSSD HRV
            % RMSSD = Root Mean Square of Successive Differences.
            % comuptes a rolling RMSSD without overlap

            dRR = diff(IBI.(subfname){sub, :}).^2;
            averaged_window = NaN(length(IBI.(subfname){sub, :}),window_length_hrv);
            for j=1:window_length_hrv
                averaged_window(j+1:end,j) = dRR(1:end-j+1);
            end
            samplesize = sum(~isnan(averaged_window),2);
            hrv_rmssd = sqrt(sum(averaged_window,2)./(samplesize-1+1)); % the +1 at the end is for normalization

            if show_plots
                sample_time = max(SmrData.EvData.EvECG(:, end)) / length(hrv_rmssd);
                f1 = figure;
                plot(hrv_rmssd.*1000, 'Color', colors.ECG.HRV)
                yline(HRV.rmssd_avg.(subfname)(1, sub)*1000, "--k", 'HRV Avg', 'LabelHorizontalAlignment', 'right');
                new_xticks = 0:50:length(hrv_rmssd);
                new_xtick_labels = round(new_xticks * sample_time/5)*5;
                xlim([0, length(hrv_rmssd)]);
                set(gca, 'XTick', new_xticks, 'XTickLabel', new_xtick_labels);
                xlabel('Recording Length (in sec)');
                ylabel('HRV Length (in ms)');
                title(strcat(subfname, ' HRV (RMSSD) of sub ', num2str(subject), ' avg HRV: ', num2str(round(HRV.rmssd_avg.(subfname)(1, sub) * 1000, 2)), ' ms'))

                % saving
                gr1 = fullfile(subject_results_folder, [subject, '_', subfname, 'HRV-RMSSD_win_', num2str(window_length_hrv),' sample.png']);
                try
                    exportgraphics(f1, gr1, "Resolution", 300);
                catch ME
                    warning("Failed to save the plot: %s", ME.message);
                end
            end
        end

        %% =============== 2. TIME FREQUENCY DECOMPOSITION EEG & LFP ==============
        if ismember('Feature Extraction EEG', steps)
            for c = 1:numel(channels)
                if seperateSTN
                    channels{8} = LfpElec.(subject){1};
                    channels{9} = LfpElec.(subject){2};
                end
                channel = channels{c};

                if epoch
                    fprintf('****************** EPOCH for %s %s...****************\n', subject, channel);

                    ChDta = SmrData.WvDataCleaned(c, :);

                    % HIGH PASS FILTER
                    if Fhp > 0
                        ChDta=ft_preproc_highpassfilter(ChDta,SR,Fhp,4,'but', 'twopass'); % twopass
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
    
                    % Epoch the Data
                    [EventTms,EvData,TmAxis]=GetEvTimeAndData(EventTms,ChDta,dtTime,tWidth,tOffset);
                    [nEvs,nData]=size(EvData);
                  

                    %  Baseline Correction
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

                    fprintf('****************** TIME FREQ DECOMP for %s %s, med: %s ...****************\n', subject, channel, subfname);
                    %TFR = []; % time frequency representation
                    ChsCmxEvFrTm = [];

                    % Get IIR Peak
                    for iev=1:nEvs
                        dx=squeeze(EvData(iev,:));
                        for ifr=1:length(Frqs)
                            vfr=Frqs(ifr);
                            df=IIRPeak_Flt(dx,NewSR,vfr,BandWidth,Qfac,FltPassDir);
                            ChsCmxEvFrTm(iev,ifr,:)=hilbert(df); % ChannelxFreqxTime
                        end
                    end

                    % EXTRACTION OF POWER AND PHASE
                    [nEvs, nFrs,nData]=size(ChsCmxEvFrTm);
                    ChsAllFrsTmSpc=zeros(nEvs,nFrs,nData);
                    ChsAllFrsTmPha=zeros(nEvs,nFrs,nData);
                    for iev=1:nEvs
                        for ifr=1:nFrs
                            xlb=squeeze(ChsCmxEvFrTm(iev,ifr,:));
                            df=abs(xlb);
                            ChsAllFrsTmSpc(iev,ifr,:)=df; % Power (eventxfreqxpower)
                            ChsAllFrsTmPha(iev,ifr,:)=angle(xlb); %Phase (eventxfreqxphase)
                        end
                    end

                end

                if ~epoch

                    fprintf('****************** TIME FREQ DECOMP for %s %s...****************\n', subject, channel);
                    %TFR = []; % time frequency representation
                    ChsCmxEvFrTm = [];
                    ChDta = SmrData.WvDataCleaned(c, :);


                    % HIGH PASS FILTER
                    if Fhp > 0
                        ChDta=ft_preproc_highpassfilter(ChDta,SR,Fhp,4,'but', 'twopass'); % twopass
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


                    % TFR HILBERT
                    for ifr=1:length(Frqs)
                        vfr=Frqs(ifr);
                        df=IIRPeak_Flt(ChDta,NewSR,vfr,BandWidth,Qfac,FltPassDir);
                        ChsCmxEvFrTm(ifr,:)=hilbert(df); % ChannelxFreqxTime
                    end

                    % EXTRACTION OF POWER AND PHASE
                    [nFrs,nData]=size(ChsCmxEvFrTm);
                    ChsAllFrsTmSpc=zeros(nFrs,nData);
                    ChsAllFrsTmPha=zeros(nFrs,nData);
                    for ifr=1:nFrs
                        xlb=squeeze(ChsCmxEvFrTm(ifr,:));
                        df=abs(xlb);
                        ChsAllFrsTmSpc(ifr,:)=df; % Power (freqxpower)
                        ChsAllFrsTmPha(ifr,:)=angle(xlb); %Phase (freqxphase)
                    end

                end

                % % DOWNSAMPLE Power and Phase
                % if NewSR > 0
                %     FsOrigin=SR;
                %     if  FsOrigin ~=  NewSR
                %         [fsorig, fsres] = rat(FsOrigin/NewSR);
                %         ChsAllFrsTmSpcDS = resample(ChsAllFrsTmSpc,fsres,fsorig);
                %         ChsAllFrsTmPhaDS = resample(ChsAllFrsTmPha,fsres,fsorig);
                %         dtTime=1/NewSR;
                %     end
                %     NewSR=1.0/dtTime;
                % end

                % CREATE TFR STRUCT
                TFR.(channel).pow = ChsAllFrsTmSpc;
                TFR.(channel).phase = ChsAllFrsTmPha;
                TFR.freqs = Frqs;
                TFR.SR = NewSR;
                TFR.times = TmAxis;
            end



            %% =========================== SAVING DATA ===============================
            if epoch & baseline
                save_path = fullfile(data_dir, 'tfr', Hz_dir, [subject,  '_TFR-EPOCH_', subfname ,'_Rest_Hilbert_Freq=', num2str(stfr),'-', num2str(enfr),'_bin=', num2str(dfr),'Hz_DS=', num2str(NewSR),'_HP=', num2str(Fhp),'Hz_EP=-',num2str(tOffset), 'to', num2str(tWidth-tOffset) ,'_BSL=', num2str(baseline_win(1)),'to', num2str(baseline_win(2)),'s.mat']);
            elseif epoch
               save_path = fullfile(data_dir, 'tfr', Hz_dir,[subject,  '_TFR-EPOCH_', subfname ,'_Rest_Hilbert_Freq=', num2str(stfr),'-', num2str(enfr),'_bin=', num2str(dfr),'Hz_DS=', num2str(NewSR),'_HP=', num2str(Fhp),'Hz_EP=-',num2str(tOffset), 'to', num2str(tWidth-tOffset), '.mat']);
            else
                save_path = fullfile(data_dir, 'tfr', Hz_dir, [subject,  '_TFR_', subfname ,'_Rest_Hilbert_Freq=', num2str(stfr),'-', num2str(enfr),'_bin=', num2str(dfr),'Hz_DS=', num2str(NewSR),'_HP=', num2str(Fhp),'Hz.mat']);
            end
            save(save_path, 'TFR', '-v7.3');
        end
    


        %% Extract ECG Epochs

        % DOES NOT WORK YET - PREPROCESSING MUST INCLUDE THE CLEANED ECG
        % SIGNAL FOR THAT 
        if ismember('ECG Epoch',steps)

            
            [EventTms,EvData,TmAxis]=GetEvTimeAndData(EventTms,ChDta,dtTime,tWidth,tOffset);
                    [nEvs,nData]=size(EvData);




        end
    end
end

%% =========================== 3. AVERAGES ================================

%% HRV AVERAGE / DESCRIPTIVE STATS

% Extractiung Mean, Median, Mode and Standard Deviation for the RMSSD HRV For all Participants
% for fn = 1:2 % Med On and Med Off
%     subfname = subfnames{fn};
%     HRV.mean = mean(HRV.rmssd_avg.(subfname)); % Mean
%     HRV.std = std(HRV.rmssd_avg.(subfname)); % STD
%     HRV.median = median(HRV.rmssd_avg.(subfname)); % Median
%     HRV.mode = mode(HRV.rmssd_avg).(subfname); % Mode
%     HRV.variance = var(HRV.rmssd_avg).(subfname); % Variance
%     %HRV.SEM = std(HRV.rmssd_avg)/sqrt(length(HRV.rmssd_avg));  % Standard Error
%     %ts = tinv([0.025  0.975],length(HRV.rmssd_avg)-1); % T-Score
%     %HRV.CI = mean(HRV.rmssd_avg) + ts*HRV.SEM; % 95% CI
%
% end
% % figure
% errorbar(HRV.mean, 1,  HRV.CI, 'horizontal', 'o')
% ylim([0.5 1.5]); % Keep y-axis focused on the single point
% yticks([]); % Remove y-axis ticks for a cleaner look
% xlim([x_axis_min - 0.1*ci_95, x_axis_max + 0.1*ci_95]); % Adjust x-axis limits to center the point
% ylabel('Mean RMSSD HRV');
% title('Mean RMSSD HRV with 95% Confidence Interval');

save_path = fullfile(avg_features_folder, ['Averages_HRV-IBI-HR_Rest_nsub=', num2str(numel(subjects.goodHeartMOff)),'.mat']); % med_name needs an alternative here
save(save_path, 'HRV', 'IBI', 'HR');

%% 3a. HRV Averages
disp('feature extraction done!');












