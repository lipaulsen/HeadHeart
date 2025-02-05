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
% - EEG & LFP: Epoched TFR (data_dir, 'tfr', Hz_dir, [subject,  '_TFR-EPOCH_', medname ,'_Rest_Hilbert_Freq=',
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

% GOOD HEART STATUS
% patients with arrithmy have been excluded after their ECG was
% investigated
GoodHeart = 1;

% get the channel info into the shape of cells
AllSubsChansRaw = cellfun(@(x) strsplit(x, ', '), {subject_info.channels_raw}, 'UniformOutput', false);
AllSubsChansStn = cellfun(@(x) strsplit(x, ', '), {subject_info.channels}, 'UniformOutput', false);

% filter which subjects and which channels you want

if MedOff == true & allsubs == true & GoodHeart % All Subs that are MedOff with good Heart
    subjects = string({subject_info([subject_info.MedOff] == 1& [subject_info.goodHeart_MedOff] == 1).ID});
elseif MedOn == true & allsubs == true & GoodHeart == 1
    subjects = string({subject_info([subject_info.MedOn] == 1& [subject_info.goodHeart_MedOn] == 1).ID});
elseif MedOn == true & newsubs == true % Only New Subs that are MedOn
    subjects = string({subject_info([subject_info.new] == 1 & [subject_info.MedOn] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.new] == 1 & [subject_info.MedOn] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.new] == 1 & [subject_info.MedOn] == 1);
elseif MedOff == true & newsubs == true  % Only New Subs that are MedOff
    subjects = string({subject_info([subject_info.new] == 1 & [subject_info.MedOff] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.new] == 1 & [subject_info.MedOff] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.new] == 1 & [subject_info.MedOff] == 1);
elseif MedOn == true & oldsubs == true  % Only Old Subs that are MedOn
    subjects = string({subject_info([subject_info.new] == 0 & [subject_info.MedOn] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.new] == 0 & [subject_info.MedOn] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.new] == 0 & [subject_info.MedOn] == 1);
elseif MedOff == true & oldsubs == true  % Only Old Subs that are MedOff
    subjects = string({subject_info([subject_info.new] == 0 & [subject_info.MedOff] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.new] == 0 & [subject_info.MedOff] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.new] == 0 & [subject_info.MedOff] == 1);
elseif MedOn == true & allsubs == true  % All Subs that are MedOn
    subjects = string({subject_info([subject_info.MedOn] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.MedOn] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.MedOn] == 1);
elseif MedOff == true & allsubs == true % All Subs that are MedOff
    subjects = string({subject_info([subject_info.MedOff] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.MedOff] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.MedOff] == 1);
end

%=========================================================================


% Define if plots are to be shown
show_plots = false;

% Define feature extraction steps to perform
steps = {'Load Data','ECG Data'}; %'Feature Extraction EEG','ECG Data', 'Feature Extraction ECG',

% Define folder variables
preprocessed_name = 'preprocessed';  % preprocessed folder (inside derivatives)
averaged_name = 'avg';  % averaged data folder (inside preprocessed)
feature_name = 'features';  % feature extraction folder (inside derivatives)

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

% Create color palette for plots
colors.ECG.IBI = [0.9569,    0.9451,    0.8706];    % Creme
colors.ECG.HRV = [0.8784,    0.4784,    0.3725];   % Pink
colors.ECG.LF_HRV = [0.2392    0.2510    0.3569];  % Light Orange
colors.ECG.HF_HRV = [0.5059    0.6980    0.6039];   % Dark Orange
colors.EEG.delta = [ 0.9490    0.8000    0.5608];  % Yellow

%If we use left and right STN as seperate subjects put this as true
%(increases subject size by 2)
seperateSTN = true;

% Suppress excessive logging if using FieldTrip
ft_defaults; % If using FieldTrip

% Downsample Parameters
NewSR=300;
stfr=0.5;   enfr=30; dfr=0.2;
Frqs=stfr:dfr:enfr;
if NewSR < 2*enfr; NewSR=(enfr+2)*2; end

% HPF Parameter
Fhp = 2;
Hz_dir = '2Hz';
FltPassDir='twopass'; % onepass  twopass

% Hilbert TFR Parameters
BandWidth=2; % BandWidth in Hz;
Qfac     =2; % Attenuation in db(-Qfac)
WaveletnCyc=6;
WaveletgWidth=3;

% Baseline und Epoch Parameter 
ChsCmxEvFrTm =[]; % 
epoch = true;
baseline = true;
baseline_win = [-0.3 -0.1]; % Baseline Time Window

% Define TFR Time Window
tWidth   = 0.9;
tOffset  = 0.3;

LfpElec.SG041 = {'L3', 'R3'};
LfpElec.SG043  = {'L4', 'R1'};
LfpElec.SG044  = {'L1', 'R3'};
LfpElec.SG045  = {'L4', 'R1'}; % NEW
LfpElec.SG046  = {'L4', 'R1'};
LfpElec.SG047  = {'L3', 'R4'};
LfpElec.SG050 = {'L3', 'R3'};
LfpElec.SG052  = {'L4', 'R2'};
LfpElec.SG056  = {'L4', 'R1'};
LfpElec.SG060  = {'L2', 'R1'}; % NEW
LfpElec.SG078  = {'L1', 'R1'}; % NEW
LfpElec.SG079  = {'L2', 'R7'}; % NEW
LfpElec.KS28  = {'L3', 'R8'}; % NEW
LfpElec.KS29  = {'L8', 'R7'}; % NEW

%% ============================ 1. LOAD DATA =============================
disp("************* STARTING Feature Extraction  *************");

% Initialize the matrix for all HRV measures
HRV.rmssd_avg = []; %
HRV.rmssd_tits{1} = {'HRV Average of subj'};

IBI = [];
HR = [];

for med = 1%:2 % MedOn
  
    if MedOn == true
        medname = 'MedOn';
    elseif MedOff == true
        medname = 'MedOff';
    end

    for sub = 1:numel(subjects) %numel(subjects.goodHeartMOff) % BE AWARE THAT THIS EXCLUDES PATIENTS WITH ARRITHYMIAS
        % Extract the subject
        subject = subjects{sub};

        if ismember('Load Data', steps)

            fprintf('Loading Data of  subject %s number %i of %i\n', subject, sub, numel(subjects));
            
            
            pattern = fullfile(data_dir, 'preproc', 'all', [subject, '_', preprocessed_name, '_', medname, '_BPReref_', '*']);
            files = dir(pattern);
            filename = fullfile(files(1).folder, files(1).name);
            load(filename, 'SmrData');
            % Load subject data
            % subject_data = fullfile(data_dir, preprocessed_name, medname, ['sub-', subject], [subject, '_preprocessed_', medname, '_Rest.mat']);
            % load(subject_data, 'SmrData');

            SR = SmrData.SR;
            EventTms = SmrData.EvData.EvECGP_Cl;
        end

        % Define the path to subject feature folder
        subject_feature_folder = fullfile(data_dir, feature_name,  medname, sprintf('sub-%s', subject));

        % Define the path to subject results folder
        subject_results_folder = fullfile(results_dir, feature_name,  medname, sprintf('sub-%s', subject));



        %% ================================== HRV ================================
        if ismember('Feature Extraction ECG', steps)
            % Calculate the HRV (Heart-Rate Variability from filtered ECG signal
            disp('Calculating HRV...');

            % For HRV we will use the Raw ECG Signal (not the cleaned one)
            IBI.(medname){sub, :} = SmrData.EvData.ECGcomp(1,:);
            HR.(medname){sub, :} = SmrData.EvData.ECGcomp(2,:);


            % Calculate RMSSD HRV (Time-Range)
            HRV.rmssd_avg.(medname)(1,sub) = sqrt(mean(diff(IBI.(medname){sub, :}).^2));% average HRV rmssd over all IBI


            % Calculate Rolling RMSSD HRV
            % RMSSD = Root Mean Square of Successive Differences.
            % comuptes a rolling RMSSD without overlap

            dRR = diff(IBI.(medname){sub, :}).^2;
            averaged_window = NaN(length(IBI.(medname){sub, :}),window_length_hrv);
            for j=1:window_length_hrv
                averaged_window(j+1:end,j) = dRR(1:end-j+1);
            end
            samplesize = sum(~isnan(averaged_window),2);
            hrv_rmssd = sqrt(sum(averaged_window,2)./(samplesize-1+1)); % the +1 at the end is for normalization

            if show_plots
                sample_time = max(SmrData.EvData.EvECG(:, end)) / length(hrv_rmssd);
                f1 = figure;
                plot(hrv_rmssd.*1000, 'Color', colors.ECG.HRV)
                yline(HRV.rmssd_avg.(medname)(1, sub)*1000, "--k", 'HRV Avg', 'LabelHorizontalAlignment', 'right');
                new_xticks = 0:50:length(hrv_rmssd);
                new_xtick_labels = round(new_xticks * sample_time/5)*5;
                xlim([0, length(hrv_rmssd)]);
                set(gca, 'XTick', new_xticks, 'XTickLabel', new_xtick_labels);
                xlabel('Recording Length (in sec)');
                ylabel('HRV Length (in ms)');
                title(strcat(medname, ' HRV (RMSSD) of sub ', num2str(subject), ' avg HRV: ', num2str(round(HRV.rmssd_avg.(medname)(1, sub) * 1000, 2)), ' ms'))

                % saving
                gr1 = fullfile('/Volumes/LP3/HeadHeart/2_results/features/HRV_RMSSD', [subject, '_', medname, 'HRV-RMSSD_win_', num2str(window_length_hrv),' sample.png']);
                try
                    exportgraphics(f1, gr1, "Resolution", 300);
                catch ME
                    warning("Failed to save the plot: %s", ME.message);
                end
            end
        end

        %% =============== 2. TIME FREQUENCY DECOMPOSITION EEG & LFP ==============
        if ismember('Feature Extraction EEG', steps)

            % Use BPReref Data
            BPReref = false; BPRerefTit = 'BPReref';
            BPRerefHi = true; BPRerefHiTit = 'BPRerefHi';
            BPRerefLw = false; BPRerefLwTit = 'BPRerefLow';
            BPRerefBest = false; BPRerefBestTit = 'BPRerefBest';

            if BPReref     
                channels = FltSubsChansStn{sub};
            else
                channels = FltSubsChansStn{sub};
            end

            % KS29 has no EEG recordings in MedOn so we delete those values
            if strcmp(medname,'MedOn') & strcmp(subject,'KS29')
                channels = {FltSubsChansStn{sub}{end-1:end}};
                SmrData.WvDataBPRerefHi(1:6,:) = [];
                SmrData.WvDataBPRerefLow (1:6,:) = [];
            end


            for c = 1:numel(channels)
                % if seperateSTN
                %     channels{8} = LfpElec.(subject){1};
                %     channels{9} = LfpElec.(subject){2};
                % end
                channel = channels{c};


                if epoch
                    fprintf('****************** EPOCH for %s %s...****************\n', subject, channel);

                    if BPReref & BPRerefHi
                    ChDta = SmrData.WvDataBPRerefHi(c, :);
                    elseif BPReref & BPRerefLw
                    ChDta = SmrData.WvDataBPRerefLow(c, :);
                    elseif BPReref & BPRerefBest
                    ChDta = SmrData.WvDataBPRerefLow(c, :); % Hier filter oder vielleicht doch schon im Preprocessing später!!
                    else 
                    ChDta = SmrData.WvDataCleaned(c, :);
                    end

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

                       fprintf('****************** Baseline Correction for %s %s med: %s ...****************\n', subject, channel, medname);
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

                    fprintf('****************** TIME FREQ DECOMP for %s %s, med: %s ...****************\n', subject, channel, medname);
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

                    f2 = figure; 
                    imagesc(TmAxis,Frqs,squeeze(ChsAllFrsTmSpc(1,:,:))), axis xy
                    title(['Power Plot, Sub: ', subject, ' med: ', medname, ' chan: ', channel])
                    colorbar; 

                    if BPReref & BPRerefHi
                        gr2 = fullfile( results_dir, 'power/2Hz/ss/BPRerefHi' , ['POWER_', subject, '_', medname, '_', channel, '_', BPRerefHiTit, '.png']);
                    elseif BPReref & BPRerefLw
                        gr2 = fullfile( results_dir, 'power/2Hz/ss/BPRerefLow' , ['POWER_', subject, '_', medname, '_', channel, '_', BPRerefLwTit, '.png']);
                    elseif BPReref & BPRerefBest
                        gr2 = fullfile( results_dir, 'power/2Hz/ss' , ['POWER_', subject, '_', medname, '_', channel, '_', BPRerefBestTit, '.png']);
                    else
                        gr2 = fullfile( results_dir, 'power/2Hz/ss' , ['POWER_', subject, '_', medname, '_', channel, '.png']);
                    end
                    exportgraphics(f2, gr2, "Resolution", 300);

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



                % CREATE TFR STRUCT
                TFR.(channel).pow = ChsAllFrsTmSpc;
                TFR.(channel).phase = ChsAllFrsTmPha;
                TFR.freqs = Frqs;
                TFR.SR = NewSR;
                TFR.times = TmAxis;
            end



            %% =========================== SAVING DATA ===============================
            if epoch & baseline & BPReref & BPRerefHi
                save_path = fullfile(data_dir, 'tfr', Hz_dir, [subject,  '_TFR-EPOCH_', medname ,'_Rest_', BPRerefHiTit, '_Hilbert_Freq=', num2str(stfr),'-', num2str(enfr),'_bin=', num2str(dfr),'Hz_DS=', num2str(NewSR),'_HP=', num2str(Fhp),'Hz_EP=-',num2str(tOffset), 'to', num2str(tWidth-tOffset) ,'_BSL=', num2str(baseline_win(1)),'to', num2str(baseline_win(2)),'s.mat']);
            elseif epoch & baseline & BPReref & BPRerefLw
                save_path = fullfile(data_dir, 'tfr', Hz_dir, [subject,  '_TFR-EPOCH_', medname ,'_Rest_', BPRerefLwTit, '_Hilbert_Freq=', num2str(stfr),'-', num2str(enfr),'_bin=', num2str(dfr),'Hz_DS=', num2str(NewSR),'_HP=', num2str(Fhp),'Hz_EP=-',num2str(tOffset), 'to', num2str(tWidth-tOffset) ,'_BSL=', num2str(baseline_win(1)),'to', num2str(baseline_win(2)),'s.mat']);
            elseif epoch & baseline & BPReref & BPRerefBest
                save_path = fullfile(data_dir, 'tfr', Hz_dir, [subject,  '_TFR-EPOCH_', medname ,'_Rest_', BPRerefBestTit, '_Hilbert_Freq=', num2str(stfr),'-', num2str(enfr),'_bin=', num2str(dfr),'Hz_DS=', num2str(NewSR),'_HP=', num2str(Fhp),'Hz_EP=-',num2str(tOffset), 'to', num2str(tWidth-tOffset) ,'_BSL=', num2str(baseline_win(1)),'to', num2str(baseline_win(2)),'s.mat']);
            elseif epoch & baseline
                save_path = fullfile(data_dir, 'tfr', Hz_dir, [subject,  '_TFR-EPOCH_', medname ,'_Rest_Hilbert_Freq=', num2str(stfr),'-', num2str(enfr),'_bin=', num2str(dfr),'Hz_DS=', num2str(NewSR),'_HP=', num2str(Fhp),'Hz_EP=-',num2str(tOffset), 'to', num2str(tWidth-tOffset) ,'_BSL=', num2str(baseline_win(1)),'to', num2str(baseline_win(2)),'s.mat']);
            elseif epoch
                save_path = fullfile(data_dir, 'tfr', Hz_dir,[subject,  '_TFR-EPOCH_', medname ,'_Rest_Hilbert_Freq=', num2str(stfr),'-', num2str(enfr),'_bin=', num2str(dfr),'Hz_DS=', num2str(NewSR),'_HP=', num2str(Fhp),'Hz_EP=-',num2str(tOffset), 'to', num2str(tWidth-tOffset), '.mat']);
            else
                save_path = fullfile(data_dir, 'tfr', Hz_dir, [subject,  '_TFR_', medname ,'_Rest_Hilbert_Freq=', num2str(stfr),'-', num2str(enfr),'_bin=', num2str(dfr),'Hz_DS=', num2str(NewSR),'_HP=', num2str(Fhp),'Hz.mat']);
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
% for med = 1:2 % Med On and Med Off
%     medname = subfnames{med};
%     HRV.mean = mean(HRV.rmssd_avg.(medname)); % Mean
%     HRV.std = std(HRV.rmssd_avg.(medname)); % STD
%     HRV.median = median(HRV.rmssd_avg.(medname)); % Median
%     HRV.mode = mode(HRV.rmssd_avg).(medname); % Mode
%     HRV.variance = var(HRV.rmssd_avg).(medname); % Variance
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
save_path = fullfile('/Volumes/LP3/HeadHeart/0_data/features/avg', ['Averages_HRV-IBI-HR_Rest_nsub=', num2str(numel(subjects)),'.mat']); % med_name needs an alternative here
save(save_path, 'HRV', 'IBI', 'HR');


%% ============================ 4. AVERAGE DATA OVER ALL SUBS =============================


% WARNNING IF YOU EVER USE THIS CODE HTERE WERE HEAVY ADJUSTMENTS MADE SO
% PLEASE BE AWARE AND TAKE CARE - I TRIED TO WRITE IN THE COMMENT WHERE A
% CRITICAL ADJUSTMENT WAS MADE 



if ismember('ECG Data', steps)
    disp("************* AVERAGE the ECG DATA  *************");

    NewSR = 300;

    epochECGall = cell(numel(subjects), 1)

 
    for f = 1%:2 % MedOn
        if MedOn == true
            medname = 'MedOn';
        elseif MedOff == true
            medname = 'MedOff';
        end
        for s = 1:numel(subjects) % BE AWARE THAT THIS EXCLUDES PATIENTS WITH ARRITHYMIAS

            % Extract the subject
            subject = subjects{s};


            fprintf('Loading Data of subject %s number %i of %i\n', subject, s, numel(subjects.goodHeartMOff));


             % Load the the cleaned ECG R Peaks Data
            pattern = fullfile(data_dir, 'itc', 'evecg' ,[subject, '_', medname, '*']);
            files = dir(pattern);
            filename = fullfile(files(1).folder, files(1).name);
            load(filename, 'EvEcgData');

            epochECGall{s} = EvEcgData;

        end

        % 
        %     % Load the the cleaned ECG R Peaks Data
        %     pattern = fullfile(data_dir, 'preproc', 'all', [subject, '_preprocessed_', medname, '*']);
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


        save_path = fullfile(data_dir, epoch_name, 'avg', ['ECG-AVG_', medname ,'n=', num2str(nSub),...
            '_Rest_Hilbert_Freq=', num2str(freqs(1)),'-', num2str(freqs(end)),'Hz_bin=', num2str(mean(diff(freqs))),...
            'HZ-Epoch=', num2str(TmAxis(1)),'to',num2str(TmAxis(end)),'s_BSL=', num2str(baseline),'.mat']);
        save(save_path, 'AVGECG', '-v7.3');

    end
end



%% 3a. HRV Averages
disp('feature extraction done!');












