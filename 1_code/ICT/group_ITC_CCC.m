% function [] = group_ITC_CCC(subjects, data_dir, results_dir)

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
% 2. ITC ANALYSIS
% 3. CCC ANALYSIS

CCCEegLfp.L1= {'STNl', 'C3'};
CCCEegLfp.L2= {'STNl', 'F3'};
CCCEegLfp.L3= {'STNl', 'P3'};
CCCEegLfp.R1= {'STNr', 'C4'};
CCCEegLfp.R2= {'STNr', 'F4'};
CCCEegLfp.R3= {'STNr', 'P4'};

%% ============= SET GLOABAL VARIABLES AND PATHS =========================
%clear all
%close all

subfnames = fieldnames(subjects);

% Define if plots are to be shown
show_plots = false;

% Define feature extraction steps to perform
steps = {'Load Data', 'Epoch and Timelock Data'};

% Define Time Window
tWidth   = 1.5;
tOffset  = 0.4;

% Define folder variables
epoch_name = 'epoch';  % feature extraction folder (inside derivatives)
average_name = 'avg';

% Define the channels 
channels = {'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'Pz', 'STNl', 'STNr'};

tCircMean=0.01; % for By TRials calc

CCC = [];

permstats = false;
numPerms = 500;

%% ============================ 1. LOAD DATA =============================
disp("************* STARTING EPOCH AND TIMELOCKING *************");

for fn = 2:2 % MedOn
    subfname = subfnames{fn};
    nSub = numel(subjects.goodHeartMOff);

    if ismember('Load Data', steps)
        % Load the Epoched Data
        fprintf('Loading AVG TFR Data for ITC and CCC Analysis\n');
        pattern = fullfile(data_dir, epoch_name, average_name, ['PHS-AVG_', subfname, 'n=', num2str(nSub), '*']);
        files = dir(pattern);
        filename = fullfile(files(1).folder, files(1).name);
        load(filename, 'PHASE', 'Params');

        fprintf('Loading AVG ECG Data\n');
        pattern = fullfile(data_dir, epoch_name, average_name, ['ECG-AVG_', subfname, 'n=', num2str(nSub), '*']);
        files = dir(pattern);
        filename = fullfile(files(1).folder, files(1).name);
        load(filename, 'AVGECG');

        fprintf('Loading IBI Data\n');
        pattern = fullfile(data_dir, 'features', average_name, ['AVERAGES_HRV', '*']);
        files = dir(pattern);
        filename = fullfile(files(1).folder, files(1).name);
        load(filename, 'IBI');

    end
   

    % Initialize the basic
    SR = Params.SR;
    dtTime=1/SR;
    freqs = Params.freqs;
    

    [nEvent, Frqs, nTm] = size(PHASE.C3.submean);
    TmAxis=zeros(nTm,1)';
    s=-1*tOffset;  for i=1:nTm;  TmAxis(i)=s;  s=s+dtTime;  end
    time_bins = TmAxis;
%% ============================ 2. Plot AVG ECG ANALYSIS =============================



    %% ============================ 2. ITC ANALYSIS =============================
    disp("************* STARTING ITC ANALYSIS OVER ALL CHANNELS *************");


    FrsTmItcAll = zeros(numel(channels), numel(freqs), numel(time_bins));

    %Loop through all channels
    for c = 1:numel(channels)
        channel = channels{c};
        fprintf('****************** Processing Channel %s...****************\n', channel);

        % Get ITC for the channel
        [FrsTmItc]=Get_PSI_ByTrials_ITC(PHASE.(channel).submean,SR,tCircMean);

        % Scale the ITc to the relative ITC of the channel 
        meanFrsTmItc = mean(mean(FrsTmItc,1),2);
        relFrsTmItc = FrsTmItc/meanFrsTmItc;

        f1=figure;
        %set(f1,'Position',[2 50 1140 946]);
        subplot(2,1,1)
        plot(time_bins, AVGECG.mean', 'Color', 'k'); hold on
        set(gca,'Position',[0.1300 0.5838 0.72 0.3])
        xline(0, "--k", 'LineWidth', 2);
        title(sprintf('Average ECG over all Subjects'))
        hold off
        subplot(2,1,2)
        imagesc(time_bins,freqs,relFrsTmItc);axis xy;
        colormap('jet');
        xline(0, "--k", 'LineWidth', 2);
        colorbar;
        title(sprintf('AVG ITC for %s', channel))

        % Save the ITC in bigger Matrix
        FrsTmItcAll(c,:,:) = FrsTmItc; % ChannelsxFreqsxTime
        relFrsTmItcAll(c,:,:) = relFrsTmItc; % ChannelsxFreqsxTime
        
        if permstats
            [ITCzscores]=ITC_permutation_test(FrsTmItc, relFrsTmItcAll, IBI.(subfname){1}, numPerms, freqs, time_bins, SR, PHASE.(channel).submean);

            itc_zscores_thresh = (ITCzscores > 4) | (ITCzscores < -4);

            figure;
            imagesc(time_bins,freqs,relFrsTmItc);axis xy;
            colormap('jet');
            hold on;
            contour(time_bins,freqs,itc_zscores_thresh, 1, 'linecolor', 'k', 'linewidth', 1.5);
            title(sprintf('AVG ITC for %s', channel))
        end
        
       

        fprintf('ITC for %s DONE...\n', channel);
    end

   % Left Hemisphere ITC 
   lhem = cat(4, PHASE.F3.submean, PHASE.C3.submean,  PHASE.P3.submean); % Concatenate along 4th dimension
   lhem_avg = mean(lhem, 4);

   [FrsTmItc]=Get_PSI_ByTrials_ITC(lhem_avg,SR,tCircMean);

   % Scale the ITc to the relative ITC of the channel
   meanFrsTmItc = mean(mean(FrsTmItc,1),2);
   relFrsTmItc = FrsTmItc/meanFrsTmItc;

   f2=figure;
   %set(f1,'Position',[2 50 1140 946]);
   subplot(2,1,1)
   plot(time_bins, AVGECG.mean', 'Color', 'k'); hold on
   set(gca,'Position',[0.1300 0.5838 0.72 0.3])
   xline(0, "--k", 'LineWidth', 2);
   title(sprintf('Average ECG over all Subjects'))
   hold off
   subplot(2,1,2)
   imagesc(time_bins,freqs,relFrsTmItc);axis xy;
   colormap('jet');
   xline(0, "--k", 'LineWidth', 2);
   colorbar;
   title(sprintf('AVG ITC for Left Hemisphere (F3, C3 & P3)'))

    % Right Hemisphere ITC 
   rhem = cat(4, PHASE.F4.submean, PHASE.C4.submean,  PHASE.P4.submean); % Concatenate along 4th dimension
   rhem_avg = mean(rhem, 4);

   [FrsTmItc]=Get_PSI_ByTrials_ITC(rhem_avg,SR,tCircMean);

   % Scale the ITc to the relative ITC of the channel
   meanFrsTmItc = mean(mean(FrsTmItc,1),2);
   relFrsTmItc = FrsTmItc/meanFrsTmItc;

   f3=figure;
   %set(f1,'Position',[2 50 1140 946]);
   subplot(2,1,1)
   plot(time_bins, AVGECG.mean', 'Color', 'k'); hold on
   set(gca,'Position',[0.1300 0.5838 0.72 0.3])
   xline(0, "--k", 'LineWidth', 2);
   title(sprintf('Average ECG over all Subjects'))
   hold off
   subplot(2,1,2)
   imagesc(time_bins,freqs,relFrsTmItc);axis xy;
   colormap('jet');
   xline(0, "--k", 'LineWidth', 2);
   colorbar;
   title(sprintf('AVG ITC for Right Hemisphere (F4, C4 & P4)'))




      %% ============================ 3. CCC ANALYSIS =============================
    disp("************* STARTING CCC ANALYSIS OVER ALL CHANNELS *************");
    
    for c = 1:numel(fieldnames(CCCEegLfp))
        CCCfn = fieldnames(CCCEegLfp);
        currfn = CCCfn{c};
        channel1 = CCCEegLfp.(currfn){1};
        channel2 = CCCEegLfp.(currfn){2};
        fprintf('****************** Processing Channel %s and %s ...****************\n', channel1, channel2);

        % Get CCC for the channel
        [FrsTmPsiTrial,FrsTmPhaTrial]=Get_PSI_ByTrials(PHASE.(channel1).submean,PHASE.(channel2).submean,SR,tCircMean);
        
        figure;
        subplot(1,2,1);
        imagesc(freqs,time_bins,FrsTmPsiTrial); axis xy;
        subplot(1,2,1);
        imagesc(freqs,time_bins,FrsTmPhaTrial); axis xy;

        % Save the CCC in a struct
        channame = append(channel1,'+', channel2)
        CCC.(channame) = FrsTmPsiTrial;

        fprintf('CCC for %s and %s DONE...\n', channel1, channel2);
    end

end