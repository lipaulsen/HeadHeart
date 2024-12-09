% function [] = plot_PSD(subjects, data_dir, results_dir)

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

% Define if plots are to be shown
show_plots = false;

%If we use left and right STN as seperate subjects put this as true
%(increases subjectsize by 2)
seperateSTN = true;

% Define feature extraction steps to perform
steps = {'Calc Single Subject ITC'}; %'Plot SubAvg PermStats', 'Calc Single Subject ITC', 'Plot SubAvg ITC', 'Plot Power'

% Define folder variables
epoch_name = 'epoch';  % feature extraction folder (inside derivatives)


channels = {'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'Pz', 'STNl', 'STNr'};

LfpElec.SG041 = {'L3', 'R3'};
LfpElec.SG043  = {'L4', 'R1'};
LfpElec.SG046  = {'L4', 'R1'};
LfpElec.SG047  = {'L3', 'R4'};
LfpElec.SG050 = {'L3', 'R3'};
LfpElec.SG052  = {'L4', 'R2'};
LfpElec.SG056  = {'L4', 'R1'};


% Define Time Window
tWidth   = 0.9;
tOffset  = 0.3;
NewSR = 300;
freq_bin = 0.2;

nSub = numel(subjects.goodHeartMOff);


plots = true;
baseline =true;

disp("************* STARTING EPOCH AND TIMELOCKING *************");

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

        if seperateSTN
            channels{8} = LfpElec.(subject){1};
            channels{9} = LfpElec.(subject){2};
        end
        
        if baseline
            fprintf('Loading TFR Data\n');
            pattern = fullfile(data_dir, 'tfr', [subject, '_TFR-EPOCH_', subfname, '*', '_BSL=', '*']);
            files = dir(pattern);
            filename = fullfile(files(1).folder, files(1).name);
            load(filename, 'TFR', '-mat');
        else
            fprintf('Loading TFR Data\n');
            pattern = fullfile(data_dir, 'tfr', [subject, '_TFR-EPOCH_', subfname, '*']);
            files = dir(pattern);
            filename = fullfile(files(1).folder, files(1).name);
            load(filename, 'TFR', '-mat');
        end

        freqs = TFR.freqs;
        times = TFR.times;


         for c = 1:numel(channels)
            channel = channels{c};

            PowerAllTrsAvg(sub,c,:,:) = squeeze(mean(TFR.(channel).pow,1));
         end
  
    end

    for c = 1:numel(channels)
        channels = {'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'Pz', 'STNl', 'STNr'};
        channel = channels{c};

        ChanPowerSubTrsAvg = squeeze(mean(mean(squeeze(PowerAllTrsAvg(:,c,:,:)),1),3));
        ChanPowerSubTrsAvg = ChanPowerSubTrsAvg.^2;
        psd = ChanPowerSubTrsAvg / freq_bin;


        f1 = figure;
        set(f1,'Position',[1949 123 1023 400]);
        plot(freqs, 10*log10(ChanPowerSubTrsAvg),'LineWidth', 1, 'Color','k')
        title(sprintf('Average PSD for in %s,med: %s', channel, subfname))
        xlabel('Frequencies (Hz)') % Add x-label
        ylabel('dB/Hz') %
        axis('tight');

        gr1 = fullfile('F:\HeadHeart\2_results\psd' , ['AvgPSD_', channel, '_', subfname,  '.png']);
        exportgraphics(f1,gr1, 'Resolution', 300)

    end
end