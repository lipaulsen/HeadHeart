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

% ========================== SUBJECT FLAGS ================================

% MEDICATION
% only one can be true at all times
MedOn = true;
MedOff = false;

% SUBJECT STATUS
% only one can be true at all times
newsubs = false;
oldsubs = true;
allsubs = false;

% make channel info into cells
AllSubsChansRaw = cellfun(@(x) strsplit(x, ', '), {subject_info.channels_raw}, 'UniformOutput', false);
AllSubsChansStn = cellfun(@(x) strsplit(x, ', '), {subject_info.channels}, 'UniformOutput', false);

% filter which subjects and which channels you want
if MedOn == true & newsubs == true % Only New Subs that are MedOn
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

%subfnames = fieldnames(subjects);

%=========================================================================

% Define if plots are to be shown
show_plots = true;

%If we use left and right STN as seperate subjects put this as true
%(increases subjectsize by 2)
seperateSTN = false;

% Define feature extraction steps to perform
steps = {'PWelch'}; %'TFR Basis', 'PWelch'

% Define folder variables
epoch_name = 'epoch';  % feature extraction folder (inside derivatives)


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


% Define Time Window
tWidth   = 0.9;
tOffset  = 0.3;
NewSR = 300;
freq_bin = 0.2;

nSub = numel(subjects);


plots = true;
baseline =true;
epoch = true;
baseline_win = [-0.3 -0.1];
% Define Time Window
tWidth   = 0.9;
tOffset  = 0.3;

NewSR=300;
stfr=0.5;   enfr=30; dfr=0.2;
Frqs=stfr:dfr:enfr;
if NewSR < 2*enfr; NewSR=(enfr+2)*2; end

window_length = 5000;           % Window length
window = hamming(window_length); % Hamming window
noverlap = window_length/2;    % 50% overlap
nfft = 10000;          % FFT points

Fhp = 2;
Hz_dir = '2Hz';


disp("************* STARTING PSD CALCULATION *************");

for med = 1 %:2 % MedOn

    if MedOn == true
        medname = 'MedOn';
    elseif MedOff == true
        medname = 'MedOff';
    end
   
    % fprintf('Loading AVG ECG Data\n'); %
    % pattern = fullfile(data_dir, epoch_name, 'avg', ['ECG-AVG_', medname, 'n=', num2str(nSub), '*']);
    % files = dir(pattern);
    % filename = fullfile(files(1).folder, files(1).name);
    % load(filename, 'AVGECG');

    for sub = 4:numel(subjects) % subjects.goodHeartMOff BE AWARE THAT THIS EXCLUDES PATIENTS WITH ARRITHYMIAS
        % Extract the subject
        subject = subjects{sub}; %goodHeartMOff{sub}

        if seperateSTN
            channels{8} = LfpElec.(subject){1};
            channels{9} = LfpElec.(subject){2};
        end
        %% Do PSD Based on Hilbert TFR

        if ismember('TFR Basis', steps)
            if baseline
                fprintf('Loading TFR Data\n');
                pattern = fullfile(data_dir, 'tfr', [subject, '_TFR-EPOCH_', medname, '*', '_BSL=', '*']);
                files = dir(pattern);
                filename = fullfile(files(1).folder, files(1).name);
                load(filename, 'TFR', '-mat');
            else
                fprintf('Loading TFR Data\n');
                pattern = fullfile(data_dir, 'tfr', [subject, '_TFR-EPOCH_', medname, '*']);
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

        %% DO PSD based on PWelch

        if ismember('PWelch', steps)
            fprintf('Loading Data of  subject %s number %i of %i\n', subject, sub, numel(subjects));
            pattern = fullfile(data_dir, 'preproc', 'all', [subject, '_preprocessed_', medname, '*']);
            files = dir(pattern);
            filename = fullfile(files(1).folder, files(1).name);
            load(filename, 'SmrData');
            % Load subject data
            % subject_data = fullfile(data_dir, preprocessed_name, medname, ['sub-', subject], [subject, '_preprocessed_', medname, '_Rest.mat']);
            % load(subject_data, 'SmrData');

            SR = SmrData.SR;
            EventTms = SmrData.EvData.EvECGP_Cl;

            % Filter the channes for the sub
            OneSubChans = string(FltSubsChansRaw{sub});

            f1 = figure;
            set(f1,'Position',[100 278 1200 800]);

            % PWelch on entire Channel Data
            for c = 1:numel(FltSubsChansRaw{sub})
                channel = OneSubChans{c};
                % Filter Data 
                ChDta = SmrData.WvDataCleaned(c, :);
                % Calculate PWelch
                [FftPwelch, f] = pwelch(ChDta, window, noverlap, nfft, SR);
                % Save in one big Matrix
                PWelchAllDtaAvg(sub,c,:) = FftPwelch;

                if show_plots

                    % Trim the PSD to desired range
                    freq_limit = 30;
                    freq_limitlow = 0.5;% Hz
                    idx = f <= freq_limit & f > freq_limitlow ;  % Index for frequencies up to 30 Hz
                    f_trimmed = f(idx);
                    FftPwelch_trimmed = squeeze(PWelchAllDtaAvg(sub,c,idx));
                    FftPwelch_trimmed = smoothdata(FftPwelch_trimmed, 'gaussian', 5); % Apply a Gaussian Filter to Smoothe the lines

                    % Plot for each channel in each subject
                    subplot(ceil(numel(FltSubsChansRaw{sub})/5),5,c);
                    plot(f_trimmed, 10*log10(FftPwelch_trimmed),'LineWidth', 1, 'Color','k')
                    title(sprintf('PSD for %s in %s,med: %s', subject, channel, medname))
                    xlabel('Frequencies (Hz)') % Add x-label
                    ylabel('dB/Hz') %
                    axis('tight');          
                end

            end

            % Save Plot
            gr1 = fullfile(results_dir, '/psd/ss' , ['PSD_PWelch_', subject ,'_', channel, '_', medname,  '.png']);
            exportgraphics(f1,gr1, 'Resolution', 300)

            % PWelch on Epochs
            % freq_limit = 30;  % Hz
            %             idx = f <= freq_limit;  % Index for frequencies up to 30 Hz
            %             f_trimmed = f(idx);
            %             FftPwelch_trimmed = FftPwelch(idx);
            %
            %             figure
            %             plot(f_trimmed, 10*log10(FftPwelch_trimmed));
            %             xlabel('Frequency (Hz)');
            %             ylabel('Power/Frequency (dB/Hz)');
            %             title('Power Spectral Density (Welch)');

        end
    end

    disp("************* STARTING PSD PLOT *************");
    %% PLOT PSD on HIlbert
    if ismember('TFR Basis', steps)
        for c = 1:numel(channels)
            channels = {'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'Pz', 'STNl', 'STNr'};
            channel = channels{c};

            ChanPowerSubTrsAvg = squeeze(mean(mean(squeeze(PowerAllTrsAvg(:,c,:,:)),1),3));
            ChanPowerSubTrsAvg = ChanPowerSubTrsAvg.^2;
            psd = ChanPowerSubTrsAvg / freq_bin;


            f1 = figure;
            set(f1,'Position',[1949 123 1023 400]);
            plot(freqs, 10*log10(ChanPowerSubTrsAvg),'LineWidth', 1, 'Color','k')
            title(sprintf('Average PSD for in %s,med: %s', channel, medname))
            xlabel('Frequencies (Hz)') % Add x-label
            ylabel('dB/Hz') %
            axis('tight');

            gr1 = fullfile(results_dir, '/psd' , ['AvgPSD_', channel, '_', medname,  '.png']);
            exportgraphics(f1,gr1, 'Resolution', 300)

        end
    end
    %% PLOT PSD on PWELCH
    if ismember('PWelch', steps)
        for c = 1:numel(FltSubsChansRaw{sub})
            channel = OneSubChans{c};

            % Average over channel
            ChanSubAllStaAvg = squeeze(mean(squeeze(PWelchAllDtaAvg(:,c,:)),1));
            
            % Trim the PSD to desired range
            freq_limit = 30;
            freq_limitlow = 2;% Hz
            idx = f <= freq_limit & f > freq_limitlow ;  % Index for frequencies up to 30 Hz
            f_trimmed = f(idx);
            FftPwelch_trimmed = ChanSubAllStaAvg(idx);
            FftPwelch_trimmed = smoothdata(FftPwelch_trimmed, 'gaussian', 10); % Apply a Gaussian Filter to Smoothe the lines
            
            % Plot Average over all subjects for each channel
            f1 = figure;
            set(f1,'Position',[1949 123 1023 400]);
            plot(f_trimmed, 10*log10(FftPwelch_trimmed),'LineWidth', 1, 'Color','k')
            title(sprintf('Average PSD for in %s,med: %s', channel, medname))
            xlabel('Frequencies (Hz)') % Add x-label
            ylabel('dB/Hz') %
            axis('tight');

            % Save Plot
            gr1 = fullfile(results_dir, '/psd' , ['AvgPSD_PWelch', channel, '_', medname,  '.png']);
            exportgraphics(f1,gr1, 'Resolution', 300)


        end
    end
end
fprintf('PSD Calculation and Plotting DONE!')