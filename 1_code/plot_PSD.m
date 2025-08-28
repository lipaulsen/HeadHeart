% function [] = plot_PSD(subjects, data_dir, results_dir)

%% Epoching and Time Locking Data for HeadHeart

% Author: Lisa Paulsen
% Contact: lisaspaulsen[at]web.de
% Created on: 1 October 2024
% Last update: 15 October 2024

%% INPUT OUTPUT
% calculate and plot the PSD using PWelch
%
% Inputs:
% Preprocessed data (EEG, LFP, ECG) from .mat file
%
% Outputs:
% Single Subject and Grand Average Plots for each Channel

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
MedOn = false;
MedOff = true;

% SUBJECT STATUS
% only one can be true at all times
newsubs = false;
oldsubs = false;
allsubs = true;

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

%=========================================================================

% Define if plots are to be shown
show_plots = true;

% Define feature extraction steps to perform
steps = {'PWelch'}; %'TFR Basis', 'PWelch'

% Define Time Window
tWidth   = 0.9;
tOffset  = 0.3;
freq_bin = 0.2;
nSub = numel(subjects);
plots = true;
baseline =true;


window_length = 5000;           % Window length
window = hamming(window_length); % Hamming window
noverlap = window_length/2;    % 50% overlap
nfft = 10000;          % FFT points


disp("************* STARTING PSD CALCULATION *************");

% for med = 1%:2 %MedOn

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

for sub = 1:numel(subjects) % subjects.goodHeartMOff BE AWARE THAT THIS EXCLUDES PATIENTS WITH ARRITHYMIAS
    % Extract the subject
    subject = subjects{sub}; %goodHeartMOff{sub}

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
        gr1 = fullfile(results_dir, '/psd/ss' , ['PSD_PWelch_', subject ,'_', medname, '.png']);
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
S
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
fprintf('PSD Calculation and Plotting DONE!')