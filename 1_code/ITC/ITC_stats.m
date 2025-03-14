% function [] = sub_ITC_stats(subjects, data_dir, results_dir)

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

% To Dos for ITC with new subs and BPReref
% new subs EEG: 22 chans 
% new subs STN: 8 chans x 3  (BPReref Hi, Low and raw)
% old subs STN: 20 chans x 2 (BPReref Hi, Low)


% MEDICATION
% only one can be true at all times
MedOn = true;
MedOff = false;

% SUBJECT STATUS
% only one can be true at all times
newsubs = true;
oldsubs = false;
allsubs = false;

% get the channel info into the shape of cells
AllSubsChansRaw = cellfun(@(x) strsplit(x, ', '), {subject_info.channels_raw}, 'UniformOutput', false);
AllSubsChansStn = cellfun(@(x) strsplit(x, ', '), {subject_info.channels}, 'UniformOutput', false);
AllSubsOnlyStn = cellfun(@(x) strsplit(x, ', '), {subject_info.STN}, 'UniformOutput', false);
AllSubsOnlyEEG = cellfun(@(x) strsplit(x, ', '), {subject_info.EEG}, 'UniformOutput', false);

% filter which subjects and which channels you want
if MedOn == true & newsubs == true % Only New Subs that are MedOn
    subjects = string({subject_info([subject_info.new] == 1 & [subject_info.MedOn] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.new] == 1 & [subject_info.MedOn] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.new] == 1 & [subject_info.MedOn] == 1);
    FltSubsOnlyStn = AllSubsOnlyStn([subject_info.new] == 1 & [subject_info.MedOn] == 1);
    FltSubsOnlyEEG = AllSubsOnlyEEG([subject_info.new] == 1 & [subject_info.MedOn] == 1);
elseif MedOff == true & newsubs == true  % Only New Subs that are MedOff
    subjects = string({subject_info([subject_info.new] == 1 & [subject_info.MedOff] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.new] == 1 & [subject_info.MedOff] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.new] == 1 & [subject_info.MedOff] == 1);
    FltSubsOnlyStn = AllSubsOnlyStn([subject_info.new] == 1 & [subject_info.MedOff] == 1);
    FltSubsOnlyEEG = AllSubsOnlyEEG([subject_info.new] == 1 & [subject_info.MedOff] == 1);
elseif MedOn == true & oldsubs == true  % Only Old Subs that are MedOn
    subjects = string({subject_info([subject_info.new] == 0 & [subject_info.MedOn] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.new] == 0 & [subject_info.MedOn] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.new] == 0 & [subject_info.MedOn] == 1);
    FltSubsOnlyStn = AllSubsOnlyStn([subject_info.new] == 0 & [subject_info.MedOn] == 1);
    FltSubsOnlyEEG = AllSubsOnlyEEG([subject_info.new] == 0 & [subject_info.MedOn] == 1);
elseif MedOff == true & oldsubs == true  % Only Old Subs that are MedOff
    subjects = string({subject_info([subject_info.new] == 0 & [subject_info.MedOff] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.new] == 0 & [subject_info.MedOff] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.new] == 0 & [subject_info.MedOff] == 1);
    FltSubsOnlyStn = AllSubsOnlyStn([subject_info.new] == 0 & [subject_info.MedOff] == 1);
    FltSubsOnlyEEG = AllSubsOnlyEEG([subject_info.new] == 0 & [subject_info.MedOff] == 1);
elseif MedOn == true & allsubs == true  % All Subs that are MedOn
    subjects = string({subject_info([subject_info.MedOn] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.MedOn] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.MedOn] == 1);
    FltSubsOnlyStn = AllSubsOnlyStn([subject_info.MedOn] == 1);
    FltSubsOnlyEEG = AllSubsOnlyEEG([subject_info.MedOn] == 1);
elseif MedOff == true & allsubs == true % All Subs that are MedOff
    subjects = string({subject_info([subject_info.MedOff] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.MedOff] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.MedOff] == 1);
    FltSubsOnlyStn = AllSubsOnlyStn([subject_info.MedOff] == 1);
    FltSubsOnlyEEG = AllSubsOnlyEEG([subject_info.MedOff] == 1);
end

%=========================================================================

% Define if plots are to be shown
show_plots = false;

%flag if baseline is on or off
baseline = true;

% Define feature extraction steps to perform
steps = {'Calc Single Subject ITC'}; %'Plot SubAvg PermStats', 'Calc Single Subject ITC', 'Plot SubAvg ITC', 'Plot Power', 'Plot Plow Single Channels', 'Plot Power Cluster', 

% Define Time Window
tWidth   = 0.9;
tOffset  = 0.3;
FltPassDir = 'twopass';
NewSR = 300;

BandWidth = 2; % BandWidth in Hz;
Qfac      = 2; % Attenuation in db(-Qfac)
tCircMean = 0.02; % for By TRials calc

permstats = true;
numPerms = 500;
surrogate = true;
trials = false;
plots = false;
ITC = [];
signif_thresh = 0.05;
Hz_dir = '2Hz';


if MedOn == true
    medname = 'MedOn';
elseif MedOff == true
    medname = 'MedOff';
end

% Use BPReref Data
BPReref = true; BPRerefTit = 'BPReref';
BPRerefHi = false; BPRerefHiTit = 'BPRerefHi';
BPRerefLw = true; BPRerefLwTit = 'BPRerefLow';
BPRerefBest = false; BPRerefBestTit = 'BPRerefBest';
% Flag if only EEG, STN or all channels
allchans = false;
onlyeeg = true;
onlystn = false;

disp("************* STARTING ITC *************");

fprintf('Loading AVG ECG Data\n');
if strcmp(medname, 'MedOn')
    pattern = fullfile(data_dir, 'ecg', ['ECG-AVG_', medname, '_n=11_', '*']);
elseif strcmp(medname, 'MedOff')
    pattern = fullfile(data_dir, 'ecg', ['ECG-AVG_', medname, '_n=7_', '*']);
end
files = dir(pattern);
filename = fullfile(files(1).folder, files(1).name);
load(filename, 'AVGECG');

if ismember ('Calc Single Subject ITC', steps)

% Initialize the Perm Matrix be sure here that the freqs and Time are
% fitting because it does not work with the getting it out of the data with
%out overwriting it
max_chan = max(cellfun(@numel, FltSubsChansStn));
PermItcAll = zeros(numel(subjects), max_chan, numPerms, 141, 271);
% ZScoresAll = zeros(numel(subjects), numel(channels), 141, 271);
% PValAll = zeros(numel(subjects), numel(channels), 141, 271);

    for sub = 1:numel(subjects)
        % Extract the subject
        subject = subjects{sub};

        % % Load the the cleaned ECG R Peaks Data
        fprintf('Loading ECG Data\n');
        pattern = fullfile(data_dir, 'ecg', 'ss' ,[subject, '_EpochECGEvData_', medname, '*']);
        files = dir(pattern);
        filename = fullfile(files(1).folder, files(1).name);
        load(filename, 'EvECG');

        if baseline & BPReref & BPRerefHi
            fprintf('Loading TFR Data for %s\n', subject);
            pattern = fullfile(data_dir, 'tfr', Hz_dir, [subject,  '_TFR-EPOCH_', medname ,'_Rest_', BPRerefHiTit, '*']);
            files = dir(pattern);
            filename = fullfile(files(1).folder, files(1).name);
            load(filename, 'TFR', '-mat');
        elseif baseline & BPReref & BPRerefLw
            fprintf('Loading TFR Data for %s\n', subject);
            pattern = fullfile(data_dir, 'tfr', Hz_dir, [subject,  '_TFR-EPOCH_', medname ,'_Rest_', BPRerefLwTit, '*']);
            files = dir(pattern);
            filename = fullfile(files(1).folder, files(1).name);
            load(filename, 'TFR', '-mat');
        elseif baseline & BPReref & BPRerefBest
            fprintf('Loading TFR Data for %s\n', subject);
            pattern = fullfile(data_dir, 'tfr', Hz_dir, [subject,  '_TFR-EPOCH_', medname ,'_Rest_', BPRerefBestTit, '*']);
            files = dir(pattern);
            filename = fullfile(files(1).folder, files(1).name);
            load(filename, 'TFR', '-mat');
        elseif baseline
            fprintf('Loading TFR Data for %s\n', subject);
            pattern = fullfile(data_dir, 'tfr', Hz_dir, [subject, '_TFR-EPOCH_', medname, '*', '_BSL=', '*']);
            files = dir(pattern);
            filename = fullfile(files(1).folder, files(1).name);
            load(filename, 'TFR', '-mat');
        else
            fprintf('Loading TFR Data for %s\n', subject);
            pattern = fullfile(data_dir, 'tfr', Hz_dir, [subject, '_TFR-EPOCH_', medname, '*']);
            files = dir(pattern);
            filename = fullfile(files(1).folder, files(1).name);
            load(filename, 'TFR', '-mat');
        end

        fprintf('Loading Smr Data\n');
        pattern = fullfile(data_dir, 'preproc', 'all', [subject, '_preprocessed_', medname, '*']);
        files = dir(pattern);
        filename = fullfile(files(1).folder, files(1).name);
        load(filename, 'SmrData');

        SR = TFR.SR;
        freqs = TFR.freqs;
        EventTms = SmrData.EvData.EvECGP_Cl;
        if baseline; times = TFR.times; else times = AVGECG.times; end

        if allchans
            channels = FltSubsChansStn{sub};
        elseif onlyeeg
            channels = FltSubsOnlyEEG{sub};
        elseif onlystn
            channels = FltSubsOnlyStn{sub};
        end

        subject_channels = FltSubsChansRaw{sub}; % this is only needed to get the correct row in the ChDta 

        % KS29 has no EEG recordings in MedOn so we delete those values
        if strcmp(medname,'MedOn') & strcmp(subject,'KS29')
            channels = {FltSubsChansStn{sub}{end-1:end}};
        end

        for c = 1:numel(channels)
            channel = channels{c};
            fprintf('************ Calculating ITC for %s in %s **************** \n', subject, channel);
            % Calc original ITC
            [FrsTmItc]=Get_PSI_ByTrials_ITC(TFR.(channel).phase,SR,tCircMean);

            % Scale the ITc to the relative ITC of the channel
            meanFrsTmItc = mean(mean(FrsTmItc,1),2);
            relFrsTmItc = FrsTmItc/meanFrsTmItc;

            ItcAll(sub,c,:,:) = FrsTmItc; % SubjectxChannelxFreqxTime
            RelItcAll(sub,c,:,:) = relFrsTmItc;
        end

        outputPDF1 = fullfile(results_dir, 'itc' , Hz_dir, 'ss', [subject, '_ITC_Allchan_med=', medname, ...
            '_win=-', num2str(tOffset),'to', num2str(tWidth-tOffset),'_BSL=', num2str(baseline), '.pdf']);

        for c = 1:numel(channels)
            channel = channels{c};

            if plots
                f1=figure;
                set(f1,'Position',[1949 123 1023 785]);
                subplot(2,1,1)
                plot(times, mean(EvECG.EvData,1), 'Color', 'k'); hold on
                set(gca,'Position',[0.1300 0.5838 0.71 0.3])
                xline(0, "--k", 'LineWidth', 2);
                ylabel('Amplitude')
                axis('tight')
                title(sprintf('Average ECG for %s in %s, medication: %s', subject, channel, medname))
                hold off
                subplot(2,1,2)
                imagesc(times,freqs,squeeze(ItcAll(sub,c,:,:)));axis xy;
                colormap('parula');
                xline(0, "--k", 'LineWidth', 2);
                col = colorbar;
                col.Label.String = 'ITC Values'; % Add title to colorbar
                xlabel('Time (s)') % Add x-label
                ylabel('Frequencies (Hz)') % Add y-label
                title(sprintf('ITC for %s in %s, medication: %s', subject, channel, medname))


                %gr1 = fullfile('F:\HeadHeart\2_results\itc\ss' , [subject, '_', medname, '_ITC_', channel, '.png']);
                %exportgraphics(f1,gr1, 'Resolution', 300)
                exportgraphics(f1, outputPDF1, 'Append', true);
            end
        end
        %% ==================== PERMUTATION ===========================
        if permstats
            outputPDF = fullfile('F:\HeadHeart\2_results\itc\ss_perm' , [subject, '_ITC-PermStats_Allchan_med=', medname, '_perm=', num2str(numPerms), ...
                '_win=-', num2str(tOffset),'to', num2str(tWidth-tOffset),'_BSL=', num2str(baseline), '_pval=', num2str(signif_thresh),'.pdf']);
            for c = 1:numel(channels)
                channel = channels{c};

                %[ITCzscores]=ITC_permutation_test(FrsTmItc, relFrsTmItcAll, IBI.(medname){1}, numPerms, freqs, time_bins, SR, TFR.(channel).phase);
                fprintf('************ Calculating Perm Stats for %s in %s **************** \n', subject, channel);

                % Intitialize variables
                %permuted_ITCs = zeros([numPerms, size(FrsTmItc, 1), size(FrsTmItc, 2)]);
                [nTrials, nFreq, nTms] = size(TFR.(channel).phase);
               
                chan_idx = find(strcmp(subject_channels, channel)); % Find index

                % Get the raw channel data
                ChDta = SmrData.WvDataCleaned(chan_idx, :);
                % Override SR here with the raw channel SR
                oldSR = SmrData.SR;


                if surrogate

                    % Pre-generate surrogate R-peaks outside of the parfor loops
                    % time_shifts_all = rand(numPerms, length(ibi_series)) - 0.3;
                    surrogate_rpeaks = zeros(numPerms, length(EventTms));
                    for p = 1:numPerms
                        surrogate_rpeaks(p, :) = EventTms + (rand(1, length(EventTms)) - 0.5);
                    end

                    % Generate Permutations
                    PermItcData = zeros(numPerms,nFreq,nTms);

                    startTime = datetime('now');
                    disp(['Start Time: ', datestr(startTime)]);

                    for perm = 1:numPerms % here chage to parfor
                        % Time Lock the surrogate R Peaks to the Channel
                        % Data and apply the filters as well as the DS and
                        % create TFR for the new epochs
                        currSurrogateRpeaks = surrogate_rpeaks(perm, :);
                        [ChsAllFrsTmPha] = time_lock_to_surrogate(ChDta, currSurrogateRpeaks, oldSR, tWidth, tOffset, numPerms, NewSR, freqs, times);

                        %  Calculate ITC with the surrogate R-peaks (one per channel)
                        [PermItcData(perm, :, :)] = Get_PSI_ByTrials_ITC(ChsAllFrsTmPha,NewSR,tCircMean);
                        fprintf('perm = %d \n', perm)
                    end

                    endTime = datetime('now'); disp(['End Time: ', datestr(endTime)]);
                    % Calculate elapsed time
                    elapsedTime = endTime - startTime; disp(['Elapsed Time: ', char(elapsedTime)]);
                end
                % Step 1: Compute z-scores
                diff_sum_perm_mean = squeeze(mean(PermItcData,1)); % Mean of the permutation distribution
                diff_sum_perm_std = squeeze(std(PermItcData,1)); % Standard deviation of permutation distribution

                % diffPerm_mean(1,:,:) = diff_sum_perm_mean;
                % diffPerm_std(1,:,:)  = diff_sum_perm_std;

                zscores = (squeeze(ItcAll(sub,c,:,:)) - diff_sum_perm_mean) ./ diff_sum_perm_std ;
                %zscores_perm = (PermItcData - diffPerm_mean) ./ diffPerm_std;
                p_orig = 2 * (1 - normcdf(zscores, 0, 1));
                p_thresh = p_orig < signif_thresh;

                % ZScoresAll(sub,c,:,:) = zscores; % SubjectxChannelxFreqxTime (Last two are ITC ZScores)
                % PValAll(sub,c,:,:) = p_orig;  % SubjectxChannelxFreqxTime (Last two are ITC PVals)
                PermItcAll(sub,c,:,:,:) = PermItcData; % SubjectxChannelxPermutationxFreqxTime


                % f3 = figure; % Sanity check that the distributions are normalized and overlapping so that my null hypothesis actually reflects my data
                % histogram(PermItcData);
                % hold on
                % histogram(squeeze(ItcAll(sub,c,:,:)));
                % %title(sprintf('Perm ITC and Original ITC Dist for %s in %s, perms: %d, med: %s', subject, channel, numPerms, medname))
                %
                % f4 = figure;
                % subplot(2,1,1);
                % histogram(zscores);
                % %title(sprintf('Z Score Dist for %s in %s, perms: %d, med: %s', subject, channel, numPerms, medname))
                % subplot(2,1,2);
                % histogram(p_orig);
                % %title(sprintf('P Val Dist for %s in %s, perms: %d, med: %s', subject, channel, numPerms, medname))

                %itc_zscores_thresh = (zscores > 2) | (zscores < -2);

                f5 = figure;
                set(f5,'Position',[1949 123 1023 785]);

                % Upper subplot
                subplot(2,1,1)
                plot(times, mean(EvEcgData,1), 'Color', 'k'); hold on
                set(gca,'Position',[0.1300 0.5838 0.71 0.3])
                xline(0, "--k", 'LineWidth', 2);
                axis('tight')
                title(sprintf('Average ECG for %s, med: %s', subject, medname))
                ylabel('Amplitude')
                hold off

                % Lower subplot
                subplot(2,1,2)
                imagesc(times, freqs, squeeze(ItcAll(sub,c,:,:))); axis xy;
                colormap('parula');
                col = colorbar;
                col.Label.String = 'ITC Values'; % Add title to colorbar
                clims = clim;
                hold on;
                contour(times, freqs, p_thresh, 1, 'linecolor', 'k', 'linewidth', 1.5);
                xline(0, "--k", 'LineWidth', 2);
                clim(clims);
                title(sprintf('ITC for %s in %s, perms: %d, med: %s, p<%.4g', subject, channel, numPerms, medname, signif_thresh))
                xlabel('Time (s)') % Add x-label
                ylabel('Frequencies (Hz)') % Add y-label
                hold off

                % outputPDF = fullfile('F:\HeadHeart\2_results\itc' , [subject, '_ITC-PermStats_chan=', channel, '_med=', medname, '_perm=', num2str(numPerms), ...
                %     '_win=-', num2str(tOffset),'to', num2str(tWidth-tOffset),'_BSL=', num2str(baseline), '_pval=', num2str(signif_thresh),'.pdf']);
                % exportgraphics(f5,gr5, 'Resolution', 300)
                exportgraphics(f5, outputPDF, 'Append', true);

                % % Define the output PDF file name
                % outputPDF = fullfile('F:\HeadHeart\2_results\itc\ss_perm' , [subject, '_ITC-PermStats_chan=', channel, '_med=', medname, '_perm=', num2str(numPerms), '.pdf']);
                %
                % % Ensure the PDF file doesn't already exist
                % if isfile(outputPDF)
                %     delete(outputPDF);
                % end
                %
                % % List of figure handles
                % figureHandles = [f3, f4, f5]; % Replace with your actual figure handles
                %
                % % Loop through the figure handles and append each to the PDF
                % for i = 1:length(figureHandles)
                %     exportgraphics(figureHandles(i), outputPDF, 'Append', true);
                % end
                %
                disp(['All figures saved to ', outputPDF]);

            end
        end



    end
    if permstats
        ITC.PERM.ZScoresAll = ZScoresAll;  % SubjectxChannelxFreqxTime (Last two are ITC ZScores)
        ITC.PERM.PValAll = PValAll; % SubjectxChannelxFreqxTime (Last two are ITC PVals)
        ITC.PERM.PermItcAll = PermItcAll; %SubjectxChannelxPermutationxFreqxTime
    end
    % Save ITC MAtrix of all Subjects and all Channels
    ITC.SR = SR;
    ITC.times = times;
    ITC.freqs = freqs;
    ITC.ItcAll = ItcAll;
    ITC.RelItcAll = RelItcAll;

    % Create Paths
    if permstats & newsubs & BPRerefHi & onlystn
        save_path = fullfile(data_dir, 'itc', ['ITC-NewSubs_', medname , '_OnlySTN_BP=', BPRerefHiTit, '_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    elseif permstats & newsubs & BPRerefLw & onlystn
        save_path = fullfile(data_dir, 'itc', ['ITC-NewSubs_', medname , '_OnlySTN_BP=', BPRerefLwTit, '_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    elseif permstats & newsubs & onlyeeg
        save_path = fullfile(data_dir, 'itc', ['ITC-NewSubs_', medname ,'_OnlyEEG_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    elseif permstats & newsubs & allchans
        save_path = fullfile(data_dir, 'itc', ['ITC-NewSubs_', medname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    elseif permstats & oldsubs & BPRerefHi & onlystn
        save_path = fullfile(data_dir, 'itc', ['ITC-OldSubs_', medname , '_OnlySTN_BP=', BPRerefHiTit, '_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    elseif permstats & oldsubs & BPRerefLw & onlystn
        save_path = fullfile(data_dir, 'itc', ['ITC-OldSubs_', medname , '_OnlySTN_BP=', BPRerefLwTit,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    elseif permstats & oldsubs & onlyeeg
        save_path = fullfile(data_dir, 'itc', ['ITC-OldSubs_', medname ,'_OnlyEEG_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    elseif permstats & oldsubs & allchans
        save_path = fullfile(data_dir, 'itc', ['ITC-OldSubs_', medname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    elseif permstats
        save_path = fullfile(data_dir, 'itc', ['ITC-AllSubs_', medname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    else
        save_path = fullfile(data_dir, 'itc', ['ITC-AllSubs_', medname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), 'HPF=', num2str(freqs(1)), '_HP=',  num2str(freqs(1)) ,'.mat']);
    end
    save(save_path, 'ITC', '-v7.3');
    fprintf('Saved ITC Data for all subs and channels to: %s\n', save_path);
end


% PLOT THE AVERAGE ITC PER SUBJECT
if ismember('Plot SubAvg ITC', steps)
    fprintf('Plot SubAvg ITC\n');
    if ismember('Calc Single Subject ITC', steps) == false

        fprintf('Loading ITC Data\n');
        pattern = fullfile(data_dir, 'itc', ['ITC-AllSubs_',  medname, '*', '_HP=', Hz_dir(1:end-2), '*']);
        files = dir(pattern);
        filename = fullfile(files(1).folder, files(1).name);
        load(filename, 'ITC', '-mat');
        SR= ITC.SR;
        freqs = ITC.freqs;

    end
    plots = true;

    if allchans
        channels = FltSubsChansStn{sub};
    elseif onlyeeg
        channels = FltSubsOnlyEEG{sub};
    elseif onlystn
        channels = FltSubsOnlySTN{sub};
    end

    for c = 1:numel(channels)
        channel = channels{c};

        ItcAll_subavg = squeeze(mean(squeeze(ITC.ItcAll(:,c,:,:)),1));
        RelItcAll_subavg = squeeze(mean(squeeze(ITC.RelItcAll(:,c,:,:)),1));

        if plots
            f2=figure;
            set(f2,'Position',[1949 123 1023 785]);
            subplot(2,1,1)
            plot(times, AVGECG.mean', 'Color', 'k'); hold on
            set(gca,'Position',[0.0900  0.6838 0.78 0.2])
            xline(0, "--k", 'LineWidth', 2);
            ylabel('Amplitude')
            axis('tight')
            title(sprintf('Average ECG over all subjects, medication: %s', medname))
            hold off
            subplot(2,1,2)
            set(gca,'Position',[0.0900 0.1200 0.8498 0.4612])
            imagesc(times,freqs,ItcAll_subavg);axis xy;
            colormap('parula');
            xline(0, "--k", 'LineWidth', 2);
            col = colorbar;
            col.Label.String = 'ITC Values'; % Add title to colorbar
            xlabel('Time (s)') % Add x-label
            ylabel('Frequencies (Hz)') % Add y-label
            title(sprintf('Average ITC for %s, med: %s, HP: %s', channel, medname, Hz_dir))

            gr2 = fullfile('F:\HeadHeart\2_results\itc' , Hz_dir, 'group', ['ITC_', channel, '_', medname,  '.png']);
            exportgraphics(f2,gr2, 'Resolution', 300)
        end
    end
end

% Now here is the Grand Average ITC over all Subs per Channel
if ismember('Plot SubAvg PermStats', steps)
    fprintf('Plot SubAvg ITC PermStats\n');

    if ismember('Calc Single Subject ITC', steps) == false
        fprintf('Loading ITC Data\n');
        pattern = fullfile(data_dir, 'itc', ['ITC-AllSubs_',  medname, '*', '_perm=', '*']);
        files = dir(pattern);
        filename = fullfile(files(1).folder, files(1).name);
        load(filename, 'ITC', '-mat');
        SR= ITC.SR;
        freqs = ITC.freqs;
        times = ITC.times;
    end

    if allchans
        channels = FltSubsChansStn{sub};
    elseif onlyeeg
        channels = FltSubsOnlyEEG{sub};
    elseif onlystn
        channels = FltSubsOnlySTN{sub};
    end

    for c = 1: numel(channels)
        channel = channels{c};

        %ChanMeanZscores_avg = squeeze(mean(squeeze(ITC.PERM.ZScoresAll(:,c,:,:)),1));  % SubjectxChannelxFreqxTime (Last two are ITC ZScores)
        %ChanMeanPVal_avg = squeeze(mean(squeeze(ITC.PERM.PValAll(:,c,:,:)),1));  % SubjectxChannelxFreqxTime (Last two are ITC PVals)
        PermItcAll_avg = squeeze(mean(squeeze(ITC.PERM.PermItcAll(:,c,:,:,:)),1)); %SubjectxChannelxPermutationxFreqxTime, Mean over all Subjects in one channel
        ItcAll_subavg = squeeze(mean(squeeze(ITC.ItcAll(:,c,:,:)),1));
        RelItcAll_subavg = squeeze(mean(squeeze(ITC.RelItcAll(:,c,:,:)),1));

        % Step 1: Compute z-scores
        diff_sum_perm_mean_all = squeeze(mean(PermItcAll_avg,1)); % Mean of the permutation distribution
        diff_sum_perm_std_all = squeeze(std(PermItcAll_avg,1)); % Standard deviation of permutation distribution

        % diffPerm_mean(1,:,:) = diff_sum_perm_mean;
        % diffPerm_std(1,:,:)  = diff_sum_perm_std;

        zscores_all = (ItcAll_subavg - diff_sum_perm_mean_all) ./ diff_sum_perm_std_all ;
        %zscores_perm = (PermItcData - diffPerm_mean) ./ diffPerm_std;
        p_orig_all = 2 * (1 - normcdf(zscores_all, 0, 1));
        signif_thresh =0.005;
        p_thresh_all = p_orig_all < signif_thresh;



        % figure; % Sanity check that the distributions are normalized and overlapping so that my null hypothesis actually reflects my data
        % histogram(PermItcAll_avg);
        % hold on
        % histogram(ItcAll_subavg);

        % figure;
        % subplot(2,1,1);
        % histogram(ChanMeanZscores_avg);
        % subplot(2,1,2);
        % histogram(ChanMeanPVal_avg);

        f6=figure;
        set(f6,'Position',[1949 123 1023 785]);
        subplot(2,1,1)
        plot(times(31:end), AVGECG.mean(31:end)', 'Color', 'k'); hold on
        set(gca,'Position',[0.1300 0.5838 0.71 0.3])
        xline(0, "--k", 'LineWidth', 2);
        axis('tight')
        ylabel('Amplitude (μV)')
        title(sprintf('Average ECG over all subjects, med: %s', medname))
        hold off


        subplot(2,1,2)
        imagesc(times(31:end),freqs,ItcAll_subavg(:,31:end));axis xy;
        colormap('parula');
        colorbar;
        clims = clim;
        hold on
        contour(times, freqs, p_thresh_all,  1, 'linecolor', 'k', 'linewidth', 1.1)
        clim(clims);
        xline(0, "--k", 'LineWidth', 2);
        xlabel('Time (s)') % Add x-label
        ylabel('Frequencies (Hz)') % Add y-label
        title(sprintf(' Average ITC in %s, perm = %d, med = %s, p<%.4g', channel, numPerms, medname, signif_thresh))

        gr6 = fullfile('F:\HeadHeart\2_results\itc\group_perm' , ['AvgITC_', channel, '_', medname, '_perm=', num2str(numPerms), '.png']);
        exportgraphics(f6,gr6, 'Resolution', 300)



    end
end

if ismember('Plot Power',steps)
    fprintf('Plot Power\n');
    for sub = 1:numel(subjects)

        % Extract the subject
        subject = subjects{sub};

        % pattern = fullfile(data_dir, 'tfr', Hz_dir, [subject, '_TFR-EPOCH_', medname, '*']);
        % files = dir(pattern);
        % filename = fullfile(files(1).folder, files(1).name);
        % load(filename, 'TFR', '-mat');
        if baseline & BPReref & BPRerefHi
            fprintf('Loading TFR Data for %s\n', subject);
            pattern = fullfile(data_dir, 'tfr', Hz_dir, [subject,  '_TFR-EPOCH_', medname ,'_Rest_', BPRerefHiTit, '*']);
            files = dir(pattern);
            filename = fullfile(files(1).folder, files(1).name);
            load(filename, 'TFR', '-mat');
        elseif baseline & BPReref & BPRerefLw
            fprintf('Loading TFR Data for %s\n', subject);
            pattern = fullfile(data_dir, 'tfr', Hz_dir, [subject,  '_TFR-EPOCH_', medname ,'_Rest_', BPRerefLwTit, '*']);
            files = dir(pattern);
            filename = fullfile(files(1).folder, files(1).name);
            load(filename, 'TFR', '-mat');
        elseif baseline & BPReref & BPRerefBest
            fprintf('Loading TFR Data for %s\n', subject);
            pattern = fullfile(data_dir, 'tfr', Hz_dir, [subject,  '_TFR-EPOCH_', medname ,'_Rest_', BPRerefBestTit, '*']);
            files = dir(pattern);
            filename = fullfile(files(1).folder, files(1).name);
            load(filename, 'TFR', '-mat');
        elseif baseline
            fprintf('Loading TFR Data for %s\n', subject);
            pattern = fullfile(data_dir, 'tfr', Hz_dir, [subject, '_TFR-EPOCH_', medname, '*', '_BSL=', '*']);
            files = dir(pattern);
            filename = fullfile(files(1).folder, files(1).name);
            load(filename, 'TFR', '-mat');
        else
            fprintf('Loading TFR Data for %s\n', subject);
            pattern = fullfile(data_dir, 'tfr', Hz_dir, [subject, '_TFR-EPOCH_', medname, '*']);
            files = dir(pattern);
            filename = fullfile(files(1).folder, files(1).name);
            load(filename, 'TFR', '-mat');
        end

        freqs = TFR.freqs;
        times = TFR.times;

        if allchans
            channels = FltSubsChansStn{sub};
        elseif onlyeeg
            channels = FltSubsOnlyEEG{sub};
        elseif onlystn
            channels = FltSubsOnlyStn{sub};
        end

        for c = 1:numel(channels)

            channel = channels{c};

            pow_all(sub,c,:,:) = squeeze(mean(TFR.(channel).pow,1)); % Mean over all Trials
        end

    end

    if ismember('Plot Plow Single Channels', steps)
        % Get unique channel names across all subjects
        if allchans
        all_channels = unique([FltSubsChansStn{:}]); % Extract all channels present across subjects
        elseif onlyeeg
        all_channels = unique([FltSubsOnlyEEG{:}]); 
        elseif onlystn
        all_channels = unique([FltSubsOnlyStn{:}]);
        end

        % Initialize storage for averaged power per channel
        pow_channel_avg = nan(numel(all_channels), size(pow_all,3), size(pow_all,4));  % [channels, freqs, times]

        % Loop through each unique channel
        for ch = 1:numel(all_channels)
            channel_name = all_channels{ch};
            subject_pow = nan(numel(subjects), size(pow_all,3), size(pow_all,4));  % Store per subject

            % Loop through subjects
            for sub = 1:numel(subjects)
                if allchans
                    subject_channels = FltSubsChansStn{sub}; % Channels for this subject
                elseif onlyeeg
                    subject_channels = FltSubsOnlyEEG{sub}; % Channels for this subject
                elseif onlystn
                    subject_channels = FltSubsOnlyStn{sub}; % Channels for this subject
                end


                chan_idx = find(strcmp(subject_channels, channel_name)); % Find index

                if ~isempty(chan_idx) % Ensure channel exists for subject
                    subject_pow(sub, :, :) = squeeze(pow_all(sub, chan_idx, :, :));
                end
            end

            % Compute mean over subjects (ignoring NaNs where channels are missing)
            pow_channel_avg(ch, :, :) = squeeze(mean(subject_pow, 1, 'omitnan'));
        end

        % Extract frequency and time axes
        times = TFR.times;
        freqs = TFR.freqs;

        % Plot time-frequency power for each channel
        figure;
        for ch = 1:numel(all_channels)
            
            f7=figure;
            set(f7,'Position',[1949 123 1023 785]);
            subplot(2,1,1)
            plot(times(31:end), AVGECG.mean(31:end)', 'Color', 'k'); hold on
            set(gca,'Position',[0.1300 0.5838 0.71 0.3])
            xline(0, "--k", 'LineWidth', 2);
            title(sprintf('Average ECG over all Sub'))
            hold off
            subplot(2,1,2)
            imagesc(times(31:end), freqs, squeeze(pow_channel_avg(ch,31:end,:)));axis xy;
            xlabel('Time (s)');
            ylabel('Frequency (Hz)');
            colormap('parula');
            colorbar;
            clims = clim;
            % hold on
            % contour(times, freqs, p_thresh_all,  1, 'linecolor', 'k', 'linewidth', 1.1)
            % clim(clims);
            xline(0, "--k", 'LineWidth', 2);
            title(sprintf('Average Power in %s, med = %s', all_channels{ch},  medname ));

            if BPReref & BPRerefHi
            gr7 = fullfile(results_dir, 'power', Hz_dir,  'group' , ['AvgPower_', all_channels{ch}, '_', BPRerefHiTit ,'_',medname, '.png']);
            elseif BPReref & BPRerefLw
            gr7 = fullfile(results_dir, 'power', Hz_dir,  'group' , ['AvgPower_', all_channels{ch}, '_', BPRerefLwTit ,'_',medname, '.png']);
            end
            exportgraphics(f7,gr7, 'Resolution', 300)
        end
    end

    if ismember('Plot Power Cluster', steps)

        % Define clusters
        clusters = ["Frontal", "Central", "Parietal", "STNleft", "STNright"];
        Frontal = ["F3", "F4", "Fz"];
        Central = ["C3", "C4", "Cz"];
        Parietal = ["P3", "P4", "Pz"];
        STNleft = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"];
        STNright = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"];

        % Store in a struct for easy access
        clusterMap = struct('Frontal', Frontal, 'Central', Central, 'Parietal', Parietal, 'STNleft', STNleft, 'STNright', STNright);

        % Initialize storage for cluster-averaged power
        pow_cluster_avg = nan(numel(clusters), size(pow_all,3), size(pow_all,4)); % [clusters, freqs, times]
        
        if allchans
            start = 1; numofcluster = numel(clusters);
        elseif onlyeeg
            start = 1; numofcluster = 3;
        elseif onlystn
            start = 4; numofcluster = 5;
        end

        % Loop through clusters
        for cl = start : numofcluster %numel(clusters)
            cluster_name = clusters(cl);
            cluster_chans = clusterMap.(cluster_name);  % Get channel list for this cluster

            subject_pow = nan(numel(subjects), size(pow_all,3), size(pow_all,4)); % Store avg per subject

            % Loop through subjects
            for sub = 1:numel(subjects)

                if allchans
                    subject_channels = FltSubsChansStn{sub}; % Channels for this subject
                elseif onlyeeg
                    subject_channels = FltSubsOnlyEEG{sub}; % Channels for this subject
                elseif onlystn
                    subject_channels = FltSubsOnlyStn{sub}; % Channels for this subject
                end

                % Find indices of channels that belong to the cluster
                cluster_idx = find(ismember(subject_channels, cluster_chans));

                if ~isempty(cluster_idx)  % Ensure at least one matching channel exists
                    % Average power across cluster channels for this subject
                    subject_pow(sub, :, :) = squeeze(mean(pow_all(sub, cluster_idx, :, :), 2));
                end
            end

            % Average across subjects
            pow_cluster_avg(cl, :, :) = squeeze(mean(subject_pow, 1, 'omitnan'));
        end

        % Extract frequency and time axes
        times = TFR.times;
        freqs = TFR.freqs;

        % Plot time-frequency power for each cluster
        figure;
        for cl = 1:numel(clusters)

            f8=figure;
            set(f8,'Position',[1949 123 1023 785]);
            subplot(2,1,1)
            plot(times(31:end), AVGECG.mean(31:end)', 'Color', 'k'); hold on
            set(gca,'Position',[0.1300 0.5838 0.71 0.3])
            xline(0, "--k", 'LineWidth', 2);
            title(sprintf('Average ECG over all Sub'))
            hold off
            subplot(2,1,2)
            imagesc(times(31:end), freqs, squeeze(pow_cluster_avg(cl,31:end,:))); axis xy;
            xlabel('Time (s)');
            ylabel('Frequency (Hz)');
            colormap('parula');
            colorbar;
            clims = clim;
            % hold on
            % contour(times, freqs, p_thresh_all,  1, 'linecolor', 'k', 'linewidth', 1.1)
            % clim(clims);
            xline(0, "--k", 'LineWidth', 2);
            title(sprintf('Average  Power in %s, med = %s', clusters{cl},  medname ))

            if BPReref & BPRerefHi
            gr8 = fullfile(results_dir, 'power', Hz_dir,  'group' , ['AvgPower_', clusters{cl}, '_', BPRerefHiTit , '_', medname, '.png']);
            elseif BPReref & BPRerefLw
            gr8 = fullfile(results_dir, 'power', Hz_dir,  'group' , ['AvgPower_', clusters{cl}, '_', BPRerefLwTit , '_', medname, '.png']);
            end
            exportgraphics(f8,gr8, 'Resolution', 300)
        end
    end
    
    
%     clusters = ["Frontal", "Central", "Parietal", "STNleft", "STNright"];
% Frontal = ["F3", "F4", "Fz"];
% Central = ["C3", "C4", "Cz"];
% Parietal = ["P3", "P4", "Pz"];
% STNleft = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"];
% STNright = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"];
% 
% 
% % Store in a struct for easy access
% clusterMap = struct('Frontal', Frontal, 'Central', Central, 'Parietal', Parietal, 'STNleft', STNleft, 'STNright', STNright);
% 
% for ci = 1:numel(clusters)
%     cluster_name = clusters{ci}; % e.g., 'frontal'
%     cluster_channels = clusterMap.(cluster_name); % Get corresponding channels
% 
% 
%     for c = 1:numel(cluster_channels)
%         channel = cluster_channels{c};
% 
%         chanIdx = find(ismember(subject_channels, cluster_channels));
% 
%         if isempty(chanIdx)
%             warning('No matching channels for subject %s in cluster %s.', subject, cluster_name);
%             continue;
%         end
% 
%         % Average over all subs for each channel
%         PowAll_subavg = squeeze(mean(squeeze(pow_all(:,c,:,:)),1));
% 
%         f7=figure;
%         set(f7,'Position',[1949 123 1023 785]);
%         subplot(2,1,1)
%         plot(times(31:end), AVGECG.mean(31:end)', 'Color', 'k'); hold on
%         set(gca,'Position',[0.1300 0.5838 0.71 0.3])
%         xline(0, "--k", 'LineWidth', 2);
%         title(sprintf('Average ECG over all Sub'))
%         hold off
%         subplot(2,1,2)
%         imagesc(times(31:end),freqs,PowAll_subavg(:,31:end));axis xy;
%         colormap('parula');
%         colorbar;
%         clims = clim;
%         % hold on
%         % contour(times, freqs, p_thresh_all,  1, 'linecolor', 'k', 'linewidth', 1.1)
%         % clim(clims);
%         xline(0, "--k", 'LineWidth', 2);
%         title(sprintf('Average  Power in %s, med = %s', channel,  medname ))
% 
%         gr7 = fullfile('F:\HeadHeart\2_results\power\', Hz_dir,  'group' , ['AvgPower_', channel, '_', medname, '.png']);
%         exportgraphics(f7,gr7, 'Resolution', 300)
%     end

%end

end





function   [ChsAllFrsTmPha] = time_lock_to_surrogate(ChDta, surrogate_rpeaks, SR, tWidth, tOffset, numPerms, NewSR, Frqs, times)
dtTime = 1/NewSR;
nEvent=length(surrogate_rpeaks);
BandWidth=2; % BandWidth in Hz;
Qfac     =2; % Attenuation in db(-Qfac)
FltPassDir='twopass';
baseline=true;
baseline_win = [-0.3 -0.1];


nWidth=int32(tWidth/dtTime)+1;
nOffset=int32(tOffset/dtTime)+1;



% HIGH PASS FILTER
ChDta=ft_preproc_highpassfilter(ChDta,SR,2,4,'but', 'twopass'); % twopass


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


% define posible namber of Events
% Find possible events for the current permutation
MasEnableEvents = int32(zeros(nEvent, 1));
nPosibleEvent = int32(0);
nDataAll = length(ChDta);

for i = 1:nEvent
    currTime = int32(surrogate_rpeaks(i)/dtTime);
    n1 = currTime - nOffset;
    n2 = n1 + nWidth - 1;

    if n1 > 0 && n2 < nDataAll
        nPosibleEvent = nPosibleEvent + 1;
        MasEnableEvents(nPosibleEvent) = i;
    end
end

nEvent = nPosibleEvent;  % Update nEvent to the number of valid events
EvData=zeros(nEvent,nWidth);
% Time-lock the data for this permutation
EvTime = zeros(1, nEvent);

for i = 1:nEvent
    EvTime(i) = surrogate_rpeaks(MasEnableEvents(i));
    currTime = int32(EvTime(i)/dtTime);
    n1 = currTime - nOffset;
    n2 = n1 + nWidth - 1;
    EvData(i, :) = ChDta(:, n1:n2);
end

[nEvs,nData]=size(EvData);

if baseline
    % My baseline is the time window -0.3 to -0.1 s
    % before my Rpeak of every trial

    % Find the the indices for the basseline window
    bidx = find(times' >= baseline_win(1) & times' <= baseline_win(2));

    % for every trial calc the mean of the baseline win
    % and subtract that from the entire epoch
    for t = 1:nEvs
        baseline_mean = mean(EvData(t, bidx(1):bidx(end)),2);
        EvData(t,:) = EvData(t,:)-baseline_mean;
    end

end

ChsCmxEvFrTm = zeros(nEvs,numel(Frqs),nData);

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
%ChsAllFrsTmSpc=zeros(nEvs,nFrs,nData);
ChsAllFrsTmPha=zeros(nEvs,nFrs,nData);
for iev=1:nEvs
    for ifr=1:nFrs
        xlb=squeeze(ChsCmxEvFrTm(iev,ifr,:));
        %df=abs(xlb);
        %ChsAllFrsTmSpc(iev,ifr,:)=df; % Power (eventxfreqxpower)
        ChsAllFrsTmPha(iev,ifr,:)=angle(xlb); %Phase (eventxfreqxphase)
    end
end
end
