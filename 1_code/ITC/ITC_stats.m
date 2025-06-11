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
newsubs = false;
oldsubs = false;
allsubs = true;

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
steps = {'Correlation Group ITC'}; %  'Group TFR Power Save', 'Group ITC Load Chan', 'Plot SubAvg PermStats', 'Plot ITC Group Cluster', 'Group ITC Save', 'Group ITC' 'Plot SubAvg PermStats', 'Calc Single Subject ITC', 'Plot SubAvg ITC', 'Plot Power', 'Plot Plow Single Channels', 'Plot Power Cluster',

% Define Time Window
tWidth   = 0.9;
tOffset  = 0.3;
FltPassDir = 'twopass';
NewSR = 300;

BandWidth = 2; % BandWidth in Hz;
Qfac      = 2; % Attenuation in db(-Qfac)
tCircMean = 0; %0.02; % for By TRials calc

permstats = true;
numPerms = 1000;
surrogate = true;
trials = false;
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
    %PermItcAll = zeros(numel(subjects), max_chan, numPerms, 148, 271);
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
            ItcSub(c,:,:) = FrsTmItc; % ChannelxFreqxTime
            RelItcAll(sub,c,:,:) = relFrsTmItc;
            RelItcSub(c,:,:) = relFrsTmItc;
        end

        outputPDF1 = fullfile(results_dir, 'itc' , Hz_dir, 'ss', [subject, '_ITC_Allchan_med=', medname, ...
            '_win=-', num2str(tOffset),'to', num2str(tWidth-tOffset),'_BSL=', num2str(baseline), '.pdf']);

        for c = 1:numel(channels)
            channel = channels{c};

            if show_plots
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
            outputPDF = fullfile(results_dir, 'itc', Hz_dir, 'ss_perm' , [subject, '_ITC-PermStats_Allchan_med=', medname, '_perm=', num2str(numPerms), ...
                '_win=-', num2str(tOffset),'to', num2str(tWidth-tOffset),'_BSL=', num2str(baseline), '_pval=', num2str(signif_thresh),'.pdf']);

            %Intitalize the PermItcAllChan Variable
            PermItcAllChan = zeros(numel(channels),numPerms, numel(freqs), numel(times));

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

                    parfor perm = 1:numPerms % here change to parfor
                        % Time Lock the surrogate R Peaks to the Channel
                        % Data and apply the filters as well as the DS and
                        % create TFR for the new epochs
                        currSurrogateRpeaks = surrogate_rpeaks(perm, :);
                        [ChsAllFrsTmPha] = time_lock_to_surrogate(ChDta, currSurrogateRpeaks, oldSR, tWidth, tOffset, NewSR, freqs, times, onlyeeg);

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
                PermItcAllChan(c,:,:,:) = PermItcData; % ChannelxPermutationxFreqxTime


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
                set(f5,'Position',[159 50 1122 774.5000]);

                % Upper subplot
                subplot(2,1,1)
                plot(times, mean(EvECG.EvData,1), 'Color', 'k'); hold on
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

        if permstats
            %ITC.PERM.ZScoresAll = ZScoresAll;  % SubjectxChannelxFreqxTime (Last two are ITC ZScores)
            %ITC.PERM.PValAll = PValAll; % SubjectxChannelxFreqxTime (Last two are ITC PVals)
            ITC.PermItc = PermItcAllChan; %SubjectxChannelxPermutationxFreqxTime
        end
        % Save ITC MAtrix of all Subjects and all Channels
        ITC.SR = SR;
        ITC.times = times;
        ITC.freqs = freqs;
        ITC.ItcAll = ItcSub;
        ITC.RelItcAll = RelItcSub;

        if permstats & BPRerefHi & onlystn
            save_path = fullfile(data_dir, 'itc', 'ss', [subject, '_ITC_', medname , '_OnlySTN-BP=', BPRerefHiTit, '_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
        elseif permstats & BPRerefLw & onlystn
            save_path = fullfile(data_dir, 'itc', 'ss',[subject, '_ITC_', medname , '_OnlySTN-BP=', BPRerefLwTit, '_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
        elseif permstats & onlyeeg
            save_path = fullfile(data_dir, 'itc', 'ss',[subject,'_ITC_', medname ,'_OnlyEEG_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
        elseif permstats & allchans & BPRerefHi
            save_path = fullfile(data_dir, 'itc', 'ss',[subject, '_ITC_', medname ,'_BP=', BPRerefHiTit,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
        elseif permstats & allchans & BPRerefLw
            save_path = fullfile(data_dir, 'itc', 'ss',[subject, '_ITC_', medname ,'_BP=', BPRerefLwTit,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
        elseif permstats & allchans
            save_path = fullfile(data_dir, 'itc', 'ss',[subject, '_ITC_', medname ,'_BP=NONE','_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
        elseif permstats
            save_path = fullfile(data_dir, 'itc', 'ss',[subject,'_ITC_', medname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
        else
            save_path = fullfile(data_dir, 'itc', 'ss',[subject,'_ITC_', medname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), 'HPF=', num2str(freqs(1)), '_HP=',  num2str(freqs(1)) ,'.mat']);
        end
        save(save_path, 'ITC', '-v7.3');
        fprintf('Saved ITC Data for all subs and channels to: %s\n', save_path);

        % Free up some storage
        clear ItcSub
        clear RelItcSub
        clear PermItcData
        clear PermItcAllChan

    end
    %     if permstats
    %        % ITC.PERM.ZScoresAll = ZScoresAll;  % SubjectxChannelxFreqxTime (Last two are ITC ZScores)
    %         %ITC.PERM.PValAll = PValAll; % SubjectxChannelxFreqxTime (Last two are ITC PVals)
    %         ITC.PERM.PermItcAll = PermItcAll; %SubjectxChannelxPermutationxFreqxTime
    %     end
    %     % Save ITC MAtrix of all Subjects and all Channels
    %     ITC.SR = SR;
    %     ITC.times = times;
    %     ITC.freqs = freqs;
    %     ITC.ItcAll = ItcAll;
    %     ITC.RelItcAll = RelItcAll;
    %
    %     % Create Paths
    %     if permstats & newsubs & BPRerefHi & onlystn
    %         save_path = fullfile(data_dir, 'itc', ['ITC-NewSubs_', medname , '_OnlySTN_BP=', BPRerefHiTit, '_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    %     elseif permstats & newsubs & BPRerefLw & onlystn
    %         save_path = fullfile(data_dir, 'itc', ['ITC-NewSubs_', medname , '_OnlySTN_BP=', BPRerefLwTit, '_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    %     elseif permstats & newsubs & onlyeeg
    %         save_path = fullfile(data_dir, 'itc', ['ITC-NewSubs_', medname ,'_OnlyEEG_BP=None_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    %     elseif permstats & newsubs & allchans
    %         save_path = fullfile(data_dir, 'itc', ['ITC-NewSubs_', medname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    %     elseif permstats & oldsubs & BPRerefHi & onlystn
    %         save_path = fullfile(data_dir, 'itc', ['ITC-OldSubs_', medname , '_OnlySTN_BP=', BPRerefHiTit, '_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    %     elseif permstats & oldsubs & BPRerefLw & onlystn
    %         save_path = fullfile(data_dir, 'itc', ['ITC-OldSubs_', medname , '_OnlySTN_BP=', BPRerefLwTit,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    %     elseif permstats & oldsubs & onlyeeg
    %         save_path = fullfile(data_dir, 'itc', ['ITC-OldSubs_', medname ,'_OnlyEEG_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    %     elseif permstats & oldsubs & allchans
    %         save_path = fullfile(data_dir, 'itc', ['ITC-OldSubs_', medname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    %     elseif permstats
    %         save_path = fullfile(data_dir, 'itc', ['ITC-AllSubs_', medname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    %     else
    %         save_path = fullfile(data_dir, 'itc', ['ITC-AllSubs_', medname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), 'HPF=', num2str(freqs(1)), '_HP=',  num2str(freqs(1)) ,'.mat']);
    %     end
    %     save(save_path, 'ITC', '-v7.3');
    %     fprintf('Saved ITC Data for all subs and channels to: %s\n', save_path);
    fprintf('Done!')
end

% % PLOT THE AVERAGE ITC PER SUBJECT
% if ismember('Plot SubAvg ITC', steps)
%     fprintf('Plot SubAvg ITC\n');
%     if ismember('Calc Single Subject ITC', steps) == false
%
%         fprintf('Loading ITC Data\n');
%         pattern = fullfile(data_dir, 'itc', ['ITC-AllSubs_',  medname, '*', '_HP=', Hz_dir(1:end-2), '*']);
%         files = dir(pattern);
%         filename = fullfile(files(1).folder, files(1).name);
%         load(filename, 'ITC', '-mat');
%         SR= ITC.SR;
%         freqs = ITC.freqs;
%
%     end
%     plots = true;
%
%     if allchans
%         channels = FltSubsChansStn{sub};
%     elseif onlyeeg
%         channels = FltSubsOnlyEEG{sub};
%     elseif onlystn
%         channels = FltSubsOnlySTN{sub};
%     end
%
%     for c = 1:numel(channels)
%         channel = channels{c};
%
%         ItcAll_subavg = squeeze(mean(squeeze(ITC.ItcAll(:,c,:,:)),1));
%         RelItcAll_subavg = squeeze(mean(squeeze(ITC.RelItcAll(:,c,:,:)),1));
%
%         if plots
%             f2=figure;
%             set(f2,'Position',[1949 123 1023 785]);
%             subplot(2,1,1)
%             plot(times, AVGECG.mean', 'Color', 'k'); hold on
%             set(gca,'Position',[0.0900  0.6838 0.78 0.2])
%             xline(0, "--k", 'LineWidth', 2);
%             ylabel('Amplitude')
%             axis('tight')
%             title(sprintf('Average ECG over all subjects, medication: %s', medname))
%             hold off
%             subplot(2,1,2)
%             set(gca,'Position',[0.0900 0.1200 0.8498 0.4612])
%             imagesc(times,freqs,ItcAll_subavg);axis xy;
%             colormap('parula');
%             xline(0, "--k", 'LineWidth', 2);
%             col = colorbar;
%             col.Label.String = 'ITC Values'; % Add title to colorbar
%             xlabel('Time (s)') % Add x-label
%             ylabel('Frequencies (Hz)') % Add y-label
%             title(sprintf('Average ITC for %s, med: %s, HP: %s', channel, medname, Hz_dir))
%
%             gr2 = fullfile('F:\HeadHeart\2_results\itc' , Hz_dir, 'group', ['ITC_', channel, '_', medname,  '.png']);
%             exportgraphics(f2,gr2, 'Resolution', 300)
%         end
%     end
% end

% Now here is the Grand Average ITC over all Subs per Channel
if ismember('Group ITC Save', steps)
    fprintf('Load Data for Group ITC\n');

    for s = 1:numel(subjects)
        subject = subjects(s);
        fprintf('Loading ITC Data for subject %s\n', subject);

        filelist = dir(fullfile(data_dir, 'itc', 'ss', [char(subject), '*']));
        %filelist = dir(fullfile(data_dir, 'tfr', '2Hz', [char(subject), '*', medname, '*', 'Freq=0.5', '*']));

        % Initialize empty
        ItcDataSub_cell = {};            %% <<< CHANGED
        ItcPermSub_cell = {};            %% <<< CHANGED
        ItcLabels = strings(0,1);
        ItcChannels = {};

        % --- Info about which BP reref electrode to use ---
        use_hi_L = subject_info(s).ITC_BPReref_L;
        use_hi_R = subject_info(s).ITC_BPReref_R;

        if use_hi_L == 1; expected_L = 'Hi'; else expected_L = 'Lw'; end
        if use_hi_R == 1; expected_R = 'Hi'; else expected_R = 'Lw'; end

        for i = 1:numel(filelist)
            fname = filelist(i).name;
            fullpath = fullfile(filelist(i).folder, fname);
            fprintf('Loading: %s\n', fname);

            load(fullpath, 'ITC', '-mat');

            parts = split(fname, '_');
            BPkey = parts{5};
            STNkey = parts{4};

            if contains(STNkey, 'OnlySTN')
                channels = FltSubsOnlyStn{s};
            elseif contains(STNkey, 'OnlyEEG')
                channels = FltSubsOnlyEEG{s};
            else
                channels = FltSubsChansStn{s};
            end

            if contains(BPkey, 'BPRerefHi')
                reref_type = 'Hi';
            elseif contains(BPkey, 'BPRerefLow')
                reref_type = 'Lw';
            elseif contains(BPkey, 'time')
                fprintf('This is an EEG file: %s\n', fname);
            end

            keep_idx = false(size(channels));
            for ch_idx = 1:numel(channels)
                chan_name = channels{ch_idx};
                if startsWith(chan_name, 'L')
                    keep_idx(ch_idx) = strcmp(reref_type, expected_L);
                elseif startsWith(chan_name, 'R')
                    keep_idx(ch_idx) = strcmp(reref_type, expected_R);
                else
                    keep_idx(ch_idx) = true;
                end
            end

            kept_channels = channels(keep_idx);
            ItcDataFiltered = ITC.ItcAll(keep_idx,:,:);
            ItcPermFiltered = ITC.PermItc(keep_idx,:,:,:);

            ItcDataSub_cell{end+1} = ItcDataFiltered;     %% <<< CHANGED
            ItcPermSub_cell{end+1} = ItcPermFiltered;     %% <<< CHANGED

            label_combined = strcat(string(BPkey), "_", string(STNkey));
            new_labels = repmat(label_combined, numel(kept_channels), 1);
            ItcLabels = [ItcLabels; new_labels];
            ItcChannels = [ItcChannels, kept_channels];
        end

        % Concatenate
        ItcDataSub = cat(1, ItcDataSub_cell{:});         %% <<< CHANGED
        ItcPermSub = cat(1, ItcPermSub_cell{:});         %% <<< CHANGED

        % Additional info
        SR = ITC.SR;
        freqs = ITC.freqs;
        times = ITC.times;
        ItcAllSubsLabels{s} = ItcLabels;
        ItcAllSubsChannels{s} = ItcChannels;

        % --- Save per-channel data to disk --- %% <<< NEW
        for c = 1:numel(ItcChannels)
            ch_name = ItcChannels{c};

            out_fname = sprintf('%s_%s_ITC_MedOn_time=-0.3-0.6_DS=300_perm=1000_HP=0.5.mat', char(subject), ch_name);
            out_path = fullfile(data_dir, 'itc' ,'ss_chan', out_fname);

            % Extract per-channel data
            ItcDataCh = squeeze(ItcDataSub(c,:,:));
            PermDataCh = squeeze(ItcPermSub(c,:,:,:));

            % Save to file (use -v7.3 for large arrays)
            save(out_path, 'ItcDataCh', 'PermDataCh', 'SR', 'freqs', 'times');
        end
    end
end

if ismember('Group ITC Load Chan', steps)

    all_files = dir(fullfile(data_dir,'itc', 'ss_chan', '*.mat'));
    valid_idx = ~startsWith({all_files.name}, '._');
    all_files = all_files(valid_idx);

    % Extract all channel names
    channel_names = strings(numel(all_files), 1);
    for i = 1:numel(all_files)
        fname = all_files(i).name;
        parts = split(fname, '_');         % Example: 'sub01_LFP1.mat'
        channel_names(i) = erase(parts{2}, '.mat');  % Get 'LFP1'
    end

    mapped_channels = strings(size(channel_names));
    for i = 1:numel(channel_names)
        chan =channel_names(i);
        if startsWith(chan, 'L')
            mapped_channels(i) = "STNleft";
        elseif startsWith(chan, 'R')
            mapped_channels(i) = "STNright";
        else
            mapped_channels(i) = chan;
        end
    end

    % Get list of unique mapped channel names
    unique_channels = unique(mapped_channels);

    % Define EEG channel clusters
    EEG_clusters = struct();
    EEG_clusters.Frontal = ["F3", "F4", "Fz"];
    EEG_clusters.Central = ["C3", "C4", "Cz"];
    EEG_clusters.Parietal = ["P3", "P4", "Pz", "Oz"];

    if ismember('Plot Single Chan', steps)
        % Initialize output struct for averages
        ITC_GroupAvg = struct();

        % Loop over channels
        for ch = 11:numel(unique_channels)
            ch_name = unique_channels(ch);
            fprintf('Processing channel: %s\n', ch_name);

            % Determine which original channels match this group
            if ch_name == "STNleft"
                match_indices = startsWith(channel_names, 'L');
            elseif ch_name == "STNright"
                match_indices = startsWith(channel_names, 'R');
            else
                match_indices = channel_names == ch_name;
            end

            % Get matching files
            files_ch = all_files(match_indices);

            % Load all subjects' ITC data for this channel
            ITC_allSubs_cell = {};
            ITC_Perm_allSubs = {};
            for f = 1:numel(files_ch)
                load(fullfile(files_ch(f).folder, files_ch(f).name), 'ItcDataCh', 'PermDataCh', 'SR', 'freqs', 'times');
                ITC_allSubs_cell{end+1} = ItcDataCh;  % [nFreqs x nTimes]
                ITC_Perm_allSubs{end+1} = PermDataCh;
            end

            % Convert to 3D matrix: [nSubjects x nFreqs x nTimes]
            nSubjects = numel(ITC_allSubs_cell);
            [nFreqs, nTimes] = size(ITC_allSubs_cell{1});
            [nPerm, nFqs, nTms] = size(ITC_Perm_allSubs{1});
            ITC_allSubs = zeros(nSubjects, nFreqs, nTimes);
            ITC_PermAvg_allsubs = zeros(nSubjects, nPerm, nFreqs, nTimes);

            for s = 1:nSubjects
                ITC_allSubs(s, :, :) = ITC_allSubs_cell{s};
                ITC_PermAvg_allsubs(s,:,:,:) = ITC_Perm_allSubs{s};
            end

            % Average over subjects
            ItcAll_subavg = squeeze(mean(ITC_allSubs,1));  % freq x time
            PermItcAll_avg = squeeze(mean(ITC_PermAvg_allsubs,1));  % permutations x freq x time

            if ismember('Plot SubAvg ITC', steps)
                fprintf('Plot ITC Averages \n');

                f2=figure;
                set(f2,'Position',[1949 123 1023 785]);
                subplot(2,1,1)
                plot(times(31:end), AVGECG.mean(31:end)', 'Color', 'k'); hold on
                set(gca,'Position',[0.0900  0.6838 0.78 0.2])
                xline(0, "--k", 'LineWidth', 2);
                ylabel('Amplitude')
                axis('tight')
                title(sprintf('Average ECG over all subjects, medication: %s', medname))
                hold off
                subplot(2,1,2)
                set(gca,'Position',[0.0900 0.1200 0.8498 0.4612])
                imagesc(times(31:end),freqs(9:end),ItcAll_subavg);axis xy;
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

            if ismember('Plot SubAvg PermStats',steps)
                fprintf('Plot ITC Averages with permutation stats \n');

                % %ChanMeanZscores_avg = squeeze(mean(squeeze(ITC.PERM.ZScoresAll(:,c,:,:)),1));  % SubjectxChannelxFreqxTime (Last two are ITC ZScores)
                % %ChanMeanPVal_avg = squeeze(mean(squeeze(ITC.PERM.PValAll(:,c,:,:)),1));  % SubjectxChannelxFreqxTime (Last two are ITC PVals)
                % PermItcAll_avg = squeeze(mean(squeeze(ITC.PERM.PermItcAll(:,c,:,:,:)),1)); %SubjectxChannelxPermutationxFreqxTime, Mean over all Subjects in one channel
                % ItcAll_subavg = squeeze(mean(squeeze(ITC.ItcAll(:,c,:,:)),1));
                % %RelItcAll_subavg = squeeze(mean(squeeze(ITC.RelItcAll(:,c,:,:)),1));

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

                % [maxVal, maxInd] = max(ItcAll_subavg(:));
                % [row, col] = ind2sub(size(ItcAll_subavg), maxInd);
                % isTrueInMask = p_thresh_all(maxInd);
                % figure
                % imagesc(times, freqs, ItcAll_subavg);axis xy;
                % contour(times, freqs, p_thresh_all,  1, 'linecolor', 'k', 'linewidth', 1.1)
                % % x = times(31 + col - 1);    % times(31) is x for col=1, so add (col-1)
                % % y = freqs(9 + row - 1);     % freqs(9) is y for row=1, so add (row-1)
                % hold on;
                % plot(times(col), freqs(row), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
                % 
                % 
                % % Maske als binäres Bild behandeln
                % [L, numRegions] = bwlabel(p_thresh_all);
                % 
                % % Region ermitteln, die den Maximalpunkt enthält
                % targetRegion = L(row, col);  % Label der Region mit maxVal [5]
                % % Alle Pixel der Zielregion
                % regionPixels = (L == targetRegion);
                % 
                % % Frequenzband (Zeilen)
                % freqRows = find(any(regionPixels, 2));  % Zeilen mit True-Werten
                % freqBand = [min(freqRows), max(freqRows)];  % [Startzeile, Endzeile]
                % 
                % % Zeitbereich (Spalten)
                % timeCols = find(any(regionPixels, 1));  % Spalten mit True-Werten
                % timeRange = [min(timeCols), max(timeCols)];  % [Startspalte, Endspalte]
                % % Frequenzband (Bezug auf freqs(9:end))
                % freqValues = freqs(9 - 1 + freqBand);  % [Start-Frequenz, End-Frequenz]
                % 
                % % Zeitbereich (Bezug auf times(31:end))
                % timeValues = times(31 - 1 + timeRange);  % [Startzeit, Endzeit
                % 
                % figure
                % imagesc(times(31:end), freqs(9:end), ItcAll_subavg(9:end,31:end));axis xy;
                % contour(times, freqs, p_thresh_all,  1, 'linecolor', 'k', 'linewidth', 1.1)
                % hold on
                % rectangle('Position', [timeValues(1), freqValues(1), ...
                %     timeValues(2)-timeValues(1), freqValues(2)-freqValues(1)], ...
                %     'EdgeColor', 'r', 'LineWidth', 2);
                % hold off;
                % 


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
                set(gca,'Position',[0.1300 0.5838 0.73 0.3])
                xline(0, "--k", 'LineWidth', 2);
                axis('tight')
                ylabel('Amplitude (μV)')
                title(sprintf('Average ECG over all subjects, med: %s', medname))
                hold off


                subplot(2,1,2)
                imagesc(times(31:end),freqs(9:end),ItcAll_subavg(9:end,31:end));axis xy;
                colormap('parula');
                colorbar;
                clims = clim;
                hold on
                contour(times, freqs, p_thresh_all,  1, 'linecolor', 'k', 'linewidth', 1.1)
                clim(clims);
                xline(0, "--k", 'LineWidth', 2);
                xlabel('Time (s)') % Add x-label
                ylabel('Frequencies (Hz)') % Add y-label
                title(sprintf(' Average ITC in %s, perm = %d, med = %s, p<%.4g, n=%d', ch_name, numPerms, medname, signif_thresh, nSubjects))

                gr6 = fullfile(results_dir, 'itc', Hz_dir, 'group_perm' , ['AvgITC_', char(ch_name), '_', medname, '_perm=', num2str(numPerms), '_n=', num2str(nSubjects), '.png']);
                exportgraphics(f6, gr6, 'Resolution', 300)

            end
        end
    end
    if ismember('Plot ITC Group Cluster', steps)
        % ========= Cluster-level ITC analysis for EEG groups ==========
        fprintf('\n Now calculating cluster-level EEG ITC...\n');

        cluster_names = fieldnames(EEG_clusters);
        for ci = 2:numel(cluster_names)
            clus_name = cluster_names{ci};
            clus_chans = EEG_clusters.(clus_name);

            % Find matching file indices for channels in this cluster
            match_indices = ismember(channel_names, clus_chans);
            files_cluster = all_files(match_indices);

            if isempty(files_cluster)
                warning('No matching channels found for cluster %s', clus_name);
                continue
            end

            % Load ITC and perm data for all matching files
            ITC_allSubs_cell = {};
            ITC_Perm_allSubs = {};
            for f = 1:numel(files_cluster)
                load(fullfile(files_cluster(f).folder, files_cluster(f).name), 'ItcDataCh', 'PermDataCh');
                ITC_allSubs_cell{end+1} = ItcDataCh;
                ITC_Perm_allSubs{end+1} = PermDataCh;
            end

            % Convert to 4D matrix: [nSubjects x nChannels x nFreqs x nTimes]
            nSubjects = numel(ITC_allSubs_cell);
            [nFreqs, nTimes] = size(ITC_allSubs_cell{1});
            [nPerm, ~, ~] = size(ITC_Perm_allSubs{1});
            nChs = numel(ITC_allSubs_cell);

            ITC_matrix = zeros(nChs, nFreqs, nTimes);
            PERM_matrix = zeros(nChs, nPerm, nFreqs, nTimes);

            for c = 1:nChs
                ITC_matrix(c,:,:) = ITC_allSubs_cell{c};
                PERM_matrix(c,:,:,:) = ITC_Perm_allSubs{c};
                fprintf('\n Now calculating chan %s\n', c);
            end

            % Average across EEG cluster channels
            ItcAll_clusAvg = squeeze(mean(ITC_matrix,1));         % [freq x time]
            PermItcAll_avg = squeeze(mean(PERM_matrix,1));        % [perm x freq x time]

            % Z-scoring
            perm_mean = squeeze(mean(PermItcAll_avg, 1));
            perm_std  = squeeze(std(PermItcAll_avg, 1));
            zscores_all = (ItcAll_clusAvg - perm_mean) ./ perm_std;
            p_orig_all = 2 * (1 - normcdf(zscores_all, 0, 1));
            signif_thresh = 0.005;
            p_thresh_all = p_orig_all < signif_thresh;

            f7=figure;
            set(f7,'Position',[1949 123 1023 785]);
            subplot(2,1,1)
            plot(times(31:end), AVGECG.mean(31:end)', 'Color', 'k'); hold on
            set(gca,'Position',[0.1300 0.5838 0.73 0.3])
            xline(0, "--k", 'LineWidth', 2);
            axis('tight')
            ylabel('Amplitude (μV)')
            title(sprintf('Average ECG over all subjects, med: %s', medname))
            hold off


            subplot(2,1,2)
            imagesc(times(31:end),freqs(9:end)',ItcAll_clusAvg(9:end,31:end));axis xy;
            colormap('parula');
            colorbar;
            clims = clim;
            hold on
            contour(times(31:end), freqs(9:end)', p_thresh_all(9:end,31:end),  1, 'linecolor', 'k', 'linewidth', 1.1)
            clim(clims);
            xline(0, "--k", 'LineWidth', 2);
            xlabel('Time (s)') % Add x-label
            ylabel('Frequencies (Hz)') % Add y-label
            title(sprintf(' Average ITC in %s, perm = %d, med = %s, p<%.4g', clus_name, numPerms, medname, signif_thresh))

            % Save
            outname = fullfile(results_dir, 'itc', Hz_dir, 'group_perm', ...
                ['ClusterAvgITC_', clus_name, '_', medname, '_perm=', num2str(numPerms), '.png']);
            exportgraphics(f7, outname, 'Resolution', 300);
        end
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

if ismember('Group TFR Power Save', steps)
    fprintf('Load Data for Group TFR Power\n');

    for s = 13:numel(subjects)
        subject = subjects(s);
        fprintf('Loading TFR Data for subject %s\n', subject);

        % TFR file pattern (Hilbert only, exclude non-BP reref files)
        filelist = dir(fullfile(data_dir, 'tfr', '2Hz', ...
            [char(subject), '*', medname, '*', 'Freq=0.5', '*.mat']));

        PowerDataSub_cell = {};
        PowerLabels = strings(0, 1);
        PowerChannels = {};

        % Subject reref preferences
        use_hi_L = subject_info(s).ITC_BPReref_L;
        use_hi_R = subject_info(s).ITC_BPReref_R;

        expected_L = 'Hi'; if ~use_hi_L, expected_L = 'Lw'; end
        expected_R = 'Hi'; if ~use_hi_R, expected_R = 'Lw'; end

        for i = 1:numel(filelist)
            fname = filelist(i).name;
            fullpath = fullfile(filelist(i).folder, fname);
            fprintf('Loading: %s\n', fname);

            load(fullpath, 'TFR', '-mat');

            parts = split(fname, '_');
            BPkey = parts{5};

            channels = FltSubsChansStn{s};

            % Determine reref type
            if contains(BPkey, 'BPRerefHi')
                reref_type = 'Hi';
            elseif contains(BPkey, 'BPRerefLow')
                reref_type = 'Lw';
            else
                fprintf('Skipping file without BP reref info: %s\n', fname);
                continue;
            end

            % Apply channel selection logic
            keep_idx = false(size(channels));
            for ch_idx = 1:numel(channels)
                chan_name = channels{ch_idx};
                if startsWith(chan_name, 'L')
                    keep_idx(ch_idx) = strcmp(reref_type, expected_L);
                elseif startsWith(chan_name, 'R')
                    keep_idx(ch_idx) = strcmp(reref_type, expected_R);
                else
                    keep_idx(ch_idx) = true;  % EEG
                end
            end

            kept_channels = channels(keep_idx);
            freqs = TFR.freqs;
            times = TFR.times;

            % Store power data
            for ch_idx = 1:numel(kept_channels)
                ch = kept_channels{ch_idx};
                if ~isfield(TFR, ch) || ~isfield(TFR.(ch), 'pow')
                    warning('Missing channel or power field: %s', ch);
                    continue;
                end
                if sum(strcmp(ch, PowerChannels)) == 1
                    warning('EEG channel %s exists already', ch);
                    continue;
                else
                    PowerDataSub_cell{end+1} = squeeze(mean(TFR.(ch).pow,1));
                    PowerChannels = [PowerChannels, ch];
                end

                label_combined = strcat(string(BPkey));
                PowerLabels = [PowerLabels; label_combined];
            end
        end

        % Save per-channel files
        for c = 1:numel(PowerChannels)
            ch_name = PowerChannels{c};
            pow_mat = PowerDataSub_cell{c};  % [freq x time]

            out_fname = sprintf('%s_%s_Power_MedOn_time=-0.3-0.6_DS=300_HP=2Hz.mat', ...
                char(subject), ch_name);
            out_path = fullfile(data_dir, 'tfr', 'ss_chan', out_fname);

            POW = pow_mat;

            % Save power
            save(out_path, 'POW', 'freqs', 'times');  % label optional
        end
    end
end


if ismember('Correlation Group ITC', steps)

    % === Define TF window ===
    tf_time_win = [0.1, 0.25];     % seconds
    tf_freq_win = [2, 5];          % Hz

    freq_idx = find(freqs >= tf_freq_win(1) & freqs <= tf_freq_win(2));
    time_idx = find(times >= tf_time_win(1) & times <= tf_time_win(2));

    % === Load all data ===
    all_files = dir(fullfile(data_dir, 'itc', 'ss_chan', '*.mat'));
    pow_files = dir(fullfile(data_dir, 'tfr', 'ss_chan', '*.mat'));
    valid_idx = ~startsWith({all_files.name}, '._');
    valid_idx_pow = ~startsWith({pow_files.name}, '._');
    pow_files = pow_files(valid_idx_pow);
    all_files = all_files(valid_idx);

    is_lfp = contains({all_files.name}, {'_L', '_R'});
    eeg_itc_files = all_files(~is_lfp);
    lfp_itc_files = all_files(is_lfp);

    is_lfp_pow = contains({pow_files.name}, {'_L', '_R'});
    eeg_pow_files = pow_files(~is_lfp_pow);
    lfp_pow_files = pow_files(is_lfp_pow);

    [Itc_eeg, Power_eeg, Subs_eeg] = process_type(eeg_itc_files, eeg_pow_files, 'EEG', freq_idx, time_idx);
    [Itc_lfp, Power_lfp, Subs_lfp] = process_type(lfp_itc_files, lfp_pow_files, 'LFP', freq_idx, time_idx);

    [rval_eeg, pval_eeg, z_itc_eeg, z_pow_eeg] = analyze_itc_power_corr(Itc_eeg, Power_eeg, Subs_eeg, 'SavePath', '/Volumes/LP3/HeadHeart/0_data/itc/corr', 'SaveName', 'ITC-POW_PearsonCorr_EEG');
    [rval_lfp, pval_lfp, z_itc_lfp, z_pow_lfp] = analyze_itc_power_corr(Itc_lfp, Power_lfp, Subs_lfp, 'SavePath', '/Volumes/LP3/HeadHeart/0_data/itc/corr', 'SaveName', 'ITC-POW_PearsonCorr_LFP');


    % % === Z-score within each subject ===
    % unique_subs = unique(Subject_ids);
    % z_itc = zeros(size(Itc_vals));
    % z_pow = zeros(size(Power_vals));
    % 
    % for s = 1:numel(unique_subs)
    %     sub_idx = strcmp(Subject_ids, unique_subs{s});
    %     z_itc(sub_idx) = zscore(Itc_vals(sub_idx));
    %     z_pow(sub_idx) = zscore(Power_vals(sub_idx));
    % end
    % 
    % % === Pearson Correlation ===
    % [rval, pval] = corr(z_pow(:), z_itc(:), 'type', 'Pearson');
    % fprintf('Correlation (z-Power vs z-ITC): r = %.3f, p = %.8f\n', rval, pval);
    % 
    % % === Bayesian Correlation ===
    % %addpath('path/to/BayesFactor');  % Change to actual BayesFactor toolbox path
    % 
    % %BF = bf_corr(z_itc(:), z_pow(:));
    % %inv_BF = 1 / BF;
    % 
    % %fprintf('Bayes Factor (JZS): BF = %.3f, Inverse BF = %.3f\n', BF, inv_BF);
    % 
    % % === Scatter Plot ===
    % figure;
    % scatter(z_itc, z_pow, 60, 'filled', 'MarkerFaceAlpha', 0.6); hold on;
    % lsline;  % regression line
    % 
    % xlabel('Z-scored ITC');
    % ylabel('Z-scored Spectral Power');
    % title(sprintf('Correlation r = %.2f, p = %.4f', rval, pval));
    % grid on; box on;
    % set(gca, 'FontSize', 12);

end


function [Itc_vals, Power_vals, Subject_ids] = process_type(file_list, pow_list, type_label, freq_idx, time_idx)
    Itc_vals = [];
    Power_vals = [];
    Subject_ids = [];

    for i = 1:numel(file_list)
        fname = file_list(i).name;
        fpath = fullfile(file_list(i).folder, fname);
        load(fpath, 'ItcDataCh');
        
        % ITC
        mean_itc = mean(ItcDataCh(freq_idx, time_idx), 'all');
        Itc_vals(end+1) = mean_itc;

        parts = split(fname, '_');
        Subject_ids{end+1} = parts{1};  % assumes format: Subject_XX_L.mat etc.
    end

    for i = 1:numel(pow_list)
        fpath = fullfile(pow_list(i).folder, pow_list(i).name);
        load(fpath, 'POW');

        mean_power = mean(mean(POW(freq_idx, time_idx), 2), 1);
        Power_vals(end+1) = mean_power;
    end

    fprintf('Finished processing %s data.\n', type_label);
end

function [rval, pval, z_itc, z_pow] = analyze_itc_power_corr(Itc_vals, Power_vals, Subject_ids, varargin)
% ANALYZE_ITC_POWER_CORR Z-scores ITC and Power within subjects,
% calculates Pearson correlation, (optionally Bayesian), and plots results.
%
% Usage:
%   [r, p, z_itc, z_pow] = analyze_itc_power_corr(Itc_vals, Power_vals, Subject_ids);
%
% Optional:
%   'PlotTitle', 'My Title'
%   'DoBayes', true
%   'SavePath'  : folder to save figure (e.g., 'results/')
%   'SaveName'  : optional filename (e.g., 'lfp_corr_plot.png'). If omitted, uses title.
%
% Requires BayesFactor toolbox (https://github.com/klabhub/bayesFactor)

    % Parse optional inputs
    p = inputParser;
    addParameter(p, 'PlotTitle', 'Correlation: Z-scored ITC vs Power', @ischar);
    addParameter(p, 'DoBayes', false, @islogical);
    addParameter(p, 'SavePath', '', @ischar);
    addParameter(p, 'SaveName', '', @ischar);
    parse(p, varargin{:});
    plot_title = p.Results.PlotTitle;
    do_bayes = p.Results.DoBayes;
     save_path = p.Results.SavePath;
    save_name = p.Results.SaveName;
    

    % Initialize
    z_itc = zeros(size(Itc_vals));
    z_pow = zeros(size(Power_vals));
    unique_subs = unique(Subject_ids);

    % Z-score within each subject
    for s = 1:numel(unique_subs)
        sub_idx = strcmp(Subject_ids, unique_subs{s});
        if sum(sub_idx) >= 3
            z_itc(sub_idx) = zscore(Itc_vals(sub_idx));
            z_pow(sub_idx) = zscore(Power_vals(sub_idx));
        else
            z_itc(sub_idx) = NaN;
            z_pow(sub_idx) = NaN;
            warning('Only one data point for subject %s — skipping z-scoring.', unique_subs{s});
        end
    end

    % Remove NaNs
    valid_idx = ~isnan(z_itc) & ~isnan(z_pow);
    z_itc = z_itc(valid_idx);
    z_pow = z_pow(valid_idx);

    % Pearson correlation
    [rval, pval] = corr(z_itc(:), z_pow(:), 'type', 'Pearson');
    fprintf('Pearson Correlation: r = %.3f, p = %.6f\n', rval, pval);

    
    % Plot
    figure;
    scatter(z_itc, z_pow, 60, 'filled', 'MarkerFaceAlpha', 0.6); hold on;
    lsline;
    xlabel('Z-scored ITC');
    ylabel('Z-scored Spectral Power');
    title(sprintf('%s\nr = %.2f, p = %.4f', plot_title, rval, pval));
    grid on; box on;
    set(gca, 'FontSize', 12);

    % Save plot if requested
    if ~isempty(save_path)
        if isempty(save_name)
            % Sanitize plot title to generate filename
            fname = regexprep(plot_title, '[^\w\d]', '_');  % replace non-word chars
            fname = lower(fname);  % lowercase
            save_name = [fname '.png'];
        end

        full_save_path = fullfile(save_path, save_name);
        try
            exportgraphics(gcf, full_save_path, 'Resolution', 300);
            fprintf('Plot saved to: %s\n', full_save_path);
        catch ME
            warning('Failed to save plot: %s', ME.message);
        end
    end


end


function   [ChsAllFrsTmPha] = time_lock_to_surrogate(ChDta, surrogate_rpeaks, SR, tWidth, tOffset, NewSR, Frqs, times, onlyeeg)
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
if onlyeeg
    ChDta=ft_preproc_highpassfilter(ChDta,SR,0.5,4,'but', 'twopass'); % twopass
else
    ChDta=ft_preproc_highpassfilter(ChDta,SR,2,4,'but', 'twopass'); % twopass
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
