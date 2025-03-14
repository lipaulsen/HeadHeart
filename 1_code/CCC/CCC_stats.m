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
steps = {'Calc Single Subject CCC'}; %'Plot SubAvg PermStats', 'Calc Single Subject CCC', 'Plot SubAvg CCC', 'Plot Power'


CCCchans.Comp1 = {'STNl', 'C3'};
CCCchans.Comp2 = {'STNl', 'F3'};
CCCchans.Comp3 = {'STNr', 'C4'};
CCCchans.Comp4 = {'STNr', 'F4'};
CCCchans.Comp5 = {'F3', 'F4'};
CCCchans.Comp6 = {'STNl', 'C4'};
CCCchans.Comp7 = {'STNl', 'F4'};
CCCchans.Comp8 = {'STNr', 'C3'};
CCCchans.Comp9 = {'STNr', 'F4'};
comps = {'Comp1', 'Comp2', 'Comp3', 'Comp4', 'Comp5', 'Comp6', 'Comp7', 'Comp8', 'Comp9'};

Hz_dir = '2Hz';

% Define Time Window
tWidth   = 0.9;
tOffset  = 0.3;
FltPassDir = 'twopass';
NewSR = 300;


nSub = numel(subjects);

BandWidth=2; % BandWidth in Hz;
Qfac     =2; % Attenuation in db(-Qfac)
tCircMean=0.05; % for By TRials calc

permstats = false;
numPerms = 500;
surrogate = true;
trials = false;
plots = true;
CCC = [];
signif_thresh = 0.05;


if MedOn == true
    medname = 'MedOn';
elseif MedOff == true
    medname = 'MedOff';
end

% Use BPReref Data
BPReref = true; BPRerefTit = 'BPReref';
BPRerefHi = true; BPRerefHiTit = 'BPRerefHi';
BPRerefLw = false; BPRerefLwTit = 'BPRerefLow';
BPRerefBest = false; BPRerefBestTit = 'BPRerefBest';

% Flag if only EEG, STN or all channels
allchans = true;
onlyeeg = false;
onlystn = false;

disp("************* STARTING CCC Stats FUnction *************");

fprintf('Loading AVG ECG Data\n');
if strcmp(medname, 'MedOn')
    pattern = fullfile(data_dir, 'ecg', ['ECG-AVG_', medname, '_n=11_', '*']);
elseif strcmp(medname, 'MedOff')
    pattern = fullfile(data_dir, 'ecg', ['ECG-AVG_', medname, '_n=7_', '*']);
end
files = dir(pattern);
filename = fullfile(files(1).folder, files(1).name);
load(filename, 'AVGECG');

if ismember ('Calc Single Subject CCC', steps)

    % Initialize the Perm Matrix be sure here that the freqs and Time are
    % fitting because it does not work with the getting it out of the data with
    %out overwriting it
    PermCccAll = zeros(nSub, numel(channels), numPerms, 141, 271);
    % ZScoresAll = zeros(nSub, numel(channels), 141, 271);
    % PValAll = zeros(nSub, numel(channels), 141, 271);

    for sub = 1:numel(subjects) % BE AWARE THAT THIS EXCLUDES PATIENTS WITH ARRITHYMIAS

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
            channels = FltSubsOnlySTN{sub};
        end

        % KS29 has no EEG recordings in MedOn so we delete those values
        if strcmp(medname,'MedOn') & strcmp(subject,'KS29')
            channels = {FltSubsChansStn{sub}{end-1:end}};
        end

        %% GET PSI / CCC By Trials
        for c1 = 1:numel(fieldnames(CCCchans))
            comp = comps{c1};
            channel1 = CCCchans.(comp){1};
            channel2 = CCCchans.(comp){2};

            if strcmp(channel1, 'STNl'); chan1 = LfpElec.(subject){1}; elseif strcmp(channel1, 'STNr'); chan1 = LfpElec.(subject){2}; else; chan1 = channel1; end
            if strcmp(channel2, 'STNl'); chan2 = LfpElec.(subject){1}; elseif strcmp(channel2, 'STNr'); chan2 = LfpElec.(subject){2}; else; chan2 = channel2; end


            fprintf('************ Calculating CCC for %s in %s **************** \n', subject, channel);
            % Calc original CCC
            [FrsTmPsiTrial,FrsTmPhaTrial]=Get_PSI_ByTrials(TFR.(chan1).phase,TFR.(chan2).phase,SR,tCircMean);

            % Scale the CCC to the relative CCC of the channel
            meanFrsTmCcc = mean(mean(FrsTmPsiTrial,1),2);
            relFrsTmCcc = FrsTmPsiTrial/meanFrsTmCcc;

            CccAll(sub,c1,:,:) = FrsTmPsiTrial; % SubjectxChannelxFreqxTime
            RelCccAll(sub,c1,:,:) = relFrsTmCcc;
        end

        %% PLOT SingleSub PSI/CCC
        outputPDF1 = fullfile('F:\HeadHeart\2_results\ccc\ss' , [subject, '_CCC_Allchancomp_med=', subfname, ...
            '_win=-', num2str(tOffset),'to', num2str(tWidth-tOffset),'_BSL=', num2str(baseline), '.pdf']);
        for c1 = 1:numel(fieldnames(CCCchans))
            comp = comps{c1};
            channel1 = CCCchans.(comp){1};
            channel2 = CCCchans.(comp){2};

            if plots
                f1=figure;
                set(f1,'Position',[1949 123 1023 785]);
                subplot(2,1,1)
                plot(times, mean(EvEcgData,1), 'Color', 'k'); hold on
                set(gca,'Position',[0.1300 0.5838 0.71 0.3])
                xline(0, "--k", 'LineWidth', 2);
                ylabel('Amplitude')
                axis('tight')
                title(sprintf('Average ECG for %s in %s, med= %s', subject, channel, subfname))
                hold off
                subplot(2,1,2)
                imagesc(times,freqs,squeeze(CccAll(sub,c1,:,:)));axis xy;
                colormap('parula');
                xline(0, "--k", 'LineWidth', 2);
                col = colorbar;
                col.Label.String = 'CCC Values'; % Add title to colorbar
                xlabel('Time (s)') % Add x-label
                ylabel('Frequencies (Hz)') % Add y-label
                title(sprintf('CCC PSI for %s, %s - %s, med= %s', subject, channel1, channel2, subfname))


                %gr1 = fullfile('F:\HeadHeart\2_results\ccc\ss' , [subject, '_', subfname, '_ITC_', channel, '.png']);
                %exportgraphics(f1,gr1, 'Resolution', 300)
                exportgraphics(f1, outputPDF1, 'Append', true);
            end
        end
        %% ==================== PERMUTATION ===========================
        if permstats
            outputPDF = fullfile('F:\HeadHeart\2_results\ccc\ss_perm' , [subject, '_ITC-PermStats_Allchan_med=', subfname, '_perm=', num2str(numPerms), ...
                '_win=-', num2str(tOffset),'to', num2str(tWidth-tOffset),'_BSL=', num2str(baseline), '_pval=', num2str(signif_thresh),'.pdf']);
            for c = 1:numel(channels)


                channel = channels{c};
                %[ITCzscores]=ITC_permutation_test(FrsTmItc, relFrsTmItcAll, IBI.(subfname){1}, numPerms, freqs, time_bins, SR, TFR.(channel).phase);
                fprintf('************ Calculating Perm Stats for %s in %s **************** \n', subject, channel);

                % Intitialize variables
                %permuted_ITCs = zeros([numPerms, size(FrsTmItc, 1), size(FrsTmItc, 2)]);
                [nTrials, nFreq, nTms] = size(TFR.(channel).phase);

                % Get the raw channel data
                ChDta = SmrData.WvDataCleaned(c, :);
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

                    parfor perm = 1:numPerms
                        % Time Lock the surrogate R Peaks to the Channel
                        % Data and apply the filters as well as the DS and
                        % create TFR for the new epochs
                        currSurrogateRpeaks = surrogate_rpeaks(perm, :);
                        [ChsAllFrsTmPha] = time_lock_to_surrogate(ChDta, currSurrogateRpeaks, oldSR, tWidth, tOffset, numPerms, NewSR, freqs, times);

                        %  Calculate CCC with the surrogate R-peaks (one per channel)
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

                zscores = (squeeze(CccAll(sub,c,:,:)) - diff_sum_perm_mean) ./ diff_sum_perm_std ;
                %zscores_perm = (PermItcData - diffPerm_mean) ./ diffPerm_std;
                p_orig = 2 * (1 - normcdf(zscores, 0, 1));
                p_thresh = p_orig < signif_thresh;

                % ZScoresAll(sub,c,:,:) = zscores; % SubjectxChannelxFreqxTime (Last two are CCC ZScores)
                % PValAll(sub,c,:,:) = p_orig;  % SubjectxChannelxFreqxTime (Last two are CCC PVals)
                PermCccAll(sub,c,:,:,:) = PermItcData; % SubjectxChannelxPermutationxFreqxTime


                % f3 = figure; % Sanity check that the distributions are normalized and overlapping so that my null hypothesis actually reflects my data
                % histogram(PermItcData);
                % hold on
                % histogram(squeeze(CccAll(sub,c,:,:)));
                % %title(sprintf('Perm CCC and Original CCC Dist for %s in %s, perms: %d, med: %s', subject, channel, numPerms, subfname))
                %
                % f4 = figure;
                % subplot(2,1,1);
                % histogram(zscores);
                % %title(sprintf('Z Score Dist for %s in %s, perms: %d, med: %s', subject, channel, numPerms, subfname))
                % subplot(2,1,2);
                % histogram(p_orig);
                % %title(sprintf('P Val Dist for %s in %s, perms: %d, med: %s', subject, channel, numPerms, subfname))

                %itc_zscores_thresh = (zscores > 2) | (zscores < -2);

                f5 = figure;
                set(f5,'Position',[1949 123 1023 785]);

                % Upper subplot
                subplot(2,1,1)
                plot(times, mean(EvEcgData,1), 'Color', 'k'); hold on
                set(gca,'Position',[0.1300 0.5838 0.71 0.3])
                xline(0, "--k", 'LineWidth', 2);
                axis('tight')
                title(sprintf('Average ECG for %s, med: %s', subject, subfname))
                ylabel('Amplitude')
                hold off

                % Lower subplot
                subplot(2,1,2)
                imagesc(times, freqs, squeeze(CccAll(sub,c,:,:))); axis xy;
                colormap('parula');
                col = colorbar;
                col.Label.String = 'CCC Values'; % Add title to colorbar
                clims = clim;
                hold on;
                contour(times, freqs, p_thresh, 1, 'linecolor', 'k', 'linewidth', 1.5);
                xline(0, "--k", 'LineWidth', 2);
                clim(clims);
                title(sprintf('CCC for %s in %s, perms: %d, med: %s, p<%.4g', subject, channel, numPerms, subfname, signif_thresh))
                xlabel('Time (s)') % Add x-label
                ylabel('Frequencies (Hz)') % Add y-label
                hold off

                % outputPDF = fullfile('F:\HeadHeart\2_results\ccc' , [subject, '_ITC-PermStats_chan=', channel, '_med=', subfname, '_perm=', num2str(numPerms), ...
                %     '_win=-', num2str(tOffset),'to', num2str(tWidth-tOffset),'_BSL=', num2str(baseline), '_pval=', num2str(signif_thresh),'.pdf']);
                % exportgraphics(f5,gr5, 'Resolution', 300)
                exportgraphics(f5, outputPDF, 'Append', true);

                % % Define the output PDF file name
                % outputPDF = fullfile('F:\HeadHeart\2_results\ccc\ss_perm' , [subject, '_ITC-PermStats_chan=', channel, '_med=', subfname, '_perm=', num2str(numPerms), '.pdf']);
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
        CCC.PERM.ZScoresAll = ZScoresAll;  % SubjectxChannelxFreqxTime (Last two are CCC ZScores)
        CCC.PERM.PValAll = PValAll; % SubjectxChannelxFreqxTime (Last two are CCC PVals)
        CCC.PERM.PermCccAll = PermCccAll; %SubjectxChannelxPermutationxFreqxTime
    end
    % Save CCC MAtrix of all Subjects and all Channels
    CCC.SR = SR;
    CCC.times = times;
    CCC.freqs = freqs;
    CCC.CccAll = CccAll;
    CCC.RelCccAll = RelCccAll;
    if permstats
        save_path = fullfile(data_dir, 'ccc', ['CCC-AllSubs_', subfname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', freqs(1) ,'.mat']);
    else
        save_path = fullfile(data_dir, 'ccc', ['CCC-AllSubs_', subfname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), 'HPF=', num2str(freqs(1)),'_HP=', freqs(1) ,'.mat']);
    end
    save(save_path, 'CCC', '-v7.3');
    fprintf('Saved CCC Data for all subs and channels to: %s\n', save_path);
end


% PLOT THE AVERAGE CCC PER SUBJECT
if ismember('Plot SubAvg CCC', steps)
    fprintf('Plot SubAvg CCC\n');
    if ismember('Calc Single Subject CCC', steps) == false

        fprintf('Loading TFR Data\n');
        pattern = fullfile(data_dir, 'CCC', ['CCC-AllSubs_',  subfname, '*']);
        files = dir(pattern);
        filename = fullfile(files(1).folder, files(1).name);
        load(filename, 'CCC', '-mat');
        SR= CCC.SR;
        freqs = CCC.freqs;

    end
    plots = true;
    for c = 1:numel(channels)
        channels = {'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'Pz', 'STNl', 'STNr'};
        channel = channels{c};

        ItcAll_subavg = squeeze(mean(squeeze(CCC.CccAll(:,c,:,:)),1));
        RelItcAll_subavg = squeeze(mean(squeeze(CCC.RelCccAll(:,c,:,:)),1));

        if plots
            f2=figure;
            set(f2,'Position',[1949 123 1023 785]);
            subplot(2,1,1)
            plot(times, AVGECG.mean', 'Color', 'k'); hold on
            set(gca,'Position',[0.1300 0.5838 0.77 0.3])
            xline(0, "--k", 'LineWidth', 2);
            title(sprintf('Average ECG over all subjects, medication: %s', subfname))
            hold off
            subplot(2,1,2)
            imagesc(times,freqs,ItcAll_subavg);axis xy;
            colormap('jet');
            xline(0, "--k", 'LineWidth', 2);
            colorbar;
            title(sprintf('Average CCC for %s, med: %s', channel, subfname))

            gr2 = fullfile('F:\HeadHeart\2_results\ccc' , ['ITC_', channel, '_', subfname,  '.png']);
            exportgraphics(f2,gr2, 'Resolution', 300)
        end
    end
end

% Now here is the Grand Average CCC over all Subs per Channel
if ismember('Plot SubAvg PermStats', steps)
    fprintf('Plot SubAvg CCC PermStats\n');

    if ismember('Calc Single Subject CCC', steps) == false
        fprintf('Loading CCC Data\n');
        pattern = fullfile(data_dir, 'CCC', ['CCC-AllSubs_',  subfname, '*', '_perm=', '*']);
        files = dir(pattern);
        filename = fullfile(files(1).folder, files(1).name);
        load(filename, 'CCC', '-mat');
        SR= CCC.SR;
        freqs = CCC.freqs;
        times = CCC.times;
    end


    for c = 1: numel(channels)
        channel = channels{c};

        %ChanMeanZscores_avg = squeeze(mean(squeeze(CCC.PERM.ZScoresAll(:,c,:,:)),1));  % SubjectxChannelxFreqxTime (Last two are CCC ZScores)
        %ChanMeanPVal_avg = squeeze(mean(squeeze(CCC.PERM.PValAll(:,c,:,:)),1));  % SubjectxChannelxFreqxTime (Last two are CCC PVals)
        PermItcAll_avg = squeeze(mean(squeeze(CCC.PERM.PermCccAll(:,c,:,:,:)),1)); %SubjectxChannelxPermutationxFreqxTime, Mean over all Subjects in one channel
        ItcAll_subavg = squeeze(mean(squeeze(CCC.CccAll(:,c,:,:)),1));
        RelItcAll_subavg = squeeze(mean(squeeze(CCC.RelCccAll(:,c,:,:)),1));

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
        title(sprintf('Average ECG over all subjects, med: %s', subfname))
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
        title(sprintf(' Average CCC in %s, perm = %d, med = %s, p<%.4g', channel, numPerms, subfname, signif_thresh))

        gr6 = fullfile('F:\HeadHeart\2_results\ccc\group_perm' , ['AvgITC_', channel, '_', subfname, '_perm=', num2str(numPerms), '.png']);
        exportgraphics(f6,gr6, 'Resolution', 300)



    end
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