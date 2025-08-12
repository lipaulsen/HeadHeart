% function [] = erp_calc_stats(subjects, data_dir, results_dir)

%% Calculating and Plotting Event Related Potential

% Author: Lisa Paulsen
% Contact: lisaspaulsen[at]web.de
% Created on: 1 October 2024
% Last update: 5 Feburary 2025

%% INPUTS AND OUTPUTS

% Extract the Event Related Potentail (ERP) fromt he Preprocessed Data and
% plot the ERP
%
% Inputs:
% Preprocessed data (EEG, LFP, ECG) from .mat file
%
% Outputs:    Plots of the ERP in the results erp folder
%
% - ECG: IBI(sub, :), HRV, HF-HRV, LF-HRV
% - EEG & LFP: Power of delta, theta, alpha, beta, gamma bands for all electrodes

% Steps:
% 1. LOAD DATA
% 2. EPOCH and TIMELOCK DATA
% 3. SAVE DATA

%% ============= SET GLOABAL VARIABLES AND PATHS =========================

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

preprocessed_name = 'preprocessed';
epoch_name = 'epoch';

% Define if plots are to be shown
show_plots = false;

nSub = numel(subjects);
numPerms = 500;

NewSR=300;

% Filter Parameters
Fhp = 2;
Flp = 30;
FltPassDir='twopass'; % onepass  twopass

steps = {'Plot Mean Clustering by Med', 'Plot Mean Clustering by Method'}; % 'Plot Mean Clustering by Med', ERP Group Cluster, ERP Group, 'Plot SS ERP', 'ERP stats', 'ERP SS Calc', ,'Clustering'

% Define Time Window
tWidth   = 0.9;
tOffset  = 0.3;

% Baseline parameters
baseline = true;
baseline_win = [-0.3 -0.1];

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
allchans = false;
onlyeeg = false;
onlystn = true;
flags = [allchans, onlyeeg, onlystn];
names = {'allchans', 'onlyeeg', 'onlystn'};
channame = names{flags};

fprintf('Loading AVG ECG Data\n');
if strcmp(medname, 'MedOn')
    pattern = fullfile(data_dir, 'ecg', ['ECG-AVG_', medname, '_n=11_', '*']);
elseif strcmp(medname, 'MedOff')
    pattern = fullfile(data_dir, 'ecg', ['ECG-AVG_', medname, '_n=7_', '*']);
end
files = dir(pattern);
filename = fullfile(files(1).folder, files(1).name);
load(filename, 'AVGECG');

max_chan = max(cellfun(@numel, FltSubsChansStn));
%EvDataAllTrsPerm = zeros(numel(subjects), max_chan, numPerms, 350, 271);
EvDataAllAvgTrsPerm = zeros(numel(subjects), max_chan, numPerms, 271);

if ismember('ERP SS Calc', steps)
    for sub = 1:numel(subjects) % BE AWARE THAT THIS EXCLUDES PATIENTS WITH ARRITHYMIAS
        % Extract the subject
        subject = subjects{sub};

        fprintf('Loading Data of subject %s number %i of %i\n', subject, sub, numel(subjects));

        pattern = fullfile(data_dir, 'preproc', 'all', [subject, '_', preprocessed_name, '_', medname, '_BPReref_', '*']);
        files = dir(pattern);
        filename = fullfile(files(1).folder, files(1).name);
        load(filename, 'SmrData');
        % Load subject data
        % subject_data = fullfile(data_dir, preprocessed_name, medname, ['sub-', subject], [subject, '_preprocessed_', medname, '_Rest.mat']);
        % load(subject_data, 'SmrData');

        % pattern = fullfile(data_dir, 'ecg', 'ss' ,[subject, '_EpochECGEvData_', medname, '*']);
        % files = dir(pattern);
        % filename = fullfile(files(1).folder, files(1).name);
        % load(filename, 'EvECG');

        % Load the the cleaned ECG R Peaks Data
        fprintf('Loading ECG Data\n');
        pattern = fullfile(data_dir, 'ecg', 'ss' ,[subject, '_EpochECGEvData_', medname, '*']);
        files = dir(pattern);
        filename = fullfile(files(1).folder, files(1).name);
        load(filename, 'EvECG');

        SR = SmrData.SR;
        EventTms = SmrData.EvData.EvECGP_Cl;

        if allchans
            channels = FltSubsChansStn{sub};
        elseif onlyeeg
            channels = FltSubsOnlyEEG{sub};
        elseif onlystn
            channels = FltSubsOnlyStn{sub};
        end

        %channels = FltSubsChansStn{sub};

        for el = 1:numel(channels)
            chanidx(el,:) = find(strcmp(FltSubsChansRaw{sub},channels{el}));
        end

        for c = 1:numel(channels)

            if ~BPReref
                channel = FltSubsChansRaw{sub}{chanidx(c)};
            else
                channel = channels{c};
            end

            %% ==================== EPOCH DATA ==========================
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
                ChDta=ft_preproc_highpassfilter(ChDta,SR,Fhp,4,'but',FltPassDir); % twopass
            end
            if Flp > 0
                ChDta = ft_preproc_lowpassfilter(ChDta, SR, Flp, 4, 'but',FltPassDir);
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

            EvDataAllAvgTrs(sub,c,:) = squeeze(mean(EvData,1)); % sub, chan, Average over Epochs
            if ismember('ERP stats', steps)
                EvDataAll(sub,c,:,:) = EvData;
            end

            %% Z Score the Ev Data
            % As the basleine has already been taken the data is alreadu
            % around 0 but I want it to have the same std as well so I will
            % do that here

            BslStd = std(EvData(:,bidx(1):bidx(end)), 0, 2);
            BslStd(BslStd == 0) = eps; % Replace zero std with a small value

            % Normalize ERPData
            EvData_zscored = EvData ./ BslStd;

            EvDataAllAvgTrsZScore(sub,c,:) = squeeze(mean(EvData_zscored,1));



            %% ================= PERMUATATION STATISTICS ================
            if ismember('ERP stats', steps)

                fprintf('************ Calculating Perm Stats for %s in %s **************** \n', subject, channel);

                % prep for the surrogate data
                chan_idx = find(strcmp(channels, channel)); % Find index
                % Get the raw channel data
                ChDta = SmrData.WvDataCleaned(chan_idx, :);
                oldSR = SmrData.SR;

                % % Pre-generate surrogate R-peaks outside of the parfor loops
                % surrogate_rpeaks = zeros(numPerms, length(EventTms));
                % for p = 1:numPerms
                %     surrogate_rpeaks(p, :) = EventTms + (rand(1, length(EventTms)) - 0.15);
                % end

                % Define safe jitter range (in seconds)
                jitter_range = 0.15;
                % Convert margins from your time window to seconds
                buffer_start = abs(tOffset);
                buffer_end = tWidth - abs(tOffset);
                % Define signal duration in seconds
                signal_dur_sec = length(ChDta) / oldSR;
                % Calculate valid min and max jittered R-peak time in seconds
                min_time = buffer_start;
                max_time = signal_dur_sec - buffer_end;
                % Pre-allocate
                surrogate_rpeaks = zeros(numPerms, length(EventTms));

                for p = 1:numPerms
                    jitter = (rand(1, length(EventTms)) - 0.2) * 2 * jitter_range;
                    temp = EventTms + jitter;
                    % Clip to keep within valid time bounds
                    temp(temp < min_time) = min_time;
                    temp(temp > max_time) = max_time;
                    surrogate_rpeaks(p, :) = temp;
                end


                startTime = datetime('now');
                disp(['Start Time: ', datestr(startTime)]);

                for pe = 1:numPerms % parfor
                    currSurrogateRpeaks = surrogate_rpeaks(pe, :);
                    [EvDataPerm] = time_lock_to_surrogate(ChDta, currSurrogateRpeaks, oldSR, tWidth, tOffset, NewSR,  Fhp, Flp, baseline_win, FltPassDir);

                    %EvDataAllAvgTrsPerm(sub,c,pe,:) = squeeze(mean(EvDataPerm,1));
                    %EvDataAllTrsPerm(sub,c, pe ,:,:) = EvDataPerm;
                    EvDataAllTrsPerm(pe ,:,:) = EvDataPerm;
                    fprintf('perm = %d \n', pe)
                end

                endTime = datetime('now'); disp(['End Time: ', datestr(endTime)]);
                % Calculate elapsed time
                elapsedTime = endTime - startTime; disp(['Elapsed Time: ', char(elapsedTime)]);

                % mean over perm
                EvDataAllTrsAvgPerm = squeeze(mean(EvDataAllTrsPerm,1));
                EvDataAllSQ = squeeze(EvDataAll(sub, c, : ,:));

                % Step 2: Compute observed t-values (real vs surrogate) at each timepoint
                tvals = zeros(1, length(TmAxis));
                for t = 1:length(TmAxis)
                    [~,~,~,stats] = ttest2(EvDataAllSQ(:,t), EvDataAllTrsAvgPerm(:,t));
                    tvals(t) = stats.tstat;
                end

                % Step 3: Find clusters above threshold (e.g., |t| > 2, corresponding to p < 0.05)
                alpha = 0.05;
                threshold = tinv(1 - alpha, 349); % adjust for your sample size
                cluster_labels = bwlabel(abs(tvals) > threshold);
                obs_clusters = regionprops(cluster_labels, abs(tvals), 'Area', 'PixelIdxList', 'MeanIntensity');

                % Compute cluster statistics (sum of t-values within each cluster)
                obs_cluster_stats = arrayfun(@(c) sum(abs(tvals(c.PixelIdxList))), obs_clusters);
                %obs_cluster_stats = [arrayfun(@(c) sum(abs(tvals(c.PixelIdxList))), obs_clusters)];


                % Step 4: Permutation test
                allHEP = cat(1, squeeze(EvDataAll(sub, c, : ,:)), EvDataAllTrsAvgPerm(:,:)); % [2*n Trials x timepoints]
                labels = [ones(nEvs,1); zeros(nEvs,1)];
                max_perm_cluster_stats = zeros(numPerms,1);

                for p = 1:numPerms
                    perm_labels = labels(randperm(length(labels)));
                    perm_real = allHEP(perm_labels==1,:);
                    perm_surr = allHEP(perm_labels==0,:);
                    perm_tvals = zeros(1, length(TmAxis));
                    for t = 1:length(TmAxis)
                        [~,~,~,stats] = ttest2(perm_real(:,t), perm_surr(:,t));
                        perm_tvals(t) = stats.tstat;
                    end
                    % Find clusters in permuted data
                    perm_cluster_labels = bwlabel(abs(perm_tvals) > threshold);
                    perm_clusters = regionprops(perm_cluster_labels, abs(perm_tvals), 'Area', 'PixelIdxList', 'MeanIntensity');
                    if ~isempty(perm_clusters)
                        perm_cluster_stats = arrayfun(@(c) sum(abs(perm_tvals(c.PixelIdxList))), perm_clusters);
                        max_perm_cluster_stats(p) = max(perm_cluster_stats);
                    else
                        max_perm_cluster_stats(p) = 0;
                    end
                end

                % Step 5: Determine significance of observed clusters
                pvals = arrayfun(@(c) mean(max_perm_cluster_stats >= c), obs_cluster_stats);

                % Output significant clusters and their time indices
                sig_clusters = find(pvals < 0.05);
                for i = 1:length(sig_clusters)
                    idx = obs_clusters(sig_clusters(i)).PixelIdxList;
                    fprintf('Significant cluster at timepoints: %s\n', mat2str(idx));
                end

                outputPDF1 = fullfile(results_dir, 'ss_chan', [subject, '_', channel, '_ERP_med=', medname, ...
                    '_win=-', num2str(tOffset),'to', num2str(tWidth-tOffset),'_BSL=', num2str(baseline), '.pdf']);

                % Optional: Plot
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
                plot(mean(realHEP), 'b', 'LineWidth', 2); hold on;
                plot(mean(surrogateHEP), 'r', 'LineWidth', 2);
                for i = 1:length(sig_clusters)
                    idx = obs_clusters(sig_clusters(i)).PixelIdxList;
                    area(idx, mean(realHEP(:,idx)), 'FaceColor', [0.8 0.8 0.8], 'EdgeColor', 'none');
                end
                legend('Real HEP','Surrogate HEP','Significant cluster');
                xlabel('Time (samples)');
                ylabel('Amplitude (\muV)');
                title('HEP Permutation Cluster Test');

                exportgraphics(f1, outputPDF1, 'Append', true);
            end
        end
        %% ============== Plotting ERPs SUBJECT LEVEL=========================
        if show_plots
            fprintf('Plotting ERPs for subject %s\n', subject);
            f1 = figure; % initialize Figure
            set(f1, 'Position', [100, 100, 1920, 1080]);
            for chan = 1:numel(channels)
                channel = channels{chan};

                row = ceil(chan / 3); % Calculate the row number
                col = mod(chan - 1, 3) + 1; % Calculate the column number
                subplot(3, 3, (row - 1) * 3 + col)

                % Plot the ERP per channel for 1 subj
                plot(TmAxis(31:end), squeeze(EvDataAllAvgTrsZScore(sub, chan, 31:end))', 'Color', 'k'); hold on
                xline(0, "--k", 'LineWidth', 2);
                title(sprintf('ERP in %s', channel))
                axis("tight");
                hold off
                % Set Labels
                xlabel('Time (ms)');
                ylabel('Amplitude (uV)');
            end

            if BPReref & BPRerefHi
                sgtitle(sprintf('ERPs for Subject %s - All Channels with %s, %s', subject, medname, BPRerefHiTit)); % Major Title
                gr1 = fullfile(results_dir, 'erp', 'ss', [ subject, '_ERP_sep-channels_', medname, '_', BPRerefHiTit ,'_EP=',num2str(TmAxis(1)), 'to', num2str(TmAxis(end)),'s_DS=', num2str(NewSR),'_HP=', num2str(Fhp), '_BSL=', num2str(baseline_win(1)),'to', num2str(baseline_win(2)),'s.png']);
            elseif BPReref & BPRerefLw
                sgtitle(sprintf('ERPs for Subject %s - All Channels with %s, %s', subject, medname, BPRerefLwTit)); % Major Title
                gr1 = fullfile(results_dir, 'erp', 'ss', [ subject, '_ERP_sep-channels_', medname, '_', BPRerefLwTit ,'_EP=',num2str(TmAxis(1)), 'to', num2str(TmAxis(end)),'s_DS=', num2str(NewSR),'_HP=', num2str(Fhp), '_BSL=', num2str(baseline_win(1)),'to', num2str(baseline_win(2)),'s.png']);
            elseif  BPReref & BPRerefBest
                sgtitle(sprintf('ERPs for Subject %s - All Channels with %s, %s', subject, medname, BPRerefBestTit)); % Major Title
                gr1 = fullfile(results_dir, 'erp', 'ss', [ subject, '_ERP_sep-channels_', medname, '_', BPRerefBestTit ,'_EP=',num2str(TmAxis(1)), 'to', num2str(TmAxis(end)),'s_DS=', num2str(NewSR),'_HP=', num2str(Fhp), '_BSL=', num2str(baseline_win(1)),'to', num2str(baseline_win(2)),'s.png']);
            else
                sgtitle(sprintf('ERPs for Subject %s - All Channels with %s', subject, medname)); % Major Title
                gr1 = fullfile(results_dir, 'erp', 'ss', [ subject, '_ERP_sep-channels_', medname, '_EP=',num2str(TmAxis(1)), 'to', num2str(TmAxis(end)),'s_DS=', num2str(NewSR),'_HP=', num2str(Fhp), '_BSL=', num2str(baseline_win(1)),'to', num2str(baseline_win(2)),'s.png']);
            end
            exportgraphics(f1, gr1, "Resolution",300);

        end
    end

    % Save the Group level ERP Data

    if ismember('ERP stats', steps)
        ERP.Perm = EvDataAllTrsAvgPerm;
    end
    ERP.EvDataAllAvgTrs = EvDataAllAvgTrs;
    ERP.EvDataAllAvgTrsZScore = EvDataAllAvgTrsZScore;


    if allsubs & onlyeeg
        save_path = fullfile(data_dir, 'erp', 'group', ['ERP_Group-Data_EEG_', medname , '_Subs=', num2str(numel(subjects)), '_EP=',num2str(TmAxis(1)), 'to', num2str(TmAxis(end)),'s_DS=', num2str(SR),'_HP=', num2str(Fhp) ,'_BSL=', num2str(baseline_win(1)),'to', num2str(baseline_win(2)),'s.mat']);
    elseif allsubs & onlystn
        save_path = fullfile(data_dir, 'erp', 'group', ['ERP_Group-Data_STN_', medname , '_Subs=', num2str(numel(subjects)), '_EP=',num2str(TmAxis(1)), 'to', num2str(TmAxis(end)),'s_DS=', num2str(SR),'_HP=', num2str(Fhp) ,'_BSL=', num2str(baseline_win(1)),'to', num2str(baseline_win(2)),'s.mat']);
    else allsubs
        save_path = fullfile(data_dir, 'erp', 'group', ['ERP_Group-Data_ALL_', medname , '_Subs=', num2str(numel(subjects)), '_EP=',num2str(TmAxis(1)), 'to', num2str(TmAxis(end)),'s_DS=', num2str(SR),'_HP=', num2str(Fhp) ,'_BSL=', num2str(baseline_win(1)),'to', num2str(baseline_win(2)),'s.mat']);
    end
    save(save_path, 'ERP', '-v7.3');
    fprintf('Saved ERP Group Data to: %s\n', save_path);

    clear EvDataAllAvgTrs EvDataAllAvgTrsZScore 

end

%% ================== HIERARCHICAL CLUSTERING ======================

if ismember('Clustering', steps)

    % if strcmp(channame, 'allsubs')
    %     chantag = 'ALL';
    % elseif strcmp(channame, 'onlyeeg')
    %     chantag = 'EEG';
    % else strcmp(channame, 'onlystn')
    %     chantag = 'STN';
    % end
    chantags = {'EEG', 'STN', 'ALL'};

    fprintf('Loading ERP Group Data \n');
    pattern = fullfile(data_dir, 'erp', 'group', ['ERP_Group-Data_', '*']);
    files = dir(pattern);
    for i = 1:length(files)
        filename = fullfile(files(i).folder, files(i).name);
        fname = files(i).name;
        parts = split(fname, '_');
        chantag = parts{3};
        medname = parts{4};
        load(filename, 'ERP');
        % Save in new struct 
        CLUS.(medname).(chantag).EvDataAllAvgTrs = ERP.EvDataAllAvgTrs;
    end

    for s = 3%:length(chantags)
        chantag = chantags{s};
%% =========================== Med On ===============================
        subjects = string({subject_info([subject_info.MedOn] == 1).ID});
        switch chantag
            case 'EEG'
                channels = AllSubsOnlyEEG([subject_info.MedOn] == 1);
            case 'STN'
                channels = AllSubsOnlyStn([subject_info.MedOn] == 1);
            case 'ALL'
                channels = AllSubsChansStn([subject_info.MedOn] == 1);
        end

        Subject = {};
        Condition   = {};
        Channel = {};

        for chan = 1:max(cellfun(@numel, channels))   % max number of channels across subjects
            for subj = 1:numel(subjects)
                chansForThisSubj = channels{subj};
                if chan <= numel(chansForThisSubj)    % only add if subject has this channel
                    Subject{end+1,1}   = subjects{subj};
                    Condition{end+1,1} = {'MedOn'};
                    Channel{end+1,1}   = chansForThisSubj{chan};
                end
            end
        end
        clusterTable = table(Subject, Condition, Channel);
        if strcmp(chantag, 'EEG')
            KS29_EEG_rm = find(strcmp(clusterTable.Subject, 'KS29'));
            clusterTable(KS29_EEG_rm, :) = [];
        elseif strcmp(chantag, 'ALL')
            KS29_EEG_rm = find(strcmp(clusterTable.Subject, 'KS29'));
            clusterTable(KS29_EEG_rm(1:end-2), :) = [];
        elseif strcmp(chantag, 'STN')
            KS29_EEG_rm = find(strcmp(clusterTable.Subject, 'KS29'));
            clusterTable(KS29_EEG_rm, :) = [];
        end

        nClusters =6; % EEG = 5, STN = 4, ALL = 6
        flipped_clusters = [1,2,3];  % EEG bei 5 Clus = 5, STN 3 bei Clus = 4, ALL 1,2,3 bei Clus = 6
        [IdxTable, clusterTable, reshaped_on] = plot_clustered_HEPs(CLUS, 'on', chantag, nClusters, flipped_clusters, AVGECG, clusterTable) % For Med On only 
        Cluster.MedOn = reshaped_on;
        clusterTable.Cluster = IdxTable.Cluster;
        clusterTable.Flipped = IdxTable.Flipped;
        Cluster.Map_Med_On = clusterTable;

        %% ===================== Med Off ==============================0
        subjects = string({subject_info([subject_info.MedOff] == 1).ID});
        switch chantag
            case 'EEG'
                channels = AllSubsOnlyEEG([subject_info.MedOff] == 1);
            case 'STN'
                channels = AllSubsOnlyStn([subject_info.MedOff] == 1);
            case 'ALL'
                channels = AllSubsChansStn([subject_info.MedOff] == 1);
        end

        Subject = {};
        Condition   = {};
        Channel = {};
        % create subject x condition x channel map Med On
        for chan = 1:max(cellfun(@numel, channels))   % max number of channels across subjects
            for subj = 1:numel(subjects)
                chansForThisSubj = channels{subj};
                if chan <= numel(chansForThisSubj)    % only add if subject has this channel
                    Subject{end+1,1}   = subjects{subj};
                    Condition{end+1,1} = {'MedOff'};
                    Channel{end+1,1}   = chansForThisSubj{chan};
                end
            end
        end
        clusterTable = table(Subject, Condition, Channel);
       
        nClusters = 5; % EEG = 5, STN = 5, All = 5 
        flipped_clusters = [5];  % EEG nix, STN nix, All = 5 bei clos =5
        [IdxTable, clusterTable, reshaped_off] = plot_clustered_HEPs(CLUS, 'off', chantag, nClusters, flipped_clusters, AVGECG, clusterTable)    % For Med Off only
        Cluster.MedOff = reshaped_off;
        clusterTable.Cluster = IdxTable.Cluster;
        clusterTable.Flipped = IdxTable.Flipped;
        Cluster.Map_Med_Off = clusterTable;

%% ============================= Save ===================================
        save(fullfile(data_dir, 'erp', 'group', 'clustering', ['Cluster_Data_Flipped_', chantag,'_MedOn_MedOff.mat']), 'Cluster', '-v7.3')
        clear Cluster
        % nClusters = 6;
        % flipped_clusters = [];  % adjust as needed
        % plot_clustered_HEPs(CLUS, 'both', chantag, nClusters, flipped_clusters)   % For side-by-side comparison
    end

end

if ismember('Plot Mean Clustering by Med', steps)

    chantags = {'EEG', 'STN', 'ALL'};

    for s = 1:length(chantags)
        chantag = chantags{s};

        fprintf('Loading %s Cluster Data \n' ,chantag);
        pattern = fullfile(data_dir, 'erp', 'group', 'clustering', ['Cluster_Data_Flipped_', chantag, '*']);
        files = dir(pattern);
        filename = fullfile(files(1).folder, files(1).name);
        load(filename, 'Cluster');

        % Filter correct field
        fields = fieldnames(Cluster);
        field_on = fields{contains(fields, 'MedOn')};
        field_off = fields{contains(fields, 'MedOff')};
        
        % Mean + SEM 
        cluster_on_mean = mean(Cluster.(field_on),1);
        cluster_on_sem = std(Cluster.(field_on)) / sqrt(length(Cluster.(field_on)));

        cluster_off_mean = mean(Cluster.(field_off),1);
        cluster_off_sem = std(Cluster.(field_off),1) / sqrt(length(Cluster.(field_off)));


        % Create one figure for all clusters in current condition
        f2 = figure; hold on;
        set(f2,'Position',[1 59 1440 738]);
        subplot(2,1,1)
        plot(AVGECG.times(31:end), mean(AVGECG.mean(31:end),1), 'Color', 'k'); hold on
        set(gca,'Position',[0.13 0.7 0.69 0.25])
        xline(0, "--k", 'LineWidth', 2);
        ylabel('Amplitude')
        axis('tight')
        title(sprintf('Average ECG'))

        subplot(2,1,2)
        hold on
        t = AVGECG.times(31:end)';
        % --- MedOn shaded SEM + line ---
        fill([t, fliplr(t)], ...
            [cluster_on_mean(31:end) + cluster_on_sem(31:end), ...
            fliplr(cluster_on_mean(31:end) - cluster_on_sem(31:end))], ...
            [0.2 0.6 1], 'FaceAlpha', 0.3, 'EdgeColor', 'none');  % blue-ish
        h1 = plot(t, cluster_on_mean(31:end), 'Color', [0 0.3 0.9], 'LineWidth', 2);

        % --- MedOff shaded SEM + line ---
        fill([t, fliplr(t)], ...
            [cluster_off_mean(31:end) + cluster_off_sem(31:end), ...
            fliplr(cluster_off_mean(31:end) - cluster_off_sem(31:end))], ...
            [1 0.4 0.4], 'FaceAlpha', 0.3, 'EdgeColor', 'none');  % red-ish
        h2 = plot(t, cluster_off_mean(31:end), 'Color', [0.9 0.1 0.1], 'LineWidth', 2);

        xline(0, "--k", 'LineWidth', 2);
        xlabel('Time (s)');
        ylabel('Amplitude');
        title(['HEP Comparison MedOn vs MedOff - ' chantag]);
        legend([h1 h2], {'MedOn', 'MedOff'}, 'Location', 'northeastoutside');
        set(gca, 'Position', [0.13 0.1 0.778 0.5]);
        grid on;
        box on;

        %gr2 = fullfile('/Volumes','LP3', 'HeadHeart', '2_results', 'erp', 'clustering', [ 'Hierarchical-Clustering_Comparison_HEP_',char(chantag),'.png']);
        gr2 = fullfile('E:', 'HeadHeart', '2_results', 'erp', 'clustering', [ 'Hierarchical-Clustering_Comparison_HEP_',char(chantag),'.png']);
        exportgraphics(f2, gr2, "Resolution", 300)
    end
end

if ismember('Plot Mean Clustering by Method', steps)

    meds = {'MedOn' 'MedOff'};

    for m = 1:length(meds)
        med = meds{m};

        fprintf('Loading EEG Cluster Data \n');
        pattern = fullfile(data_dir, 'erp', 'group', 'clustering', ['Cluster_Data_Flipped_EEG', '*']);
        files = dir(pattern);
        filename = fullfile(files(1).folder, files(1).name);
        load(filename, 'Cluster');

        % Filter correct field
        fields = fieldnames(Cluster);
        field_eeg = fields{contains(fields, med)};
        %field_eeg_off = fields{contains(fields, 'MedOff')};

        % Mean + SEM EEG
        cluster_eeg_mean = mean(Cluster.(field_eeg),1);
        cluster_eeg_sem = std(Cluster.(field_eeg)) / sqrt(length(Cluster.(field_eeg)));
        % cluster_off_eeg_mean = mean(Cluster.(field_eeg_off),1);
        % cluster_off_eeg_sem = std(Cluster.(field_eeg_off),1) / sqrt(length(Cluster.(field_eeg_off)));

        fprintf('Loading STN Cluster Data \n');
        pattern = fullfile(data_dir, 'erp', 'group', 'clustering', ['Cluster_Data_Flipped_STN', '*']);
        files = dir(pattern);
        filename = fullfile(files(1).folder, files(1).name);
        load(filename, 'Cluster')

        % Filter correct field
        fields = fieldnames(Cluster);
        field_stn = fields{contains(fields, med)};
        %field_stn_off = fields{contains(fields, 'MedOff')};

        % Mean + SEM STN
        cluster_stn_mean = mean(Cluster.(field_stn),1);
        cluster_stn_sem = std(Cluster.(field_stn)) / sqrt(length(Cluster.(field_stn)));
        %cluster_off_stn_mean = mean(Cluster.(field_stn_off),1);
        %cluster_off_stn_sem = std(Cluster.(field_stn_off),1) / sqrt(length(Cluster.(field_stn_off)));

        % Create one figure for all clusters in current condition
        f2 = figure; hold on;
        set(f2,'Position',[1 59 1440 738]);
        subplot(2,1,1)
        plot(AVGECG.times(31:end), mean(AVGECG.mean(31:end),1), 'Color', 'k'); hold on
        set(gca,'Position',[0.13 0.7 0.778 0.25])
        xline(0, "--k", 'LineWidth', 2);
        ylabel('Amplitude')
        axis('tight')
        title(sprintf('Average ECG'))

        subplot(2,1,2)
        hold on
        t = AVGECG.times(31:end)';
        % --- STN MedOn shaded SEM + line ---
        fill([t, fliplr(t)], ...
            [cluster_stn_mean(31:end) + cluster_stn_sem(31:end), ...
            fliplr(cluster_stn_mean(31:end) - cluster_stn_sem(31:end))], ...
            [0.2 0.6 1], 'FaceAlpha', 0.3, 'EdgeColor', 'none');  % blue-ish
        h1 = plot(t, cluster_stn_mean(31:end), 'Color', [0 0.3 0.9], 'LineWidth', 2);

        % --- EEG Med On shaded SEM + line ---
        fill([t, fliplr(t)], ...
            [cluster_eeg_mean(31:end) + cluster_eeg_sem(31:end), ...
            fliplr(cluster_eeg_mean(31:end) - cluster_eeg_sem(31:end))], ...
            [1 0.4 0.4], 'FaceAlpha', 0.3, 'EdgeColor', 'none');  % red-ish
        h2 = plot(t, cluster_eeg_mean(31:end), 'Color', [0.9 0.1 0.1], 'LineWidth', 2);

        xline(0, "--k", 'LineWidth', 2);
        xlabel('Time (s)');
        ylabel('Amplitude');
        title(['HEP Comparison EEG vs STN, Med: ', med]);
        legend([h1 h2], {'STN', 'EEG'}, 'Location', 'northeastoutside');
        set(gca, 'Position', [0.13 0.1 0.778 0.5]);
        grid on;
        box on;

        % gr2 = fullfile('/Volumes','LP3', 'HeadHeart', '2_results', 'erp', 'clustering', [ 'Hierarchical-Clustering_Comparison_STN_EEG_HEP_',char(med),'.png']);
        gr2 = fullfile('E:', 'HeadHeart', '2_results', 'erp', 'clustering', [ 'Hierarchical-Clustering_Comparison_STN_EEG_HEP_',char(med),'.png']);
        exportgraphics(f2, gr2, "Resolution", 300)
    end
end

function [IdxTable, clusterTable, reshaped] = plot_clustered_HEPs(CLUS, condition, chantag, nClusters, flipped_clusters, AVGECG, clusterTable)
% Inputs:
%   CLUS: struct with fields like CLUS.MedOn.EvDataAllAvgTrs
%   condition: 'on', 'off', or 'both'

% Parameters
time = linspace(-0.3, 0.6, size(CLUS.MedOn.(chantag).EvDataAllAvgTrs, 3));  % time vector

% Load data based on input
switch lower(condition)
    case 'on'
        data_on = CLUS.MedOn.(chantag).EvDataAllAvgTrs;  % [sub, chan, time]
        titles = {'MedOn'};
    case 'off'
        data_off = CLUS.MedOff.(chantag).EvDataAllAvgTrs;
        titles = {'MedOff'};
    case 'both'
        data_on = CLUS.MedOn.(chantag).EvDataAllAvgTrs;
        data_off = CLUS.MedOff.(chantag).EvDataAllAvgTrs;
        titles = {'MedOn', 'MedOff'};
    otherwise
        error('Condition must be "on", "off", or "both"');
end

% Process single or both conditions
for iCond = 1:length(titles)
    thisTitle = lower(strrep(titles{iCond}, ' ', ''));  % converts 'med on' → 'medon'

    switch thisTitle
        case 'medon'
            data = data_on;
            sub2remove = []; % All 6 Stn 6
            data(sub2remove, :, :) = [];
            if ~isempty(sub2remove)
                uniqueSubs = unique(clusterTable.Subject, 'stable');
                subjectNameToRemove = uniqueSubs{sub2remove};
                rmidx = find(strcmp(clusterTable.Subject, subjectNameToRemove));
                clusterTable(rmidx, :) = [];
            end
        case 'medoff'
            data = data_off;
            sub2remove = []; %  All 5
            data(sub2remove, :, :) = [];
            if ~isempty(sub2remove)
                uniqueSubs = unique(clusterTable.Subject, 'stable');
                subjectNameToRemove = uniqueSubs{sub2remove};
                rmidx = find(strcmp(clusterTable.Subject, subjectNameToRemove));
                clusterTable(rmidx, :) = [];
            end
        otherwise
            error('Unknown condition title: %s', titles{iCond});
    end

    [nSub, nChans, nTime] = size(data);
    reshaped = reshape(data, nSub * nChans, nTime);  % [waveforms x time]
    reshaped_copy = reshaped;

    % Sanity check function for the reshaoed order 
    % L = 3 + (271-1) * (nSub*nChans);   % L is linear index into data(:)
    % [s,c,tt] = ind2sub([nSub, nChans, nTime], L);

    % Make mapping table
   idxMap = [repmat((1:nSub)', nChans, 1), kron((1:nChans)', ones(nSub,1))];

    % Remove NaNs
    zero_idx = find(any(isnan(reshaped), 2) | all(reshaped == 0, 2));
    reshaped = reshaped(~any(isnan(reshaped), 2) & ~all(reshaped == 0, 2), :);
    idxMap(zero_idx,:) = [];

    if height(clusterTable) ~= length(idxMap)
    warning('Length of Cluster Table and IdxMap is different -> check')
    end

    % remove outliers
    switch thisTitle
        case 'medon'
            rowsToRemove = [7, 20, 89, 97, 98, 104, 110]; % EEG 7,20 ALL 7, 20, 89, 97, 98, 104, 110
            reshaped(rowsToRemove, :) = [];
            idxMap(rowsToRemove,:) = [];
            clusterTable(rowsToRemove,:) = [];
        case 'medoff'
            rowsToRemove = [27, 68,69, 70, 77 ]; % EEG 27, STN 16,7 ALL 27, 68,69, 70, 77 
            reshaped(rowsToRemove, :) = [];
            idxMap(rowsToRemove,:) = [];
            clusterTable(rowsToRemove,:) = [];
    end
    
    % Clustering
    D = pdist(reshaped, 'euclidean');
    Z = linkage(D, 'ward');
    labels = cluster(Z, 'maxclust', nClusters);

    % %plot_dendogram(Z, reshaped, AllSubsChansStn, AllSubsOnlyStn, AllSubsOnlyEEG, subject_info, chantag, titles)
    % f3=figure;
    % dendrogram(Z);
    % gr3 = fullfile('/Volumes','LP3', 'HeadHeart', '2_results','erp', 'clustering', 'dendrogram', ['Dendrogram_', char(titles), '_', char(chantag), '.png']);
    % exportgraphics(f3, gr3,"Resolution", 300);

    % Create logical array marking flipped clusters
    isFlipped = ismember(labels, flipped_clusters);

    IdxTable = table(idxMap(:,1), idxMap(:,2), labels, isFlipped, ...
    'VariableNames', {'Subject', 'Channel', 'Cluster', 'Flipped'});

    % Polarity correction
    for cl = flipped_clusters
        reshaped(labels == cl, :) = -reshaped(labels == cl, :);
    end

    f1 = figure;
    set(f1,'Position',[1 59 1440 738]);
    % Plotting
    for cl = 1:nClusters
        subplot(nClusters, length(titles), (cl-1)*length(titles) + iCond);
        idx = labels == cl;
        m = mean(reshaped(idx,:),1);
        e = std(reshaped(idx,:),0,1)/sqrt(sum(idx));
        fill([time fliplr(time)], [m+e fliplr(m-e)], 'b', 'FaceAlpha',0.2, 'EdgeColor','none'); hold on;
        plot(time, m, 'b', 'LineWidth', 1.5);
        title([titles{iCond} ' - Cluster ' num2str(cl) ', N=' num2str(sum(idx))]);
        xlabel('Time (s)');
        ylabel('Amplitude');
        grid on;
    end
    %gr1 = fullfile( '/Volumes','LP3', 'HeadHeart', '2_results','erp', 'clustering', [ 'Hierarchical-Clustering_Single_HEP_',char(chantag), '_', char(titles), '_nClusters=', num2str(nClusters), '_nflippedcluster=', num2str(flipped_clusters),'.png']);
    gr1 = fullfile( 'E:', 'HeadHeart', '2_results','erp', 'clustering', [ 'Hierarchical-Clustering_Single_HEP_',char(chantag), '_', char(titles), '_nClusters=', num2str(nClusters), '_nflippedcluster=', num2str(flipped_clusters),'.png']);
    %exportgraphics(f1, gr1, "Resolution", 300)

    % Create one figure for all clusters in current condition
    f2 = figure; hold on;
    set(f2,'Position',[1 59 1440 738]);
    subplot(2,1,1)
    plot(AVGECG.times(31:end), mean(AVGECG.mean(31:end),1), 'Color', 'k'); hold on
    set(gca,'Position',[0.13 0.7 0.69 0.25])
    xline(0, "--k", 'LineWidth', 2);
    ylabel('Amplitude')
    axis('tight')
    title(sprintf('Average ECG, Med: %s', char(titles)))

    subplot(2,1,2)
    hold on
    plotHandles = gobjects(nClusters, 1); 
    for cl = 1:nClusters
        idx = labels == cl;
        ninclus(cl) = sum(idx);
        m = mean(reshaped(idx,:), 1);
        e = std(reshaped(idx,:), 0, 1) / sqrt(sum(idx));
        % Shaded area for SEM
        meplus = m+e;
        meminus = m-e;
        fill([time(31:end), fliplr(time(31:end))], [meplus(31:end), fliplr(meminus(31:end))], 'k', 'FaceAlpha', 0.2, 'EdgeColor', 'none');  % Default gray; replaced by color below
        % Plot mean line (color auto-cycles)
        plotHandles(cl) = plot(time(31:end), m(31:end), 'LineWidth', 1.5);
        set(gca,'Position',[0.13 0.1 0.778 0.5])
        xline(0, "--k", 'LineWidth', 2);
    end
    xlabel('Time (s)');
    ylabel('Amplitude');
    title([titles{iCond} ' - HEP Clusters - ' chantag ]);
    legend(plotHandles, compose('Cluster %d, N = %d', (1:nClusters)', ninclus(:)), 'Location', 'northeastoutside');
    grid on;
    box on;

    %gr2 = fullfile('/Volumes','LP3', 'HeadHeart', '2_results', 'erp', 'clustering', [ 'Hierarchical-Clustering_All_HEP_',char(chantag) ,'_', char(titles), '_nClusters=', num2str(nClusters), '_nflippedcluster=', num2str(flipped_clusters),'.png']);
    gr2 = fullfile('E:','HeadHeart', '2_results', 'erp', 'clustering', [ 'Hierarchical-Clustering_All_HEP_',char(chantag) ,'_', char(titles), '_nClusters=', num2str(nClusters), '_nflippedcluster=', num2str(flipped_clusters),'.png']);
    %exportgraphics(f2, gr2, "Resolution", 300)

end
end

function [labels_dendo] = plot_dendogram(Z, reshaped, AllSubsChansStn, AllSubsOnlyStn, AllSubsOnlyEEG, subject_info, chantag, titles)

if strcmp(titles, 'MedOn')  % All Subs that are MedOn
    subjects = string({subject_info([subject_info.MedOn] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.MedOn] == 1);
    FltSubsOnlyStn = AllSubsOnlyStn([subject_info.MedOn] == 1);
    FltSubsOnlyEEG = AllSubsOnlyEEG([subject_info.MedOn] == 1);
elseif strcmp(titles, 'MedOff') % All Subs that are MedOff
    subjects = string({subject_info([subject_info.MedOff] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.MedOff] == 1);
    FltSubsOnlyStn = AllSubsOnlyStn([subject_info.MedOff] == 1);
    FltSubsOnlyEEG = AllSubsOnlyEEG([subject_info.MedOff] == 1);
end

% Filter channels
if strcmp(chantag, 'ALL')
    channels = FltSubsChansStn;
elseif strcmp(chantag, 'EEG')
    channels = FltSubsOnlyEEG;
elseif strcmp(chantag, 'STN')
    channels = FltSubsOnlyStn;
end

% Generate labels
labels_dendo = {};
for iSub = 1:length(subjects)
    sub = subjects{iSub};
    chans = channels{iSub};
    for iChan = 1:numel(chans)
        labels_dendo{end+1} = sprintf('%s-%s', sub, chans{iChan});
    end
end

% Output dimensions for reshaping 
valid_idx = ~any(isnan(reshaped), 2);
labels_dendo = labels_dendo(valid_idx);  % column

figure;  % create new figure
dendrogram(Z, 0, 'Labels', labels_dendo);
title(sprintf('Dendrogram - %s - %s', chantag, titles), 'Interpreter', 'none');
xtickangle(45);

% Create filename from condition and chantag
filename = fullfiles('Volumes','LP3', 'HeadHeart', '0_data','erp', 'group', 'dendogram', ['Dendrogram_', char(titles), '_', char(chantag), '.png']);
saveas(gcf, filename);  % save figure as PNG

end


%% ============== Plotting ERPs GROUP LEVEL=========================
% Channel combination needs to be created
if ismember('ERPs Group', steps)
    for c = 1:numel(channels)
        channels = {'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'Pz', 'STNl', 'STNr'};
        channel = channels{c};

        colors = lines(15);


        EvDataAll_chanavg = squeeze(mean(squeeze(EvDataAllAvgTrsZScore(:,c,:)),1));
        EvDataAll_chanavg = smoothdata(EvDataAll_chanavg, 'gaussian', 10); % Apply a Gaussian Filter to Smoothe the lines

        f2 = figure;
        set(f2,'Position', [1949 123 1023 785]);

        subplot(2,1,1)
        plot(TmAxis(31:end), AVGECG.mean(31:end)', 'Color', 'k'); hold on
        set(gca,'Position',[0.1300 0.5838 0.77 0.3])
        xline(0, "--k", 'LineWidth', 2);
        title(sprintf('Grand Average ECG in %s, medication: %s', channel, medname))
        axis("tight");
        ylabel('Amplitude (μV)')
        hold off

        subplot(2,1,2)
        for s = 1:numel(subjects.goodHeartMOff)
            subject = subjects.goodHeartMOff{sub};
            EvDataSubAvgTrs = squeeze(EvDataAllAvgTrsZScore(s,c,:));
            plot(TmAxis(31:end), EvDataSubAvgTrs(31:end), 'Color', colors(s, :), 'DisplayName', subject, 'LineWidth', 1);
            hold on
        end
        plot(TmAxis(31:end), EvDataAll_chanavg(31:end), 'Color', 'r', 'LineWidth', 3, 'DisplayName', 'Average');
        %legend('Location','southwest', 'FontSize',6)
        xline(0, "--k", 'LineWidth', 2, 'HandleVisibility','off');
        xlabel('Time (s)') % Add x-label
        ylabel('Amplitude (μV)') % Add y-label
        title(sprintf('Grand Average ERP in %s, medication: %s, LPF = %uHz, GF=10', channel, medname, Flp))
        axis("tight");

        gr2 = fullfile('F:\HeadHeart\2_results\erp' , ['AvgERP_', channel, '_', medname, '_HP=',  num2str(Fhp), '_LP=',  num2str(Flp), '_BSL=', num2str(baseline_win(1)), 'to', num2str(baseline_win(2)), 'GF=On', '.png']);
        exportgraphics(f2,gr2, 'Resolution', 300)

    end
end
%% ====== Plot ERP For Frontal Central and Parietal Electrodes together ========
if ismember('ERP Group Cluster', steps)

    % Define EEG clusters
    clusters = ["Frontal", "Central", "Parietal", "STNleft", "STNright"];
    Frontal = ["F3", "F4", "Fz"];
    Central = ["C3", "C4", "Cz"];
    Parietal = ["P3", "P4", "Pz"];
    STNleft = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"];
    STNright = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"];


    % Store in a struct for easy access
    clusterMap = struct('Frontal', Frontal, 'Central', Central, 'Parietal', Parietal, 'STNleft', STNleft, 'STNright', STNright);

    for ci = 4:numel(clusters)
        cluster_name = clusters{ci}; % e.g., 'frontal'
        cluster_channels = clusterMap.(cluster_name); % Get corresponding channels

        % Initialize figure
        f3 = figure;
        set(f3, 'Position', [1949 123 1023 785]);

        % ECG Plot (Remains unchanged)
        subplot(2,1,1);
        plot(TmAxis(31:end), AVGECG.mean(31:end)', 'Color', 'k'); hold on;
        set(gca, 'Position', [0.1300 0.5838 0.77 0.3]);
        xline(0, "--k", 'LineWidth', 2);
        title(sprintf('Grand Average ECG, medication: %s', medname));
        axis("tight");
        ylabel('Amplitude (μV)');
        hold off;

        % Iterate through EEG clusters
        subplot(2,1,2);
        hold on;
        colors = lines(numel(subjects)); % Generate unique colors per subject

        % EvDataAllChanAvg = squeeze(mean(squeeze(mean(EvDataAllAvgTrsZScore(:,1:2,:),2)),1));
        % EvDataAllChanAvg = smoothdata(EvDataAll_chanavg, 'gaussian', 10); % Apply a Gaussian Filter to Smoothe the lines
        %
        % % Initialize matrix for storing averages per subject
        % EvDataAllChanAvg = zeros(numel(subjects), numel(TmAxis));

        if sum(strcmp(["STNleft", "STNright"], cluster_name))==1
            sg47idx = find(ismember(subjects, "SG047"));
            FltSubsChansStn(sg47idx)= [];
            subjects(subjects == "SG047") = [];
            EvDataAllAvgTrsZScore(sg47idx,:,:) = [];
            size(EvDataAllAvgTrsZScore)
        end

        for s = 1:numel(subjects)
            subject = subjects{s};

            if BPReref
                subject_channels = FltSubsChansStn{s};
            else
                subject_channels = FltSubsChansRaw{s}; % Extract available EEG channels for this subject % here maybe channels because also STNl and STNr
            end

            % Find available channel indices for this subject
            chanIdx = find(ismember(subject_channels, cluster_channels));

            if isempty(chanIdx)
                warning('No matching channels for subject %s in cluster %s.', subject, cluster_name);
                continue;
            end

            subplot(2,1,2)

            % Extract and average data for available channels
            for ch = 1:numel(chanIdx)
                cha = chanIdx(ch);
                EvDataSub = squeeze(EvDataAllAvgTrsZScore(s, cha, :));
                plot(TmAxis(31:end), EvDataSub(31:end), 'Color', colors(s, :), 'DisplayName', subject, 'LineWidth', 0.3);
                hold on
                EvDataAllClus(s, ch, :) = EvDataSub;
            end

            % Code für wenn man die Channels meaned bevor man sie plotted
            % for ch = 1: numel(chanIdx)
            %     cha = chanIdx(ch);
            %     EvDataSub = squeeze(EvDataAllAvgTrsZScore(s, cha, :));
            %     EvDataAllClus(s, ch, :) = EvDataSub;
            % end
            % % EvDataSubAvg = squeeze(mean(EvDataAllClus(s,:,:), 2));
            % % plot(TmAxis(31:end), EvDataSubAvg(31:end), 'Color', colors(s, :), 'DisplayName', subject, 'LineWidth', 0.5);
            % % hold on
        end

        % Compute grand average across subjects
        grandAvg = squeeze(mean(squeeze(mean(EvDataAllClus, 2)),1));
        grandAvg = smoothdata(grandAvg, 'gaussian', 10); % Apply Gaussian smoothing

        % Plot grand average
        plot(TmAxis(31:end), grandAvg(31:end), 'Color', 'r', 'LineWidth', 5, 'DisplayName', ['Avg ' cluster_name]);
        % Final plot formatting
        xline(0, "--k", 'LineWidth', 2, 'HandleVisibility', 'off');
        xlabel('Time (s)');
        ylabel('Amplitude (μV)');
        title(sprintf('Grand Average ERP %s, medication: %s, LPF = %uHz, GF=10', cluster_name, medname, Flp));
        axis("tight");
        %legend('Location', 'southwest', 'FontSize', 6);
        hold off;
        % Save the figure
        if BPReref
            gr3 = fullfile(results_dir, '/erp', ['AvgERP_', cluster_name, '_', medname, '_HP=', num2str(Fhp), '_LP=', num2str(Flp), '_BSL=', num2str(baseline_win(1)), 'to', num2str(baseline_win(2)), '_GF=On_', BPRerefTit, '.png']);
        else
            gr3 = fullfile(results_dir, '/erp', ['AvgERP_', cluster_name, '_', medname, '_HP=', num2str(Fhp), '_LP=', num2str(Flp), '_BSL=', num2str(baseline_win(1)), 'to', num2str(baseline_win(2)), '_GF=On_','.png']);
        end
        exportgraphics(f3, gr3, 'Resolution', 300);
    end
end

%  clusters = {'frontal', 'central', 'parietal'};
% frontal = {'F3', 'F4' 'Fz'};
% central = {'C3', 'C4', 'Cz'};
% parietal = {'P3', 'P4', 'Pz'};
%
% EvDataAll_chanavg = squeeze(mean(squeeze(mean(EvDataAllAvgTrsZScore(:,1:2,:),2)),1));
% EvDataAll_chanavg = smoothdata(EvDataAll_chanavg, 'gaussian', 10); % Apply a Gaussian Filter to Smoothe the lines
%
% colors = lines(15);
%
% f3 = figure;
% set(f3,'Position', [1949 123 1023 785]);
%
% subplot(2,1,1)
% plot(TmAxis(31:end), AVGECG.mean(31:end)', 'Color', 'k'); hold on
% set(gca,'Position',[0.1300 0.5838 0.77 0.3])
% xline(0, "--k", 'LineWidth', 2);
% title(sprintf('Grand Average ECG, medication: %s', medname))
% axis("tight");
% ylabel('Amplitude (μV)')
% hold off
%
% subplot(2,1,2)
% for s = 1:numel(subjects)
%     for c = 1:2
%         subject = subjects{s};
%         EvDataSubAvgTrs = squeeze(EvDataAllAvgTrsZScore(s,c,:));
%         plot(TmAxis(31:end), EvDataSubAvgTrs(31:end), 'Color', colors(s, :), 'DisplayName', subject, 'LineWidth', 1);
%         hold on
%     end
% end
% plot(TmAxis(31:end), EvDataAll_chanavg(31:end), 'Color', 'r', 'LineWidth', 5, 'DisplayName', 'Average');
% %legend('Location','southwest', 'FontSize',6)
% xline(0, "--k", 'LineWidth', 2, 'HandleVisibility','off');
% xlabel('Time (s)') % Add x-label
% ylabel('Amplitude (μV)') % Add y-label
% title(sprintf('Grand Average ERP in Frontal EEG (F3 + F4), medication: %s, LPF = %uHz, GF=10', medname, Flp))
% axis("tight");
%
% gr3 = fullfile('F:\HeadHeart\2_results\erp' , ['AvgERP_Frontals_', medname, '_HP=',  num2str(Fhp), '_LP=',  num2str(Flp), '_BSL=', num2str(baseline_win(1)), 'to', num2str(baseline_win(2)), 'GF=On','.png']);
% exportgraphics(f3,gr3, 'Resolution', 300)
% end

disp('================= ERP DONE! =======================')



function   [EvDataPerm] = time_lock_to_surrogate(ChDta, surrogate_rpeaks, SR, tWidth, tOffset, NewSR, Fhp, Flp, baseline_win, FltPassDir)
nEvent=length(surrogate_rpeaks);

% HIGH PASS FILTER
if Fhp > 0
    ChDta=ft_preproc_highpassfilter(ChDta,SR,Fhp,4,'but',FltPassDir); % twopass
end
if Flp > 0
    ChDta = ft_preproc_lowpassfilter(ChDta, SR, Flp, 4, 'but',FltPassDir);
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

[EventTms,EvDataPerm,TmAxis]=GetEvTimeAndData(surrogate_rpeaks,ChDta,dtTime,tWidth,tOffset);
[nEvs,nData]=size(EvDataPerm);

%fprintf('****************** Baseline Correction for %s %s med: %s ...****************\n', subject, channel, medname);
% My baseline is the time window -0.3 to -0.1 s
% before my Rpeak of every trial

% Find the the indices for the basseline window
bidx = find(TmAxis' >= baseline_win(1) & TmAxis' <= baseline_win(2));

% for every trial calc the mean of the baseline win
% and subtract that from the entire epoch
for t = 1:nEvs
    baseline_mean = mean(EvDataPerm(t, bidx(1):bidx(end)),2);
    EvDataPerm(t,:) = EvDataPerm(t,:)-baseline_mean;
end


end

