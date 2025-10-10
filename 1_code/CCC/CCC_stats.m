% function [] = sub_CCC_stats(subjects, data_dir, results_dir)

%% Epoching and Time Locking Data for HeadHeart

% Author: Lisa Paulsen
% Contact: lisaspaulsen[at]web.de
% Created on: 1 October 2024
% Last update: 15 October 2024

%% REQUIRED TOOLBOXES
% Image Processing Toolbox
% Signal Processing Toolbox
% Statistics and Machine Learning Toolbox

% Get the Cross-Channel Coherence (CCC) through the Phase Synchronization
% Index and use Permutation Statistics for the statistical Analysis
%
% Inputs:
% Preprocessed data (EEG, LFP, ECG) from .mat file
%
% Outputs:
% CCC in Time Frequency Plots
%

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

%flag if baseline is on or off
baseline = true;

% Define feature extraction steps to perform
steps = {'Group Load TTest by med'}; % 'Parametric Stats CCC', 'Group CCC Load Chan', 'PermStats', 'Calc Single Subject CCC', 'Plot SubAvg CCC', 'Plot Power', 'Plot SubAvg PermStats'

% ipsilateral
CCCchans.Comp1 = {'STNl', 'C3'};
CCCchans.Comp2 = {'STNl', 'F3'};
CCCchans.Comp3 = {'STNl', 'Pz'};
CCCchans.Comp4 = {'STNr', 'C4'};
CCCchans.Comp5 = {'STNr', 'F4'};
CCCchans.Comp6 = {'STNr', 'Pz'};

% contralateral
CCCchans.Comp7 = {'STNl', 'C4'};
CCCchans.Comp8 = {'STNl', 'F4'};
CCCchans.Comp9 = {'STNr', 'C3'};
CCCchans.Comp10 = {'STNr', 'F3'};
comps = {'Comp1', 'Comp2', 'Comp3', 'Comp4', 'Comp5', 'Comp6', 'Comp7', 'Comp8', 'Comp9','Comp10'};
comp_names = {'STNl - C3','STNl - F3','STNl - Pz', 'STNr - C4', 'STNr - F4', 'STNr - Pz', 'STNl - C4', 'STNl - F4','STNr - C3', 'STNr - F4'};

Hz_dir = '2Hz';

% Define Time Window
tWidth   = 0.9;
tOffset  = 0.3;
FltPassDir = 'twopass';
NewSR = 300;

nSub = numel(subjects);

BandWidth=2; % BandWidth in Hz;
Qfac     =2; % Attenuation in db(-Qfac)
tCircMean=0; % for By TRials calc

permstats = true;
numPerms = 1000;
surrogate = true;
trials = false;
plots = false;
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
    %if permstats
    %max_chan = max(cellfun(@numel, FltSubsChansStn));
    %PermCccAll = zeros(nSub, max_chan, numPerms, 148, 271);
    %end
    % ZScoresAll = zeros(nSub, numel(channels), 141, 271);
    % PValAll = zeros(nSub, numel(channels), 141, 271);

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
            pattern = fullfile(data_dir, 'tfr', Hz_dir, [subject,  '_TFR-EPOCH_', medname ,'_Rest_', BPRerefHiTit, '*', '_HP=', Hz_dir, '*']);
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
        for c1 = 10:numel(fieldnames(CCCchans))
            comp = comps{c1};
            channel1 = CCCchans.(comp){1};
            channel2 = CCCchans.(comp){2};

            if strcmp(channel1, 'STNl'); chan1 = FltSubsOnlyStn{sub}{1}; elseif strcmp(channel1, 'STNr'); chan1 = FltSubsOnlyStn{sub}{2}; else; chan1 = channel1; end
            if strcmp(channel2, 'STNl'); chan2 = FltSubsOnlyStn{sub}{1}; elseif strcmp(channel2, 'STNr'); chan2 = FltSubsOnlyStn{sub}{2}; else; chan2 = channel2; end


            availchans = FltSubsOnlyEEG{sub}; % EEG channels for this subject

            % Find valid EEG channels that exist for this subject
            validchans = intersect(CCCchans.(comp), availchans, 'stable');

            if isempty(validchans)
                fprintf('Subject %s does not have the fitting EEG channels for %s.\n', subject, comp);
                continue; % Skip this subject
            end

            fprintf('************ Calculating CCC for %s in %s and %s **************** \n', subject, channel1, channel2);
            % Calc original CCC
            [FrsTmPsiTrial]=Get_PSI_ByTrials(TFR.(chan2).phase,TFR.(chan1).phase,SR,tCircMean);

            % Scale the CCC to the relative CCC of the channel
            % meanFrsTmCcc = mean(mean(FrsTmPsiTrial,1),2);
            % relFrsTmCcc = FrsTmPsiTrial/meanFrsTmCcc;

            %CccAll(sub,c1,:,:) = FrsTmPsiTrial; % SubjectxChannelxFreqxTime
            %RelCccAll(sub,c1,:,:) = relFrsTmCcc;


            % % --- Input: Phase data from two channels
            % % Ch1EvsFrsTmPha: [nTrials x nFreqs x nTimes]
            % % Ch2EvsFrsTmPha: [nTrials x nFreqs x nTimes]
            %
            % [nTrials, nFreqs, nTimes] = size(TFR.(chan1).phase);
            % Ch1EvsFrsTmPha = TFR.(chan1).phase;
            % Ch2EvsFrsTmPha = TFR.(chan2).phase;
            %
            % PSI = zeros(nFreqs, nTimes);     % Phase slope index
            %
            % for itm = 1:nTimes
            %     dphi_trials = zeros(nTrials, nFreqs);  % [nTrials x nFreqs]
            %     for tr = 1:nTrials
            %         pha1 = squeeze(Ch1EvsFrsTmPha(tr, :, itm));  % [1 x nFreqs]
            %         pha2 = squeeze(Ch2EvsFrsTmPha(tr, :, itm));
            %
            %         % Calculate phase difference directly (skip unwrapping and polyfit)
            %         dphi = angle(exp(1i * (pha1 - pha2)));  % Phase difference without unwrapping
            %
            %         % Store phase differences (no unwrapping step)
            %         dphi_trials(tr, :) = dphi;
            %     end
            %
            %     % === PSI calculation === (slope of phase diff across freqs)
            %     for ifr = 1:nFreqs
            %         dphi_current_freq = dphi_trials(:, ifr);  % Phase difference at current frequency
            %
            %         % Calculate PSI (synchrony measure) at the current frequency and time
            %         % Here we just take the mean of the phase differences (or you can use a more complex measure)
            %         psi_slope = mean(dphi_current_freq);  % Mean phase difference for the current frequency
            %
            %         % Store PSI value for this frequency and time point
            %         PSI(ifr, itm) = psi_slope;
            %     end
            % end

            if BPRerefHi
                outputPDF1 = fullfile(results_dir, 'ccc', 'ss' ,[subject,'_CCC_', medname , '_BP=', BPRerefHiTit, '_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_HP=', num2str(freqs(1)) ,'.pdf']);
            elseif BPRerefLw
                outputPDF1 = fullfile(results_dir, 'ccc', 'ss' ,[subject,'_CCC_', medname , '_BP=', BPRerefLwTit, '_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_HP=', num2str(freqs(1)) ,'.pdf']);
            else
                outputPDF1 = fullfile(results_dir, 'ccc', 'ss' ,[subject,'_CCC_', medname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_HP=', num2str(freqs(1)) ,'.pdf']);
            end


            if plots
                f1=figure;
                set(f1,'Position',[1949 123 1023 785]);
                subplot(2,1,1)
                plot(times, mean(EvECG.EvData,1), 'Color', 'k'); hold on
                set(gca,'Position',[0.1300 0.5838 0.71 0.3])
                xline(0, "--k", 'LineWidth', 2);
                ylabel('Amplitude')
                axis('tight')
                title(sprintf('Average ECG for %s, med= %s', subject, medname))
                hold off
                subplot(2,1,2)
                imagesc(times,freqs,squeeze(CccAll(sub,c1,:,:)));axis xy; %
                colormap('parula');
                xline(0, "--k", 'LineWidth', 2);
                col = colorbar;
                col.Label.String = 'CCC Values'; % Add title to colorbar
                xlabel('Time (s)') % Add x-label
                ylabel('Frequencies (Hz)') % Add y-label
                title(sprintf('CCC PSI for %s, %s - %s, med= %s', subject, channel1, channel2, medname))


                %gr1 = fullfile('F:\HeadHeart\2_results\ccc\ss' , [subject, '_', medname, '_CCC_', channel, '.png']);
                %exportgraphics(f1,gr1, 'Resolution', 300)
                exportgraphics(f1, outputPDF1, 'Append', true);
            end

            % Save CCC MAtrix of all Subjects and all Channels
            CCC.SR = SR;
            CCC.times = times;
            CCC.freqs = freqs;
            CCC.CCC = FrsTmPsiTrial; %FreqxTime

            if BPRerefHi & onlystn
                save_path = fullfile(data_dir, 'ccc', 'ss_chan', [subject, '_', channel1, '_', channel2, '_CCC_', medname , '_OnlySTN_BP=', BPRerefHiTit, '_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_HP=', num2str(freqs(1)) ,'.mat']);
            elseif BPRerefLw & onlystn
                save_path = fullfile(data_dir, 'ccc', 'ss_chan',[subject, '_', channel1, '_', channel2, '_CCC_', medname , '_OnlySTN_BP=', BPRerefLwTit, '_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_HP=', num2str(freqs(1)) ,'.mat']);
            elseif onlyeeg
                save_path = fullfile(data_dir, 'ccc', 'ss_chan',[subject, '_', channel1, '_', channel2, '_CCC_', medname ,'_OnlyEEG_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_HP=', num2str(freqs(1)) ,'.mat']);
            elseif allchans & BPRerefHi
                save_path = fullfile(data_dir, 'ccc', 'ss_chan',[subject, '_', channel1, '_', channel2, '_CCC_', medname ,'_BP=', BPRerefHiTit,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_HP=', num2str(freqs(1)) ,'.mat']);
            elseif allchans & BPRerefLw
                save_path = fullfile(data_dir, 'ccc', 'ss_chan',[subject, '_', channel1, '_', channel2, '_CCC_', medname ,'_BP=', BPRerefLwTit,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_HP=', num2str(freqs(1)) ,'.mat']);
            elseif allchans
                save_path = fullfile(data_dir, 'ccc', 'ss_chan',[subject, '_', channel1, '_', channel2, '_CCC_', medname ,'_BP=NONE','_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_HP=', num2str(freqs(1)) ,'.mat']);
            else
                save_path = fullfile(data_dir, 'ccc','ss_chan', [subject,'_', channel1, '_', channel2, '_CCC_', medname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), 'HPF=', num2str(freqs(1)), '_HP=',  num2str(freqs(1)) ,'.mat']);
            end
            save(save_path, 'CCC', '-v7.3');
            fprintf('Saved CCC Data for sub %s and channels %s %s to: %s\n', subject, channel1, channel2, save_path);

            clear FrsTmPsiTrial
        end

        %% ==================== PERMUTATION ===========================
        if ismember ('PermStats', steps)

            if permstats &  BPRerefHi
                outputPDF = fullfile(results_dir, 'ccc', 'ss_perm' ,[subject,'_CCC_', medname , '_BP=', BPRerefHiTit, '_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'_pval=', num2str(signif_thresh),'.pdf']);
            elseif permstats & BPRerefLw
                outputPDF = fullfile(results_dir, 'ccc', 'ss_perm' ,[subject,'_CCC_', medname , '_BP=', BPRerefLwTit, '_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'_pval=', num2str(signif_thresh),'.pdf']);
            elseif permstats
                outputPDF = fullfile(results_dir, 'ccc', 'ss_perm' ,[subject,'_CCC_', medname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'_pval=', num2str(signif_thresh),'.pdf']);
            end

            %Intitalize the PermCccAllChan Variable
            %PermCccAllChan = zeros(numel(channels),numPerms, freqs, times);


            for c = 1:numel(fieldnames(CCCchans))
                comp = comps{c};

                channel1 = CCCchans.(comp){1};
                channel2 = CCCchans.(comp){2};

                if strcmp(channel1, 'STNl'); chan1 = FltSubsOnlyStn{sub}{1}; elseif strcmp(channel1, 'STNr'); chan1 = FltSubsOnlyStn{sub}{2}; else; chan1 = channel1; end
                if strcmp(channel2, 'STNl'); chan2 = FltSubsOnlyStn{sub}{1}; elseif strcmp(channel2, 'STNr'); chan2 = FltSubsOnlyStn{sub}{2}; else; chan2 = channel2; end


                availchans = FltSubsOnlyEEG{sub}; % EEG channels for this subject

                % Find valid EEG channels that exist for this subject
                validchans = intersect(CCCchans.(comp), availchans, 'stable');

                if isempty(validchans)
                    fprintf('Subject %s does not have the fitting EEG channels for %s.\n', subject, comp);
                    continue; % Skip this subject
                end


                %[CCCzscores]=CCC_permutation_test(FrsTmCcc, relFrsTmCccAll, IBI.(medname){1}, numPerms, freqs, time_bins, SR, TFR.(channel).phase);
                fprintf('************ Calculating CCC for %s in %s and %s **************** \n', subject, channel1, channel2);

                % Intitialize variables
                %permuted_CCCs = zeros([numPerms, size(FrsTmCcc, 1), size(FrsTmCcc, 2)]);
                [nTrials, nFreq, nTms] = size(TFR.(chan1).phase);

                % Get the raw channel data
                chan1_idx = find(strcmp(chan1, SmrData.WvTitsCleaned));
                ChDta1 = SmrData.WvDataCleaned(chan1_idx, :);
                chan2_idx = find(strcmp(chan2, SmrData.WvTitsCleaned));
                ChDta2 = SmrData.WvDataCleaned(chan2_idx, :);
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
                    PermPSIData = zeros(numPerms,nFreq,nTms);

                    startTime = datetime('now');
                    disp(['Start Time: ', datestr(startTime)]);

                    for perm = 1:numPerms %par
                        % Time Lock the surrogate R Peaks to the Channel
                        % Data and apply the filters as well as the DS and
                        % create TFR for the new epochs
                        currSurrogateRpeaks = surrogate_rpeaks(perm, :);
                        [ChsAllFrsTmPha1] = time_lock_to_surrogate(ChDta1, currSurrogateRpeaks, oldSR, tWidth, tOffset, numPerms, NewSR, freqs, times);
                        [ChsAllFrsTmPha2] = time_lock_to_surrogate(ChDta2, currSurrogateRpeaks, oldSR, tWidth, tOffset, numPerms, NewSR, freqs, times);

                        %  Calculate CCC with the surrogate R-peaks (one per channel)
                        [PermPSIData(perm, :, :)] = Get_PSI_ByTrials(ChsAllFrsTmPha1,ChsAllFrsTmPha2,SR,tCircMean);

                        fprintf('perm = %d \n', perm)
                    end

                    endTime = datetime('now'); disp(['End Time: ', datestr(endTime)]);
                    % Calculate elapsed time
                    elapsedTime = endTime - startTime; disp(['Elapsed Time: ', char(elapsedTime)]);
                end
                % Step 1: Compute z-scores
                diff_sum_perm_mean = squeeze(mean(PermPSIData,1)); % Mean of the permutation distribution
                diff_sum_perm_std = squeeze(std(PermPSIData,1)); % Standard deviation of permutation distribution

                % diffPerm_mean(1,:,:) = diff_sum_perm_mean;
                % diffPerm_std(1,:,:)  = diff_sum_perm_std;

                zscores = (squeeze(CccAll(sub,c,:,:)) - diff_sum_perm_mean) ./ diff_sum_perm_std ;
                %zscores_perm = (PermPSIData - diffPerm_mean) ./ diffPerm_std;
                p_orig = 2 * (1 - normcdf(zscores, 0, 1));
                p_thresh = p_orig < signif_thresh;

                % ZScoresAll(sub,c,:,:) = zscores; % SubjectxChannelxFreqxTime (Last two are CCC ZScores)
                % PValAll(sub,c,:,:) = p_orig;  % SubjectxChannelxFreqxTime (Last two are CCC PVals)
                %PermCccAllChan(c,:,:,:) = PermPSIData; % SubjectxChannelxPermutationxFreqxTime


                 f3 = figure; % Sanity check that the distributions are normalized and overlapping so that my null hypothesis actually reflects my data
                 histogram(PermCccData);
                 title(sprintf('Perm CCC Distribution for %s in %s - %s, perms: %d', subject, channel1, channel2, numPerms))
                 exportgraphics(f3, outputPDF, 'Append', true);
                 f4 = figure; 
                 histogram(squeeze(CccAll(sub,c,:,:)));
                 title(sprintf('Original CCC Distribution for %s in %s - %s', subject, channel1, channel2, numPerms))
                 exportgraphics(f4, outputPDF, 'Append', true);
                 %title(sprintf('Perm CCC and Original CCC Dist for %s in %s, perms: %d, med: %s', subject, channel, numPerms, medname))
                %
                % f4 = figure;
                % subplot(2,1,1);
                % histogram(zscores);
                % %title(sprintf('Z Score Dist for %s in %s, perms: %d, med: %s', subject, channel, numPerms, medname))
                % subplot(2,1,2);
                % histogram(p_orig);
                % %title(sprintf('P Val Dist for %s in %s, perms: %d, med: %s', subject, channel, numPerms, medname))

                %ccc_zscores_thresh = (zscores > 2) | (zscores < -2);

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
                imagesc(times, freqs, squeeze(CccAll(sub,c,:,:))); axis xy;
                colormap('parula');
                col = colorbar;
                col.Label.String = 'CCC Values'; % Add title to colorbar
                clims = clim;
                hold on;
                contour(times, freqs, p_thresh, 1, 'linecolor', 'k', 'linewidth', 1.5);
                xline(0, "--k", 'LineWidth', 2);
                clim(clims);
                title(sprintf('CCC for %s,  %s - %s, perms: %d, med: %s, p<%.4g', subject, channel1, channel2, numPerms, medname, signif_thresh))
                xlabel('Time (s)') % Add x-label
                ylabel('Frequencies (Hz)') % Add y-label
                hold off

                % outputPDF = fullfile('F:\HeadHeart\2_results\ccc' , [subject, '_CCC-PermStats_chan=', channel, '_med=', medname, '_perm=', num2str(numPerms), ...
                %     '_win=-', num2str(tOffset),'to', num2str(tWidth-tOffset),'_BSL=', num2str(baseline), '_pval=', num2str(signif_thresh),'.pdf']);
                % exportgraphics(f5,gr5, 'Resolution', 300)
                exportgraphics(f5, outputPDF, 'Append', true);
                disp(['All figures saved to ', outputPDF]);

                % Save CCC MAtrix of all Subjects and all Channels
                CCC.SR = SR;
                CCC.times = times;
                CCC.freqs = freqs;
                CCC.PermCcc = squeeze(PermPSIData); %ChannelxPermutationxFreqxTime
                CCC.CCC = squeeze(CccAll(sub,c,:,:)); %FreqxTime

                if permstats & BPRerefHi & onlystn
                    save_path = fullfile(data_dir, 'ccc', 'ss_chan', [subject, '_', channel1, '_', channel2, '_CCC_', medname , '_OnlySTN_BP=', BPRerefHiTit, '_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
                elseif permstats & BPRerefLw & onlystn
                    save_path = fullfile(data_dir, 'ccc', 'ss_chan',[subject, '_', channel1, '_', channel2, '_CCC_', medname , '_OnlySTN_BP=', BPRerefLwTit, '_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
                elseif permstats & onlyeeg
                    save_path = fullfile(data_dir, 'ccc', 'ss_chan',[subject, '_', channel1, '_', channel2, '_CCC_', medname ,'_OnlyEEG_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
                elseif permstats & allchans & BPRerefHi
                    save_path = fullfile(data_dir, 'ccc', 'ss_chan',[subject, '_', channel1, '_', channel2, '_CCC_', medname ,'_BP=', BPRerefHiTit,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
                elseif permstats & allchans & BPRerefLw
                    save_path = fullfile(data_dir, 'ccc', 'ss_chan',[subject, '_', channel1, '_', channel2, '_CCC_', medname ,'_BP=', BPRerefLwTit,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
                elseif permstats & allchans
                    save_path = fullfile(data_dir, 'ccc', 'ss_chan',[subject, '_', channel1, '_', channel2, '_CCC_', medname ,'_BP=NONE','_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
                elseif permstats
                    save_path = fullfile(data_dir, 'ccc', 'ss_chan',[subject, '_', channel1, '_', channel2, '_CCC_', medname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
                else
                    save_path = fullfile(data_dir, 'ccc','ss_chan', [subject,'_', channel1, '_', channel2, '_CCC_', medname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), 'HPF=', num2str(freqs(1)), '_HP=',  num2str(freqs(1)) ,'.mat']);
                end
                save(save_path, 'CCC', '-v7.3');
                fprintf('Saved CCC Data for sub %s and channels %s %s to: %s\n', subject, channel1, channel2, save_path);

            end
        end

        % if permstats
        %     %CCC.PERM.ZScoresAll = ZScoresAll;  % SubjectxChannelxFreqxTime (Last two are CCC ZScores)
        %     %CC.PERM.PValAll = PValAll; % SubjectxChannelxFreqxTime (Last two are CCC PVals)
        %     CCC.PermCcc = PermPSIData; %ChannelxPermutationxFreqxTime
        % end
        %
        % % Save CCC MAtrix of all Subjects and all Channels
        % CCC.SR = SR;
        % CCC.times = times;
        % CCC.freqs = freqs;
        % CCC.CccAll = CccAll;
        % CCC.RelCccAll = RelCccAll;
        %
        % if permstats & BPRerefHi & onlystn
        %     save_path = fullfile(data_dir, 'ccc', 'ss', [subject, '_CCC_', medname , '_OnlySTN_BP=', BPRerefHiTit, '_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
        % elseif permstats & BPRerefLw & onlystn
        %     save_path = fullfile(data_dir, 'ccc', 'ss',[subject, '_CCC_', medname , '_OnlySTN_BP=', BPRerefLwTit, '_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
        % elseif permstats & onlyeeg
        %     save_path = fullfile(data_dir, 'ccc', 'ss',[subject, '_CCC_', medname ,'_OnlyEEG_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
        % elseif permstats & allchans & BPRerefHi
        %     save_path = fullfile(data_dir, 'ccc', 'ss',[subject, '_CCC_', medname ,'_BP=', BPRerefHiTit,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
        % elseif permstats & allchans & BPRerefLw
        %     save_path = fullfile(data_dir, 'ccc', 'ss',[subject, '_CCC_', medname ,'_BP=', BPRerefLwTit,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
        % elseif permstats & allchans
        %     save_path = fullfile(data_dir, 'ccc', 'ss',[subject, '_CCC_', medname ,'_BP=NONE','_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
        % elseif permstats
        %     save_path = fullfile(data_dir, 'ccc', 'ss',[subject, '_CCC_', medname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
        % else
        %     save_path = fullfile(data_dir, 'ccc','ss', [subject, '_CCC_', medname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), 'HPF=', num2str(freqs(1)), '_HP=',  num2str(freqs(1)) ,'.mat']);
        % end
        % save(save_path, 'CCC', '-v7.3');
        % fprintf('Saved CCC Data for sub %s and all channels to: %s\n', subject, save_path);

        % Free up some storage
        clear CccSub
        clear RelCccSub
        clear PermCccData
        clear PermCccAllChan
        clear TFR SmrData
    end

    % if permstats
    %     CCC.PERM.ZScoresAll = ZScoresAll;  % SubjectxChannelxFreqxTime (Last two are CCC ZScores)
    %     CCC.PERM.PValAll = PValAll; % SubjectxChannelxFreqxTime (Last two are CCC PVals)
    %     CCC.PERM.PermCccAll = PermCccAll; %SubjectxChannelxPermutationxFreqxTime
    % end
    % % Save CCC MAtrix of all Subjects and all Channels
    % CCC.SR = SR;
    % CCC.times = times;
    % CCC.freqs = freqs;
    % CCC.CccAll = CccAll;
    % CCC.RelCccAll = RelCccAll;
    %
    % if permstats & newsubs & BPRerefHi & onlystn
    %     save_path = fullfile(data_dir, 'ccc', ['CCC-NewSubs_', medname , '_OnlySTN_BP=', BPRerefHiTit, '_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    % elseif permstats & newsubs & BPRerefLw & onlystn
    %     save_path = fullfile(data_dir, 'ccc', ['CCC-NewSubs_', medname , '_OnlySTN_BP=', BPRerefLwTit, '_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    % elseif permstats & newsubs & onlyeeg
    %     save_path = fullfile(data_dir, 'ccc', ['CCC-NewSubs_', medname ,'_OnlyEEG_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    % elseif permstats & newsubs & allchans
    %     save_path = fullfile(data_dir, 'ccc', ['CCC-NewSubs_', medname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    % elseif permstats & oldsubs & BPRerefHi & onlystn
    %     save_path = fullfile(data_dir, 'ccc', ['CCC-OldSubs_', medname , '_OnlySTN_BP=', BPRerefHiTit, '_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    % elseif permstats & oldsubs & BPRerefLw & onlystn
    %     save_path = fullfile(data_dir, 'ccc', ['CCC-OldSubs_', medname , '_OnlySTN_BP=', BPRerefLwTit,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    % elseif permstats & oldsubs & onlyeeg
    %     save_path = fullfile(data_dir, 'ccc', ['CCC-OldSubs_', medname ,'_OnlyEEG_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    % elseif permstats & oldsubs & allchans
    %     save_path = fullfile(data_dir, 'ccc', ['CCC-OldSubs_', medname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    % elseif permstats
    %     save_path = fullfile(data_dir, 'ccc', ['CCC-AllSubs_', medname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), '_perm=', num2str(numPerms),'_HP=', num2str(freqs(1)) ,'.mat']);
    % else
    %     save_path = fullfile(data_dir, 'ccc', ['CCC-AllSubs_', medname ,'_time=', num2str(times(1)),'-', num2str(times(end)),'_DS=', num2str(SR), 'HPF=', num2str(freqs(1)), '_HP=',  num2str(freqs(1)) ,'.mat']);
    % end
    % save(save_path, 'CCC', '-v7.3');
    % fprintf('Saved CCC Data for all subs and channels to: %s\n', save_path);
end

%% ----------------- Group Load & TTest by med (CCC/PSI) -----------------
if ismember('Group Load TTest by med', steps)

    save_dta = true;

    pro = { 'Change and Save', 'Load', 'TTest SC'}; % options: , 'TTest SC', 'TTest Clus', 'Change and Save'

    %subject, '_', channel1, '_', channel2, '_CCC_', medname ,
    % Build list of all possible pairs (you can also set this manually)
    % Option A: use a predefined list:
    % allPairs = {'LSTN-C3','RSTN-C3','F3-P3', ...};
    % Option B: auto-detect from file names in ccc folder:
    ccc_dir = fullfile(data_dir, 'ccc', 'ss_chan');
    files_all = dir(fullfile(ccc_dir, '*.mat'));
    valid_idx = ~startsWith({files_all.name}, '._');
    files_all = files_all(valid_idx);
    allPairs = {};
    for ff = 1:numel(files_all)
        parts = split(files_all(ff).name, '_');
        % heuristic: parts might be [subj pair1 pair2 med ...]. Adapt if needed.
        if numel(parts) >= 3
            pairname = strcat(parts{2}, '_', parts{3}); % e.g. LSTN_RSTN
            allPairs{end+1} = pairname;
        end
    end
    allPairs = unique(allPairs);

    %% -------------------- Change and Save: gather per-pair files --------------------
    if ismember('Change and Save', pro)

        % Preallocate container when we know sizes (will be lazy-initialized)
        % allCCC will be [nSubjects x nFreq x nTime x nPairs]
        allCCC = [];

        for p = 1:numel(allPairs)
            pairname = allPairs{p};
            fprintf('Processing pair: %s\n', pairname);

            for s = 1:numel(subjects)
                subject = subjects(s);
                fprintf('Processing subject: %s %i/%i\n', subject, s, numel(subjects));

                use_hi_L = false;
                use_hi_R = false;

                if contains(pairname, 'STNl')
                    use_hi_L = subject_info(s).ITC_BPReref_L;
                    if use_hi_L == 1; expected_L = 'BPRerefHi'; else expected_L = 'BPRerefLow'; end
                elseif contains(pairname, 'STNr')
                    use_hi_R = subject_info(s).ITC_BPReref_R;
                    if use_hi_R == 1; expected_R = 'BPRerefHi'; else expected_R = 'BPRerefLow'; end
                end

                % --- Find file matching subject, channel, medication, and reref --
                if use_hi_L == 1 || use_hi_R == 1
                    filelist = dir(fullfile(ccc_dir, [char(subject), '*', pairname, '*', medname, '*BPRerefHi', '*.mat']));
                elseif use_hi_L == 0 || use_hi_R == 0
                    filelist = dir(fullfile(ccc_dir, [char(subject), '*', pairname,  '*',medname, '*BPRerefLow', '*.mat']));
                end

                if isempty(filelist)
                    warning('No CCC file found for %s / %s. skipping subject.', subject, pairname);
                    continue;
                end

                % load the CCC variable (adjust variable name if different)
                load(fullfile(filelist(1).folder, filelist(1).name), 'CCC', '-mat');

                cccMat = CCC.CCC;         % [nFreq x nTime]
                freqs = CCC.freqs;       % optional
                times = CCC.times;       % optional

                % initialize allCCC on first successful file
                if isempty(allCCC)
                    [nFreq, nTime] = size(cccMat);
                    allCCC = nan(numel(subjects), nFreq, nTime, numel(allPairs));
                end

                % store subject data for this pair
                allCCC(s,:,:,p) = squeeze(cccMat);
            end

            %% Save pairwise data for med
            if save_dta
                out_fname = sprintf('%s_AllSubs_CCC_%s.mat', pairname, medname);
                out_path = fullfile(data_dir, 'ccc', 'group_pair', out_fname);

                AllSubOnePairCCC = squeeze(allCCC(:,:,:,p)); % [sub x freq x time]
                if ~exist(fullfile(data_dir, 'ccc', 'group_pair'),'dir')
                    mkdir(fullfile(data_dir, 'ccc', 'group_pair'));
                end
                save(out_path, 'AllSubOnePairCCC', 'freqs', 'times');
                fprintf('Saved %s\n', out_path);
            end
        end
    end

    %% -------------------- Load group pair files --------------------
    if ismember('Load', pro)
        % Load the Single Pair data of MedOn and MedOff to big matrices
        % Output: MedOffCccDta and MedOnCccDta with dims [sub x freq x time x pair]
        MedOffCccDta = [];
        MedOnCccDta  = [];
        for p = 8%1:numel(allPairs)
            pairname = allPairs{p};
            fprintf('Loading pair: %s\n', pairname);

            filelist = dir(fullfile(data_dir, 'ccc', 'group_pair', [pairname, '*_AllSubs_CCC_*.mat']));
            if numel(filelist) < 2
                warning('Expected two files (MedOff/MedOn) for %s but found %d. Skipping.', pairname, numel(filelist));
                continue;
            end

            % find which is Off/On by filename containing medname or MedOff/MedOn convention
            % Simple approach: load both and assign by substring
            tmp1 = load(fullfile(filelist(1).folder, filelist(1).name), 'AllSubOnePairCCC', 'freqs', 'times');
            tmp2 = load(fullfile(filelist(2).folder, filelist(2).name), 'AllSubOnePairCCC', 'freqs', 'times');

            % assign based on medname appearing in filename
            if contains(filelist(1).name, 'MedOff') || ~contains(filelist(1).name, 'MedOn') && contains(filelist(2).name, 'MedOn')
                MedOffCccDta(:,:,:,p) = tmp1.AllSubOnePairCCC;
                MedOnCccDta(:,:,:,p)  = tmp2.AllSubOnePairCCC;
            else
                MedOffCccDta(:,:,:,p) = tmp2.AllSubOnePairCCC;
                MedOnCccDta(:,:,:,p)  = tmp1.AllSubOnePairCCC;
            end

            % % set freqs/times if not already defined
            % if ~exist('freqs','var') && isfield(tmp1,'freqs'); freqs = tmp1.freqs; end
            % if ~exist('times','var') && isfield(tmp1,'times'); times = tmp1.times; end
        end
    end

    %% -------------------- Single-channel (pair) paired t-test --------------------
    if ismember('TTest SC', pro)
        fprintf('Calculating paired t-test for each CCC pair\n');

        for p = 8%1:numel(allPairs)
            pairname = allPairs{p};
            fprintf('Processing pair: %s\n', pairname);

            % Extract per-pair matrices
            MedOffCccSC = squeeze(MedOffCccDta(:,:,:,p)); % [sub x freq x time]
            MedOnCccSC  = squeeze(MedOnCccDta(:,:,:,p));

            % select subjects that have both MedOn and MedOff
            idx_on = [subject_info.MedOn] & [subject_info.MedOff];
            idx_on(14) = 0;
            MedOn = MedOnCccSC(idx_on,:,:);
            idx_off = [true(1,8) false];
            MedOff = MedOffCccSC(idx_off,:,:);

            % optional manual row deletions (adapt to your dataset if needed)
            % if p <= 10; MedOn(9,:,:) = []; MedOff(9,:,:) = []; end

            [nSubj, nFreq, nTime] = size(MedOn);

            tvals = nan(nFreq, nTime);
            pvals = nan(nFreq, nTime);

            for f = 1:nFreq
                for t = 1:nTime
                    % paired t-test across subjects at each TF point
                    [~, pval, ~, stats] = ttest(squeeze(MedOn(:,f,t)), squeeze(MedOff(:,f,t)));
                    tvals(f,t) = stats.tstat;
                    pvals(f,t) = pval;
                end
            end

            % FDR correction (BH)
            [p_fdr, crit_p] = fdr_bh(pvals(:), 0.05, 'pdep', 'yes');
            p_fdr = reshape(p_fdr, size(pvals));
            sig_mask = pvals < 0.05;

            % Plotting: mean difference and significant mask
            meanDiff = squeeze(mean(MedOn - MedOff, 1, 'omitnan')); % [freq x time]
            particals = split(pairname, '_');
            chan1 = particals{1};
            chan2 = particals{2};

            f5 = figure('Name', sprintf('CCC %s MedOn - MedOff', pairname));
            set(f5,'Position',[159 50 1122 774.5000]);

            % Upper subplot
            subplot(2,1,1)
            plot(times(31:end), AVGECG.mean(31:end), 'Color', 'k'); hold on
            set(gca,'Position',[0.1300 0.5838 0.73 0.3])
            xline(0, "--k", 'LineWidth', 2);
            axis('tight')
            title(sprintf('Average ECG over all subjects'))
            ylabel('Amplitude')
            hold off

            subplot(2,1,2)
            imagesc(times(31:end), freqs(9:end), meanDiff(9:end,31:end)); axis xy;
            colormap('parula');
            col = colorbar;
            col.Label.String = 'Difference in CCC Values'; % Add title to colorbar
            clims = clim;
            hold on;
            contour(times(31:end), freqs(9:end), sig_mask(9:end,31:end), 1, 'linecolor', 'k', 'linewidth', 1.5);
            xline(0, "--k", 'LineWidth', 2);
            clim(clims);
            title(sprintf('CCC Difference %s - %s in MedOn - MedOff, df=%i, p<%.4g', chan1, chan2, stats.df, signif_thresh))
            xlabel('Time (s)') % Add x-label
            ylabel('Frequencies (Hz)') % Add y-label
            hold off

            % save figure
            gr5 = fullfile(results_dir, 'ccc', 'group_med', ['CCC_', pairname, '_MedOn-MedOff_TTest_p0.05.png']);
            exportgraphics(f5,gr5, 'Resolution', 300)

        end
    end

    %% -------------------- Cluster-level test across pairs or regions --------------------
    if ismember('TTest Clus', pro)
        % Example: perform regional cluster test by combining a set of pairs.
        % Here we demonstrate a simple region average approach similar to ITC ROI.
        % Define regions as groups of pairnames if relevant:
        CCC_regions = struct();
        % Example: CCC_regions.STN = ["LSTN_C3", "RSTN_C3"]; % adapt to your pair naming
        % CCC_regions.FrontalPairs = ["F3_C3", "F4_C4"]; % example

        regionNames = fieldnames(CCC_regions);
        for r = 1:numel(regionNames)
            regName = regionNames{r};
            thesePairs = CCC_regions.(regName);

            % find indices
            [~, pairIdx] = ismember(thesePairs, allPairs);
            pairIdx = pairIdx(pairIdx>0);

            % select subset of subjects with MedOn & MedOff
            idx = [subject_info.MedOn] & [subject_info.MedOff];

            MedOnSubset  = MedOnCccDta(idx,:,:,pairIdx);   % [sub x freq x time x pairs]
            MedOffSubset = MedOffCccDta(idx,:,:,pairIdx);

            % average across pairs
            MedOn_reg  = squeeze(mean(MedOnSubset, 4, 'omitnan'));  % [sub x freq x time]
            MedOff_reg = squeeze(mean(MedOffSubset, 4, 'omitnan'));

            % paired t-test per TF point
            [nSubj, nFreq, nTime] = size(MedOn_reg);
            tvals = nan(nFreq, nTime);
            pvals = nan(nFreq, nTime);
            for f = 1:nFreq
                for t = 1:nTime
                    [~, pval, ~, stats] = ttest(squeeze(MedOn_reg(:,f,t)), squeeze(MedOff_reg(:,f,t)));
                    tvals(f,t) = stats.tstat;
                    pvals(f,t) = pval;
                end
            end

            % FDR & plotting as above
            [~, ~, ~, p_fdr] = fdr_bh(pvals(:), 0.05, 'pdep', 'yes');
            p_fdr = reshape(p_fdr, size(pvals));
            sig_mask = p_fdr < 0.05;
            meanDiff = squeeze(mean(MedOn_reg - MedOff_reg,1,'omitnan'));

            figure; imagesc(times, freqs, meanDiff); axis xy; hold on;
            contour(times, freqs, sig_mask, 1, 'k', 'LineWidth', 1.5);
            title(sprintf('CCC Region %s MedOn - MedOff', regName));
            hold off;
        end
    end

end







if ismember('Group CCC Load Chan', steps)

    all_files = dir(fullfile(data_dir,'ccc', 'ss_chan', '*.mat'));
    valid_idx = ~startsWith({all_files.name}, '._');
    all_files = all_files(valid_idx);

    % Extract all channel names
    %channel_names = strings(numel(all_files), 1);
    for i = 1:numel(all_files)
        fname = all_files(i).name;
        parts = split(fname, '_');         % Example: 'sub01_LFP1.mat'
        channel_names{i} = {parts{2}, parts{3}};  % Get 'LFP1'
    end

    % Initialize output struct for averages
    CCC_GroupAvg = struct();

    % Loop over channels
    comps = fieldnames(CCCchans);
    for c = 1:numel(comps)
        compname = comps{c};
        chans = CCCchans.(compname);
        chanA = chans{1};
        chanB = chans{2};
        fprintf('Processing comp: %s - %s\n', chanA, chanB);

        % check for match
        files_comp = all_files(contains({all_files.name}, chanA) & contains({all_files.name}, chanB));  % ←←← CHANGED

        nSubjects = numel(files_comp);
        CCC_allSubs_cell = {};
        CCC_Perm_allSubs = {};

        for f = 1:numel(files_comp)
            load(fullfile(files_comp(f).folder, files_comp(f).name), 'CCC');
            CCC_allSubs_cell{end+1} = CCC.CCC;  % [nFreqs x nTimes]
            CCC_Perm_allSubs{end+1} = CCC.PermCcc;
            times = CCC.times;
            freqs = CCC.freqs;
            SR = CCC.SR;
        end

        % Convert to 3D matrix: [nSubjects x nFreqs x nTimes]
        nSubjects = numel(CCC_allSubs_cell);
        [nFreqs, nTimes] = size(CCC_allSubs_cell{1});
        [nPerm, nFqs, nTms] = size(CCC_Perm_allSubs{1});
        CCC_allSubs = zeros(nSubjects, nFreqs, nTimes);
        CCC_PermAvg_allsubs = zeros(nSubjects, nPerm, nFreqs, nTimes);

        for s = 1:nSubjects
            CCC_allSubs(s, :, :) = CCC_allSubs_cell{s};
            CCC_PermAvg_allsubs(s,:,:,:) = CCC_Perm_allSubs{s};
        end

        % Average over subjects
        CccAll_subavg = squeeze(mean(CCC_allSubs,1));  % freq x time
        PermCccAll_avg = squeeze(mean(CCC_PermAvg_allsubs,1));  % permutations x freq x time
        
        f8 = figure;
        histogram(CccAll_subavg,  'FaceColor','b');
        title(sprintf('Average CCC Histogram for %s - %s, med: %s', chanA, chanB, medname))
        ylabel('Number of Points')
        xlabel('CCC Values (0-1)')
        gr8 = fullfile(results_dir, 'ccc' ,'ccc_perm_distributions', ['CCC_', chanA, '_', chanB, '_', medname, '_n=', num2str(nSubjects), 'distibution_histogram.png']);
        exportgraphics(f8,gr8, 'Resolution', 300)

        f9 = figure;
        histogram(PermCccAll_avg,  'FaceColor','r');
        title(sprintf('Average CCC Perm Histogram for %s - %s, med: %s', chanA, chanB, medname))
        ylabel('Number of Points')
        xlabel('CCC Perm Values (0-1)')
        gr9 = fullfile(results_dir, 'ccc' ,'ccc_perm_distributions', ['CCC_Perm_', chanA, '_', chanB, '_', medname, '_n=', num2str(nSubjects), 'distibution_histogram.png']);
        exportgraphics(f9,gr9, 'Resolution', 300)

        f10 = figure;
        histogram(PermCccAll_avg, 'FaceColor','r');
        hold on
        histogram(CccAll_subavg,  'FaceColor','b');
        title(sprintf('Average CCC and Perm Histogram for %s - %s, med: %s', chanA, chanB, medname))
        ylabel('Number of Points')
        xlabel('CCC Values (0-1)')
        gr10 = fullfile(results_dir, 'ccc' ,'ccc_perm_distributions', ['CCC_vs_Perm_', chanA, '_', chanB, '_', medname, '_n=', num2str(nSubjects), 'distibution_histogram.png']);
        exportgraphics(f10,gr10, 'Resolution', 300)

        if ismember('Plot SubAvg CCC', steps)
            fprintf('Plot CCC Averages \n');

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
            imagesc(times(31:end),freqs(9:end),CccAll_subavg);axis xy;
            colormap('parula');
            xline(0, "--k", 'LineWidth', 2);
            col = colorbar;
            col.Label.String = 'CCC Values'; % Add title to colorbar
            xlabel('Time (s)') % Add x-label
            ylabel('Frequencies (Hz)') % Add y-label
            title(sprintf('Average CCC for %s - %s, med: %s, HP: %s', chanA, chanB, medname, Hz_dir))

            gr2 = fullfile(results_dir, 'ccc' ,'group', ['CCC_', chanA, '_', chanB, '_', medname, '_n=', num2str(nSubjects), '.png']);
            exportgraphics(f2,gr2, 'Resolution', 300)

        end

        if ismember('Plot SubAvg PermStats',steps)
            fprintf('Plot CCC Averages with permutation stats \n');

            nSubjects = numel(CCC_allSubs_cell);
            z_thresh = 1.64;  % z-score threshold
            min_cluster_size = 5;  % to ignore small noise clusters

            ClusterMaps = cell(1, nSubjects);  % Store binary cluster maps per subject

            for subj = 1:nSubjects
                % Load real and permuted CCC data
                real_map = CCC_allSubs_cell{subj};                     % [nFreq x nTime]
                perm_maps = CCC_Perm_allSubs{subj};                    % [nPerm x nFreq x nTime]

                % Get sizes
                [nPerm, nFreq, nTime] = size(perm_maps);

                % Compute mean and std across permutations
                perm_mean = squeeze(mean(perm_maps, 1));  % [nFreq x nTime]
                perm_std  = squeeze(std(perm_maps, 0, 1)); % [nFreq x nTime]

                % Z-score real data
                z_map = (real_map - perm_mean) ./ perm_std;

                % Threshold map
                bin_map = z_map > z_thresh;

                % Find clusters
                CC = bwconncomp(bin_map);

                % Compute cluster stats for real data
                cluster_stats_real = zeros(1, CC.NumObjects);
                for i = 1:CC.NumObjects
                    cluster_stats_real(i) = sum(z_map(CC.PixelIdxList{i}));
                end

                % Null distribution of max cluster stats
                max_cluster_stats_perm = zeros(1, nPerm);
                for p = 1:nPerm
                    perm_z = (squeeze(perm_maps(p,:,:)) - perm_mean) ./ perm_std;
                    perm_bin = perm_z > z_thresh;
                    perm_CC = bwconncomp(perm_bin);
                    perm_stats = zeros(1, perm_CC.NumObjects);
                    for k = 1:perm_CC.NumObjects
                        perm_stats(k) = sum(perm_z(perm_CC.PixelIdxList{k}));
                    end
                    if ~isempty(perm_stats)
                        max_cluster_stats_perm(p) = max(perm_stats);
                    else
                        max_cluster_stats_perm(p) = 0;
                    end
                end

                % Significance threshold
                cluster_thresh = prctile(max_cluster_stats_perm, 95);

                % Keep significant clusters
                signif_map = false(size(z_map));
                for i = 1:CC.NumObjects
                    if cluster_stats_real(i) > cluster_thresh && numel(CC.PixelIdxList{i}) >= min_cluster_size
                        signif_map(CC.PixelIdxList{i}) = true;
                    end
                end

                ClusterMaps{subj} = signif_map;  % store binary cluster map
            end

            nSubjects = numel(ClusterMaps);
            [nFreqs, nTimes] = size(ClusterMaps{1});
            nPerm = size(CCC_Perm_allSubs{1}, 1);  % assumed same for all

            z_thresh = 1.64;  % threshold for z-scoring individual permutations
            min_cluster_size = 5;  % discard tiny clusters

            % Step 1: Get the real group-level cluster count map
            RealGroupMap = zeros(nFreqs, nTimes);
            for subj = 1:nSubjects
                RealGroupMap = RealGroupMap + double(ClusterMaps{subj});
            end

            % Step 2: Start permutation loop
            PermGroupMaps = zeros(nPerm, nFreqs, nTimes);  % [nPerm x nFreqs x nTimes]

            for p = 1:nPerm
                TempMap = zeros(nFreqs, nTimes);  % One permuted group map

                for subj = 1:nSubjects
                    % Get permuted CCC map for this permutation
                    perm_map = squeeze(CCC_Perm_allSubs{subj}(p,:,:));  % [nFreq x nTime]

                    % Use subject's full perm distribution to z-score
                    perm_mean = squeeze(mean(CCC_Perm_allSubs{subj}, 1));
                    perm_std  = squeeze(std(CCC_Perm_allSubs{subj}, 0, 1));

                    perm_z = (perm_map - perm_mean) ./ perm_std;
                    bin_map = perm_z > z_thresh;

                    % Find clusters
                    CC = bwconncomp(bin_map);
                    perm_signif = false(nFreqs, nTimes);

                    for k = 1:CC.NumObjects
                        if numel(CC.PixelIdxList{k}) >= min_cluster_size
                            perm_signif(CC.PixelIdxList{k}) = true;
                        end
                    end

                    TempMap = TempMap + double(perm_signif);
                end

                PermGroupMaps(p,:,:) = TempMap;
            end

            % Step 3: Compute p-value map
            p_values = zeros(nFreqs, nTimes);
            for f = 1:nFreqs
                for t = 1:nTimes
                    real_val = RealGroupMap(f, t);
                    null_dist = squeeze(PermGroupMaps(:, f, t));
                    p_values(f, t) = mean(null_dist >= real_val);  % proportion of perms exceeding real
                end
            end

            % Step 4: Threshold to get final significant group cluster map
            alpha = 0.05;
            GroupClusterMap = p_values < alpha;  % logical map for contour

            % === Optional: You could also apply FDR correction here ===


            % %ChanMeanZscores_avg = squeeze(mean(squeeze(CCC.PERM.ZScoresAll(:,c,:,:)),1));  % SubjectxChannelxFreqxTime (Last two are CCC ZScores)
            % %ChanMeanPVal_avg = squeeze(mean(squeeze(CCC.PERM.PValAll(:,c,:,:)),1));  % SubjectxChannelxFreqxTime (Last two are CCC PVals)
            % PermCccAll_avg = squeeze(mean(squeeze(CCC.PERM.PermCccAll(:,c,:,:,:)),1)); %SubjectxChannelxPermutationxFreqxTime, Mean over all Subjects in one channel
            % CccAll_subavg = squeeze(mean(squeeze(CCC.CccAll(:,c,:,:)),1));
            % %RelCccAll_subavg = squeeze(mean(squeeze(CCC.RelCccAll(:,c,:,:)),1));

            % Step 1: Compute z-scores
            diff_sum_perm_mean_all = squeeze(mean(PermCccAll_avg,1)); % Mean of the permutation distribution
            diff_sum_perm_std_all = squeeze(std(PermCccAll_avg,1)); % Standard deviation of permutation distribution

            % diffPerm_mean(1,:,:) = diff_sum_perm_mean;
            % diffPerm_std(1,:,:)  = diff_sum_perm_std;

            zscores_all = (CccAll_subavg - diff_sum_perm_mean_all) ./ diff_sum_perm_std_all ;
            %zscores_perm = (PermCccData - diffPerm_mean) ./ diffPerm_std;
            p_orig_all = 2 * (1 - normcdf(zscores_all, 0, 1));
            signif_thresh =0.05;
            p_thresh_all = p_orig_all < signif_thresh;

            % [maxVal, maxInd] = max(CccAll_subavg(:));
            % [row, col] = ind2sub(size(CccAll_subavg), maxInd);
            % isTrueInMask = p_thresh_all(maxInd);
            % figure
            % imagesc(times, freqs, CccAll_subavg);axis xy;
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
            % imagesc(times(31:end), freqs(9:end), CccAll_subavg(9:end,31:end));axis xy;
            % contour(times, freqs, p_thresh_all,  1, 'linecolor', 'k', 'linewidth', 1.1)
            % hold on
            % rectangle('Position', [timeValues(1), freqValues(1), ...
            %     timeValues(2)-timeValues(1), freqValues(2)-freqValues(1)], ...
            %     'EdgeColor', 'r', 'LineWidth', 2);
            % hold off;
            %


            figure; % Sanity check that the distributions are normalized and overlapping so that my null hypothesis actually reflects my data
            histogram(PermCccAll_avg);
            hold on
            histogram(CccAll_subavg);
            figure; 
            histogram(zscores_all);

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
            imagesc(times(31:end),freqs(9:end),CccAll_subavg(9:end,31:end));axis xy;
            colormap('parula');
            colorbar;
            clims = clim;
            hold on
            contour(times, freqs, p_thresh_all,  1, 'linecolor', 'k', 'linewidth', 1.1)
            clim(clims);
            xline(0, "--k", 'LineWidth', 2);
            xlabel('Time (s)') % Add x-label
            ylabel('Frequencies (Hz)') % Add y-label
            title(sprintf(' Average CCC in %s, perm = %d, med = %s, p<%.4g, n=%d', ch_name, numPerms, medname, signif_thresh, nSubjects))

            gr6 = fullfile(results_dir, 'ccc', Hz_dir, 'group_perm' , ['AvgCCC_', char(ch_name), '_', medname, '_perm=', num2str(numPerms), '_n=', num2str(nSubjects), '.png']);
            exportgraphics(f6, gr6, 'Resolution', 300)

        end
        clear CCC_allSubs_cell CCC_Perm_allSubs CCC_allSubs CCC_PermAvg_allsubs
    end
end


% PLOT THE AVERAGE CCC PER SUBJECT
if ismember('Plot SubAvg CCC', steps)
    fprintf('Plot SubAvg CCC\n');
    if ismember('Calc Single Subject CCC', steps) == false

        fprintf('Loading TFR Data\n');
        pattern = fullfile(data_dir, 'CCC', ['CCC-AllSubs_',  medname, '*']);
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

        CccAll_subavg = squeeze(mean(squeeze(CCC.CccAll(:,c,:,:)),1));
        RelCccAll_subavg = squeeze(mean(squeeze(CCC.RelCccAll(:,c,:,:)),1));

        if plots
            f2=figure;
            set(f2,'Position',[1949 123 1023 785]);
            subplot(2,1,1)
            plot(times, AVGECG.mean', 'Color', 'k'); hold on
            set(gca,'Position',[0.1300 0.5838 0.77 0.3])
            xline(0, "--k", 'LineWidth', 2);
            title(sprintf('Average ECG over all subjects, medication: %s', medname))
            hold off
            subplot(2,1,2)
            imagesc(times,freqs,CccAll_subavg);axis xy;
            colormap('jet');
            xline(0, "--k", 'LineWidth', 2);
            colorbar;
            title(sprintf('Average CCC for %s, med: %s', channel, medname))

            gr2 = fullfile('F:\HeadHeart\2_results\ccc' , ['CCC_', channel, '_', medname,  '.png']);
            exportgraphics(f2,gr2, 'Resolution', 300)
        end
    end
end

% Now here is the Grand Average CCC over all Subs per Channel
if ismember('Plot SubAvg PermStats', steps)
    fprintf('Plot SubAvg CCC PermStats\n');

    if ismember('Calc Single Subject CCC', steps) == false
        fprintf('Loading CCC Data\n');
        pattern = fullfile(data_dir, 'CCC', ['CCC-AllSubs_',  medname, '*', '_perm=', '*']);
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
        PermCccAll_avg = squeeze(mean(squeeze(CCC.PermCcc(:,c,:,:,:)),1)); %SubjectxChannelxPermutationxFreqxTime, Mean over all Subjects in one channel
        CccAll_subavg = squeeze(mean(squeeze(CCC.CccAll(:,c,:,:)),1));
        RelCccAll_subavg = squeeze(mean(squeeze(CCC.RelCccAll(:,c,:,:)),1));

        % Step 1: Compute z-scores
        diff_sum_perm_mean_all = squeeze(mean(PermCccAll_avg,1)); % Mean of the permutation distribution
        diff_sum_perm_std_all = squeeze(std(PermCccAll_avg,1)); % Standard deviation of permutation distribution

        % diffPerm_mean(1,:,:) = diff_sum_perm_mean;
        % diffPerm_std(1,:,:)  = diff_sum_perm_std;

        zscores_all = (CccAll_subavg - diff_sum_perm_mean_all) ./ diff_sum_perm_std_all ;
        %zscores_perm = (PermCccData - diffPerm_mean) ./ diffPerm_std;
        p_orig_all = 2 * (1 - normcdf(zscores_all, 0, 1));
        signif_thresh =0.005;
        p_thresh_all = p_orig_all < signif_thresh;



        % figure; % Sanity check that the distributions are normalized and overlapping so that my null hypothesis actually reflects my data
        % histogram(PermCccAll_avg);
        % hold on
        % histogram(CccAll_subavg);

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
        imagesc(times(31:end),freqs,CccAll_subavg(:,31:end));axis xy;
        colormap('parula');
        colorbar;
        clims = clim;
        hold on
        contour(times, freqs, p_thresh_all,  1, 'linecolor', 'k', 'linewidth', 1.1)
        clim(clims);
        xline(0, "--k", 'LineWidth', 2);
        xlabel('Time (s)') % Add x-label
        ylabel('Frequencies (Hz)') % Add y-label
        title(sprintf(' Average CCC in %s, perm = %d, med = %s, p<%.4g', channel, numPerms, medname, signif_thresh))

        gr6 = fullfile('F:\HeadHeart\2_results\ccc\group_perm' , ['AvgCCC_', channel, '_', medname, '_perm=', num2str(numPerms), '.png']);
        exportgraphics(f6,gr6, 'Resolution', 300)



    end
end

if ismember('Parametric Stats CCC', steps)
    %% subject_level_PSI_cluster_permutation.m
    % Cluster-based permutation testing on subject-level PSI (CCC) maps
    % -----------------------------------------------------------------
    % This script assumes you have, for each subject and each channel-pair,
    % a .mat file that contains a struct named `CCC` and the field
    % `CCC.CCC` which is a matrix [nFreq x nTime] of PSI values (calculated
    % across trials already). The script loads these maps, optionally applies
    % Fisher z-transform (atanh), and runs a one-sample cluster-based
    % permutation test across subjects using FieldTrip.
    %
    % -----------------------------------------------------------------

    %% ---------------------- USER INPUTS (FILL IN) ----------------------
    
    % List of channel-pairs you want to analyze.
    comps = fieldnames(CCCchans); 

    % Do you want to apply Fisher z-transform (recommended)? true/false
    doFisherZ = false; % set false to skip atanh

    % Permutation parameters
    nPerm = 1000;          % number of permutations for cluster test (increase if you can)
    clusterAlpha = 0.005;   % cluster forming threshold
    alpha = 0.005;          % test alpha level

    % Whether to average across channel pairs or run each pair separately.
    % 'average' -> average PSI across ChanPairs for each subject
    % 'separate' -> run a separate test for each ChanPair
    pairMode = 'separate'; % options: 'average' or 'separate'  <-- choose

    %% ---------------------- END USER INPUTS ----------------------

    % Add FieldTrip to path if needed (uncomment and set your path)
    ft_defaults;

    nPairs = numel(comps);

    % Placeholder for loaded data: subj x freq x time x pair (optional)
    allPSI = [];

    % Validate the files
    all_files = dir(fullfile(data_dir,'ccc', 'ss_chan', '*.mat'));
    valid_idx = ~startsWith({all_files.name}, '._');
    all_files = all_files(valid_idx);

    for p = 1:nPairs
        compname = comps{p};
        chans = CCCchans.(compname);
        chanA = chans{1};
        chanB = chans{2};
        fprintf('Processing comp: %s - %s\n', chanA, chanB);

        % check for match
        files_comp = all_files(contains({all_files.name}, chanA) & contains({all_files.name}, chanB));
        nSubjects = numel(files_comp);

        for s = 1:nSubjects
            load(fullfile(files_comp(s).folder, files_comp(s).name), 'CCC');
            psiMat = CCC.CCC;  % [nFreqs x nTimes]
            surrogatePSIMat = CCC.PermCcc; %[nSurr x nFreqs x nTimes]
            times = CCC.times;
            freqs = CCC.freqs;
            SR = CCC.SR;

            % clip to avoid exact 0 or 1 values before atanh
            epsv = 1e-6;
            psiMat = min(max(psiMat, epsv), 1-epsv);

            % optional Fisher transform
            if doFisherZ
                zMat = atanh(psiMat);
            else
                zMat = psiMat;
            end

            if isempty(allPSI)
                [nFreq, nTime] = size(zMat);
                allPSI = nan(nSubjects, nFreq, nTime, nPairs); % subj x chan x freq x time x pair
            else
                % check consistent sizes (optional safety)
                if any([nFreq, nTime] ~= size(zMat))
                    error('Dimension mismatch for file %s', files_comp(s).name);
                end
            end

            %%% CHANGE HERE: Store with dummy channel dimension
            allPSI(s,:,:,p) = zMat;
            surrogatePSI(s,:,:,:,p) = surrogatePSIMat;
            
        end
    end
  
    [nSubj, nFreq, nTime, nPairs] = size(allPSI);
    nSurrogates = size(surrogatePSI,2);

    for p = 1:nPairs
        data_p = squeeze(allPSI(:,:,:,p));        % [nSub x nFreq x nTime]
        meanObs = mean(data_p,1);                % mean across subjects
        meanObs = squeeze(meanObs);              % [nFreq x nTime]
        
        surrogate_p = squeeze(surrogatePSI(:,:,:,:,p));  % [nSubj x nSurrogates x nFreq x nTime]
        surrogate_p = squeeze(mean(surrogatePSI,1));  % mean across subjects[nSurrogates x nFreq x nTime]

        % Compute percentile rank at each freq/time
        percentileMap = nan(nFreq, nTime);
        for f = 1:nFreq
            for t = 1:nTime
                percentileMap(f,t) = mean(surrogate_p(:,f,t) < meanObs(f,t));
                % fraction of surrogates below observed
            end
        end

        % Optional: define "empirical significance" at 95th percentile
        sigMaskEmpirical = percentileMap >= 0.95;

        % Plot
        figure;
        imagesc(times, freqs, meanObs);
        axis xy; colorbar;
        title(sprintf('Mean PSI - Pair %d (empirical percentile)', p));
        hold on;
        contour(times, freqs, sigMaskEmpirical, [1 1], 'LineColor','k', 'LineWidth',1.5);
    end
    
nBoot = 1000;  % number of bootstrap resamples

for p = 1:nPairs
    fprintf('Processing pair %d of %d\n', p, nPairs);
    
    data_p = squeeze(allPSI(:,:,:,p)); % [nSub x nFreq x nTime]

    % Compute mean across subjects
    meanMap = mean(data_p,1);  % [1 x nFreq x nTime]
    meanMap = squeeze(meanMap);

    % Bootstrap CI
    ciLow = nan(nFreq, nTime);
    ciHigh = nan(nFreq, nTime);
    
    for f = 1:nFreq
        for t = 1:nTime
            bootSamples = nan(nBoot,1);
            for b = 1:nBoot
                idx = randsample(nSubj, nSubj, true);  % resample subjects with replacement
                bootSamples(b) = mean(data_p(idx,f,t));
            end
            ciLow(f,t)  = prctile(bootSamples, 2.5);
            ciHigh(f,t) = prctile(bootSamples, 97.5);
        end
    end

    % Plot mean + CI (example for visualization)
    figure;
    imagesc(times, freqs, meanMap);
    axis xy;
    colorbar;
    title(sprintf('Mean PSI - Pair %d', p));
    xlabel('Time');
    ylabel('Frequency');

    % Optional: overlay regions where CI does not cross zero
    hold on;
    sigMask = (ciLow > 0);  % robust positive PSI
    contour(1:nTime, 1:nFreq, sigMask, [1 1], 'LineColor','k', 'LineWidth',1.5);
end


    for p = 1:nPairs
    fprintf('Processing channel pair %d of %d...\n', p, nPairs);
    
    % Extract data for this pair: [sub x freq x time]
    data_p = squeeze(allPSI(:,:,:,p));


    
   
   % pairStats = cell(nPairs,1);
   % 
   % for p = 1:nPairs
   %     fprintf('Running cluster test for pair %d of %d\n', p, nPairs);
   % 
   %     % data extraction
   %     data_p = squeeze(allPSI(:,:,:,p));  % size: [nSubj x nFreq x nTime]
   % 
   %     [nSubj, nFreq, nTime] = size(data_p);
   % 
   %     % preallocate
   %     zmap   = nan(nFreq, nTime);   % signed-rank z-values
   %     pvals  = nan(nFreq, nTime);   % raw p-values
   % 
   %     % loop over frequencies and times
   %     for f = 1:nFreq
   %         for t = 1:nTime
   %             vec = squeeze(data_p(:,f,t));   % subjects at (freq,time)
   %             try
   %                 [p,~,stats] = signrank(vec, 0, 'method','approx');   % test median vs 0
   %                 pvals(f,t) = p;
   %                 if isfield(stats,'zval')
   %                     zmap(f,t) = stats.zval;   % Wilcoxon z statistic
   %                 else
   %                     zmap(f,t) = NaN;
   %                 end
   %             catch
   %                 % signrank can error if all values are identical
   %                 pvals(f,t) = NaN;
   %                 zmap(f,t) = NaN;
   %             end
   %         end
   %     end
       
       % % ttest
       % mu = mean(data_p,1);            % mean over subjects
       % se = std(data_p,[],1) ./ sqrt(nSubj); % standard error
       % tmap = squeeze(mu ./ se);       % observed t-stat map [freq x time]
       % 
       % % thresholding 
       % 
       % df = nSubj - 1;
       % tcrit = tinv(1 - clusterAlpha/2, df); % two-sided threshold
       % sigMask = abs(tmap) > tcrit;          % logical mask of supra-threshold points
       % pvals = 2 * (1 - tcdf(abs(tmap), df));
       % pvals_pos = 1 - tcdf(tmap, df);
       % sigMask = pvals < 0.001;

       pvec = pvals(:);  % flatten 2D [freq x time] matrix
       [h, crit_p, adj_ci_cvrg, adj_p] = fdr_bh(pvec, 0.01, 'pdep', 'yes');  % Benjamini-Hochberg
       sigMaskFDR = reshape(h, size(pvals));

       f5 = figure;
       set(f5,'Position',[159 50 1122 774.5000]);

       % Upper subplot
       subplot(2,1,1)
       plot(times(31:end), AVGECG.mean(31:end)', 'Color', 'k'); hold on
       set(gca,'Position',[0.1300 0.5838 0.71 0.3])
       xline(0, "--k", 'LineWidth', 2);
       axis('tight')
       title(sprintf('Average ECG over all subjects, med: %s', medname))
       ylabel('Amplitude')
       hold off

       % Lower subplot
       subplot(2,1,2)
       imagesc(times, freqs, squeeze(mu)); %axis xy;
       colormap('parula');
       col = colorbar;
       col.Label.String = 'CCC Values'; % Add title to colorbar
       clims = clim;
       hold on;
       contour(times, freqs, sigMaskFDR, 1, 'linecolor', 'k', 'linewidth', 1.5);
       xline(0, "--k", 'LineWidth', 2);
       clim(clims);
       title(sprintf('CCC %s - %s FDR corrected, med: %s, p<0.001', chanA, chanB, medname))
       xlabel('Time (s)') % Add x-label
       ylabel('Frequencies (Hz)') % Add y-label
       hold off



       % identify clusters using 2D connected components (freq x time)
       CC = bwconncomp(sigMask, 4); % 4-connectivity: adjacent points
       nClusters = CC.NumObjects;
       clusterStats = zeros(1,nClusters);
       for c = 1:nClusters
           clusterStats(c) = sum(abs(tmap(CC.PixelIdxList{c}))); % cluster-level stat: sum of |t|
       end

       % --------------------------------------------------
       % 4) permutation testing (sign-flip)
       % --------------------------------------------------
       maxClusterDist = zeros(1,nPerm);

       for perm = 1:nPerm
           % randomly flip the sign of each subject
           flips = (rand(nSubj,1) > 0.5)*2 - 1;
           permData = data_p .* reshape(flips, [nSubj 1 1]);

           % compute permuted t-map
           mu_perm = mean(permData,1);
           se_perm = std(permData,[],1) ./ sqrt(nSubj);
           tmap_perm = squeeze(mu_perm ./ se_perm);

           % threshold
           sigMask_perm = abs(tmap_perm) > tcrit;
           CC_perm = bwconncomp(sigMask_perm, 4);

           if CC_perm.NumObjects > 0
               clusterVals = zeros(1, CC_perm.NumObjects);
               for c = 1:CC_perm.NumObjects
                   clusterVals(c) = sum(abs(tmap_perm(CC_perm.PixelIdxList{c})));
               end
               maxClusterDist(perm) = max(clusterVals);
           else
               maxClusterDist(perm) = 0;
           end
       end

       % --------------------------------------------------
       % 5) compute cluster-level p-values
       % --------------------------------------------------
       cluster_pvals = ones(1, nClusters);
       for c = 1:nClusters
           cluster_pvals(c) = mean(maxClusterDist >= clusterStats(c));
       end

       % significant clusters
       sigClusters = find(cluster_pvals < alpha);
       fprintf('Pair %d: found %d significant clusters\n', p, numel(sigClusters));

       % --------------------------------------------------
       % 6) store results
       % --------------------------------------------------
       pairStats{p}.clusters      = CC;
       pairStats{p}.clusterStats  = clusterStats;
       pairStats{p}.cluster_pvals = cluster_pvals;
       pairStats{p}.sigClusters   = sigClusters;
       pairStats{p}.T_obs         = tmap;
       pairStats{p}.T_thresh      = tcrit;
   end



    %% Plotting
    figure('Name', ['Cluster results - ' thisLabel], 'NumberTitle','off');
    imagesc(freqData.time, freqData.freq, out.meanMap);
    axis xy; xlabel('Time (s)'); ylabel('Frequency (Hz)');
    title(['Mean zPSI across subjects - ' thisLabel]);
    colorbar;
    hold on;

    if isfield(stat,'posclusters') && ~isempty(stat.posclusters)
        for c = 1:length(stat.posclusters)
            if stat.posclusters(c).prob < alpha
                mask = stat.posclusterslabelmat == c;
                contour(freqData.time, freqData.freq, mask, [1 1], 'LineColor','k', 'LineWidth', 1.5);
            end
        end
    end
    if isfield(stat,'negclusters') && ~isempty(stat.negclusters)
        for c = 1:length(stat.negclusters)
            if stat.negclusters(c).prob < alpha
                mask = stat.negclusterslabelmat == c;
                contour(freqData.time, freqData.freq, mask, [1 1], 'LineColor','w', 'LineWidth', 1.5);
            end
        end
    end
end

fprintf('All done.\n');







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