% function [] = time_frequency_decomp(subjects, data_dir, results_dir)

%% HRV Comparison for HeadHeart 

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
% - ECG: IBI, HRV, HF-HRV, LF-HRV
% - EEG & LFP: Power of delta, theta, alpha, beta, gamma bands for all electrodes 

% Steps:
% 1. LOAD DATA
% 2. FEATURE EXTRACTION ECG
%   2a. Calculate HRV features from IBI data
%   2b. Save HRV features in a tsv file
%   2c. Plot HRV features
% 3. FEATURE EXTRACTION EEG
%   3a. Calculate the power of all EEG frequency bands
%   3b. Calculate the mean power of the EEG frequency bands for a region of interest (ROI)
%   3c. Save EEG power features in a tsv file
%   3d. Plot EEG power features
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
% patients with arrithmyia have been excluded after their ECG was
% investigated
GoodHeart = 0;

% get the channel info into the shape of cells
AllSubsChansRaw = cellfun(@(x) strsplit(x, ', '), {subject_info.channels_raw}, 'UniformOutput', false);
AllSubsChansStn = cellfun(@(x) strsplit(x, ', '), {subject_info.channels}, 'UniformOutput', false);
AllSubsOnlyStn = cellfun(@(x) strsplit(x, ', '), {subject_info.STN}, 'UniformOutput', false);

% filter which subjects and which channels you want

if MedOff == true & allsubs == true & GoodHeart % All Subs that are MedOff with good Heart
    subjects = string({subject_info([subject_info.MedOff] == 1& [subject_info.goodHeart_MedOff] == 1).ID});
elseif MedOn == true & allsubs == true & GoodHeart == 1 % All Subs that are MedOn with good Heart
    subjects = string({subject_info([subject_info.MedOn] == 1& [subject_info.goodHeart_MedOn] == 1).ID});
elseif MedOn == true & newsubs == true % Only New Subs that are MedOn
    subjects = string({subject_info([subject_info.new] == 1 & [subject_info.MedOn] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.new] == 1 & [subject_info.MedOn] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.new] == 1 & [subject_info.MedOn] == 1);
    FltSubsOnlyStn = AllSubsOnlyStn([subject_info.new] == 1 & [subject_info.MedOn] == 1);
elseif MedOff == true & newsubs == true  % Only New Subs that are MedOff
    subjects = string({subject_info([subject_info.new] == 1 & [subject_info.MedOff] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.new] == 1 & [subject_info.MedOff] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.new] == 1 & [subject_info.MedOff] == 1);
    FltSubsOnlyStn = AllSubsOnlyStn([subject_info.new] == 1 & [subject_info.MedOff] == 1);
elseif MedOn == true & oldsubs == true  % Only Old Subs that are MedOn
    subjects = string({subject_info([subject_info.new] == 0 & [subject_info.MedOn] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.new] == 0 & [subject_info.MedOn] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.new] == 0 & [subject_info.MedOn] == 1);
    FltSubsOnlyStn = AllSubsOnlyStn([subject_info.new] == 0 & [subject_info.MedOn] == 1);
elseif MedOff == true & oldsubs == true  % Only Old Subs that are MedOff
    subjects = string({subject_info([subject_info.new] == 0 & [subject_info.MedOff] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.new] == 0 & [subject_info.MedOff] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.new] == 0 & [subject_info.MedOff] == 1);
    FltSubsOnlyStn = AllSubsOnlyStn([subject_info.new] == 0 & [subject_info.MedOff] == 1);
elseif MedOn == true & allsubs == true  % All Subs that are MedOn
    subjects = string({subject_info([subject_info.MedOn] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.MedOn] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.MedOn] == 1);
    FltSubsOnlyStn = AllSubsOnlyStn([subject_info.MedOn] == 1);
elseif MedOff == true & allsubs == true % All Subs that are MedOff
    subjects = string({subject_info([subject_info.MedOff] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.MedOff] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.MedOff] == 1);
    FltSubsOnlyStn = AllSubsOnlyStn([subject_info.MedOff] == 1);
end
%=========================================================================

% Define if plots are to be shown
show_plots = true;


% Define folder variables 
averaged_name = 'avg';  % averaged data folder (inside preprocessed)
feature_name = 'features';  % feature extraction folder (inside derivatives)

if MedOn == true
    medname = 'MedOn';
elseif MedOff == true
    medname = 'MedOff';
end


%% ============================ 1. LOAD DATA =============================
disp("************* Starting HRV IBI and HR Stats *************");


%fprintf('Loading Data of  subject %s number %i of %i\n', subject, sub, numel(subjects));

% Load MEd ON
pattern = fullfile(data_dir, 'features/avg/',['Averages_HRV-IBI-HR_', '*' ,'_Rest_nsub=11','*']);
files = dir(pattern);
filename = fullfile(files(1).folder, files(1).name);
load(filename, 'HRV', 'IBI', 'HR');
AllHR.MedOn = HR.MedOn;
AllIBI.MedOn = IBI.MedOn;
AllHRV.MedOn = HRV.rmssd_avg.MedOn;

% %Load Med Off
% pattern = fullfile(data_dir, 'features/avg/',['Averages_HRV-IBI-HR_', '*' ,'_Rest_nsub=11','*']);
% files = dir(pattern);
% filename = fullfile(files(1).folder, files(1).name);
% load(filename, 'HRV', 'IBI', 'HR');
% AllHR.MedOn = HR.MedOn;
% AllIBI.MedOn = IBI.MedOn;
% AllHRV.MedOn = HRV.rmssd_avg.MedOn;



% plot(HRV_MedOn.rmssd_avg, 'k')
% hold on
% plot(HRV_MedOff.rmssd_avg, 'b')
% legend('MedOn', 'MedOff')
% label(['HRV RMSSD: t(' num2str(stats.df) ')=' num2str(stats.tstat) ', p ='  num2str(p)])

%% Prep IBI AND HR DATA 

%Extract themeans over subject
for num = 1:numel(IBI.(medname))
    % Mean over Subjects IBI 
    IBI.SubMean(1,num) = mean(IBI.(medname){num}); 
    IBI.SubMean(1,num) = mean(IBI.(medname){num});
    % Mean over Subjects HR 
    HR.SubMean(1,num) = mean(HR.(medname){num});
    HR.SubMeanFreqs(1,num) = HR.SubMean(1,num)/60;
end

% Calculate the mean over all subjects
IBI.MedOffMean = mean(IBI.MedOffSubMean); IBI.MedOnMean = mean(IBI.MedOnSubMean);
HR.MedOffMean = mean(HR.MedOffSubMean); HR.MedOnMean = mean(HR.MedOnSubMean);

%% Stats
% Paired TTest MedOn vs. Med Off HRV
[h_hrv, p_hrv, ci_hrv, stats_hrv] = ttest2(HRV.rmssd_avg.MedOn, HRV.rmssd_avg.MedOff, 'Tail','right' , Vartype='equal');
%cohens_d = mean(HRV.rmssd_avg.MedOn-HRV.rmssd_avg.MedOff)/std(HRV.rmssd_avg.MedOn-HRV.rmssd_avg.MedOff);
sprintf(['t(' num2str(stats_hrv.df) ')=' num2str(stats_hrv.tstat) ', p='  num2str(p_hrv) ]) %', d=' num2str(cohens_d)])
 
% Paired TTest MedOn vs. Med Off IBI
[h_ibi, p_ibi, ci_ibi, stats_ibi] = ttest2(IBI.MedOnSubMean, IBI.MedOffSubMean, 'Tail', 'left', Vartype='equal');
cd_ibi = mean(IBI.MedOnSubMean-IBI.MedOffSubMean)/std(IBI.MedOnSubMean-IBI.MedOffSubMean);
sprintf(['t(' num2str(stats_ibi.df) ')=' num2str(stats_ibi.tstat) ', p='  num2str(p_ibi) ', d=' num2str(cd_ibi)])

% Paired TTest MedOn vs. Med Off HR
[h_hr, p_hr, ci_hr, stats_hr] = ttest2(HR.MedOnSubMean, HR.MedOffSubMean, 'Tail', 'right', Vartype='equal');
cd_hr = mean(HR.MedOnSubMean-HR.MedOffSubMean)/std(HR.MedOnSubMean-HR.MedOffSubMean);
sprintf(['t(' num2str(stats_hr.df) ')=' num2str(stats_hr.tstat) ', p='  num2str(p_hr) ', d=' num2str(cd_hr)])

%% Plot 
% Plot the HRV 
f1 = figure;
data = [IBI.MedOnSubMean', IBI.MedOffSubMean']*1000;
face=[240/256, 130/256, 0; 30/256, 160/256, 200/256;];
violin(data,'facecolor',face,'facealpha',0.5,'mc',[]);
ylabel('IBI Length (in ms)');
title(['IBI: t(' num2str(stats_ibi.df) ')=' num2str(stats_ibi.tstat) ', p='  num2str(p_ibi) ', d=' num2str(cd_ibi)])
set(gca, 'XTick', []);

for i = 1:size(data, 2)  % Loop through each group
    x = repmat(i, size(data, 1), 1);  % X positions for the data points
    scatter(x, data(:,i), 'filled','MarkerFaceColor',face(i,:),  'HandleVisibility', 'off');% Scatter plot of the data points  
end 

for j = 1:size(data, 1)  % Loop through each individual
    plot(1:size(data, 2), data(j,:), ':ok', 'HandleVisibility', 'off');  % Line plot connecting data points across groups
end

% Save the Plot
gr1 = fullfile(results_dir, feature_name, ['ViolinPlot_IVI_', num2str(numel(subjects.goodHeartMOff)),'_subjects_MedOnvs.MedOff.png']);
try
    exportgraphics(f1, gr1, "Resolution", 300);
catch ME
    warning("Failed to save the plot: %s", ME.message);
end

clear data

f2 = figure;
data = [HRV.rmssd_avg.MedOn', HRV.rmssd_avg.MedOff']*1000;
face=[240/256, 130/256, 0; 30/256, 160/256, 200/256;];
violin(data,'facecolor',face,'facealpha',0.5,'mc',[]);
ylabel('HRV Length (in ms)');
title(['HRV RMSSD: t(' num2str(stats_hrv.df) ')=' num2str(stats_hrv.tstat) ', p='  num2str(p_hrv)])  %', d=' num2str(cohens_d)
set(gca, 'XTick', []);

for i = 1:size(data, 2)  % Loop through each group
    x = repmat(i, size(data, 1), 1);  % X positions for the data points
    scatter(x, data(:,i), 'filled','MarkerFaceColor',face(i,:),  'HandleVisibility', 'off');% Scatter plot of the data points  
end 


for j = 1:size(data, 1)  % Loop through each individual
    plot(1:size(data, 2), data(j,:), ':ok', 'HandleVisibility', 'off');  % Line plot connecting data points across groups
end

% Save the Plot
gr2 = fullfile(results_dir, feature_name, ['ViolinPlot_HRV_', num2str(numel(subjects.goodHeartMOff)),'_subjects_MedOnvs.MedOff.png']);
try
    exportgraphics(f2, gr2, "Resolution", 300);
catch ME
    warning("Failed to save the plot: %s", ME.message);
end

clear data

f3 = figure;
data = [HR.MedOnSubMean', HR.MedOffSubMean'];
face=[240/256, 130/256, 0; 30/256, 160/256, 200/256;];
violin(data,'facecolor',face,'facealpha',0.5,'mc',[]);
ylabel('HR Length (in ms)');
title(['HR: t(' num2str(stats_hr.df) ')=' num2str(stats_hr.tstat) ', p='  num2str(p_hr) ', d=' num2str(cd_hr)])
set(gca, 'XTick', []);

for i = 1:size(data, 2)  % Loop through each group
    x = repmat(i, size(data, 1), 1);  % X positions for the data points
    scatter(x, data(:,i), 'filled','MarkerFaceColor',face(i,:),  'HandleVisibility', 'off');% Scatter plot of the data points  
end 

for j = 1:size(data, 1)  % Loop through each individual
    plot(1:size(data, 2), data(j,:), ':ok', 'HandleVisibility', 'off');  % Line plot connecting data points across groups
end

% Save the Plot
gr3 = fullfile(results_dir, feature_name, ['ViolinPlot_HR_', num2str(numel(subjects.goodHeartMOff)),'_subjects_MedOnvs.MedOff.png']);
try
    exportgraphics(f3, gr3, "Resolution", 300);
catch ME
    warning("Failed to save the plot: %s", ME.message);
end




