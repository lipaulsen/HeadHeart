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

% Define if plots are to be shown
show_plots = true;


% Define folder variables 
averaged_name = 'avg';  % averaged data folder (inside preprocessed)
feature_name = 'features';  % feature extraction folder (inside derivatives)

% Initialize how many subjects you will be looking at 
n_sub_medon = 6;
n_sub_medoff = 6;


%% ============================ 1. LOAD DATA =============================
disp("************* STARTING Feature Extraction  *************");


fprintf('Loading Data of  subject %s number %i of %i\n', subject, sub, numel(subjects));

% Load Med On
subject_data = fullfile(data_dir, feature_name, averaged_name, ['Averages_', num2str(n_sub_medon), '_subjects_MedOn_Rest.mat']);
load(subject_data, 'HRV');
% give new name
HRV_MedOn = HRV;

% Load Med Off
subject_data = fullfile(data_dir, feature_name, averaged_name, ['Averages_', num2str(n_sub_medoff), '_subjects_MedOff_Rest.mat']);
load(subject_data, 'HRV');
% give new name
HRV_MedOff = HRV;

plot(HRV_MedOn.rmssd_avg, 'k')
hold on
plot(HRV_MedOff.rmssd_avg, 'b')
legend('MedOn', 'MedOff')
label(['HRV RMSSD: t(' num2str(stats.df) ')=' num2str(stats.tstat) ', p ='  num2str(p)])

[h, p, ci, stats] = ttest2(HRV_MedOn.rmssd_avg, HRV_MedOff.rmssd_avg, 'Tail','left', Vartype='equal')
cohens_d = mean(HRV_MedOn.rmssd_avg-HRV_MedOff.rmssd_avg)/std(HRV_MedOn.rmssd_avg-HRV_MedOff.rmssd_avg)
sprintf(['t(' num2str(stats.df) ')=' num2str(stats.tstat) ', p='  num2str(p) ', d=' num2str(cohens_d)])

f1 = figure;
data = [HRV_MedOn.rmssd_avg', HRV_MedOff.rmssd_avg']*1000;
face=[240/256, 130/256, 0; 30/256, 160/256, 200/256;];
violin(data,'facecolor',face,'facealpha',0.5,'mc',[]);
ylabel('HRV Length (in ms)');
title(['HRV RMSSD: t(' num2str(stats.df) ')=' num2str(stats.tstat) ', p='  num2str(p) ', d=' num2str(cohens_d)])
set(gca, 'XTick', []);

for i = 1:size(data, 2)  % Loop through each group
    x = repmat(i, size(data, 1), 1);  % X positions for the data points
    scatter(x, data(:,i), 'filled','MarkerFaceColor',face(i,:),  'HandleVisibility', 'off');% Scatter plot of the data points  
end 


for j = 1:size(data, 1)  % Loop through each individual
    plot(1:size(data, 2), data(j,:), ':ok', 'HandleVisibility', 'off');  % Line plot connecting data points across groups
end

gr1 = fullfile(results_dir, feature_name, ['ViolinPlot_HRV_', num2str(length(subjects)),'_subjects_MedOnvs.MedOff.pdf']);
try
    exportgraphics(f1, gr1, "Resolution", 300, "padding", "figure");
catch ME
    warning("Failed to save the plot: %s", ME.message);
end








