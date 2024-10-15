% function [] = time_frequency_decomp(subjects, data_dir, results_dir, plots_dir)

%% Time Frequency Decomposition for HeadHeart 

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
clear all
close all

% Define if using Windows or Mac 
windows = true; % if windows true then windows if false than mac

% Define if plots are to be shown
show_plots = true;

% Define preprocessing steps to perform
steps = {'Load Data', 'Feature Extraction ECG', 'Feature Extraction EEG'}; 

% Define folder variables 
preprocessed_name = 'preproc';  % preprocessed folder (inside derivatives)
averaged_name = 'avg';  % averaged data folder (inside preprocessed)
feature_name = 'features';  % feature extraction folder (inside derivatives)

% Create the features data folder if it does not exist
for subject = subjects
    subject_features_folder = fullfile(data_dir, feature_name, ...
                                       ['sub-', subject{1}]);
    if ~exist(subject_features_folder, 'dir')
        mkdir(subject_features_folder);
    end
end

avg_features_folder = fullfile(data_dir, feature_name, averaged_name);
if ~exist(avg_features_folder, 'dir')
    mkdir(avg_features_folder);
end

% Define parameters for time-frequency analysis of both ECG and EEG data
window_length = 2;  % 2s window  % Length of the window for smoothing
overlap = 0.5;  % 50% overlap    % Overlap of the windows for smoothing
mirror_length = 180;  % Length of the mirror extension for symmetric padding

% Define parameters for time-frequency analysis of ECG data
sampling_frequency_ibi = 1;  % Hz IBI and HR data are already sampled at 1 Hz from the preprocessing

% Define low and high frequency bands for HRV analysis
lf_band = [0.04, 0.15];
hf_band = [0.15, 0.4];

% Define parameters for time-frequency analysis of EEG data
sampling_frequency_eeg = 200;  % Hz EEG data will be downsampled to 100 Hz
frequencies = 0.5:0.5:50;  % Resolution 0.5 Hz

% Define frequency bands of interest
bands = struct('delta', [0.3, 4], 'theta', [4, 8], 'alpha', [8, 13], ...
               'beta', [13, 30], 'gamma', [30, 45]);

% Create color palette for plots
colors.ECG.IBI = "#F0E442";    % Yellow
colors.ECG.HRV = "#CC79A7";   % Pink
colors.ECG.LF_HRV = "#E69F00";  % Light Orange
colors.ECG.HF_HRV = "#D55E00";   % Dark Orange

colors.EEG.delta = "#F0E442";  % Yellow
colors.EEG.theta = "#D55E00";    % Dark Orange
colors.EEG.alpha = "#CC79A7"; % Pink
colors.EEG.beta = "#56B4E9";   % Light Blue
colors.EEG.gamma = "#009E73";   % Green

% Features for averaging across participants
features_averaging.ecg = {'ibi', 'hrv', 'lf-hrv', 'hf-hrv'};
features_averaging.eeg = {'delta', 'theta', 'alpha', 'beta', 'gamma'};

% Suppress excessive logging if using FieldTrip 
ft_defaults; % If using FieldTrip















