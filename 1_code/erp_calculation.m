% function [] = erp_calculation(subjects, data_dir, results_dir, plots_dir)

%% Calculating ERPs for HeadHeart 

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
% Outputs:   
% Plots with time-locked ERPs and summary statistics

% Steps:
% 1. LOAD DATA
% 2. Calculating ERPs
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

% Define if using Windows or Mac 
windows = true; % if windows true then windows if false than mac

% Define if plots are to be shown
show_plots = true;

% Define folder variables 
preprocessed_name = 'preproc';  % preprocessed folder (inside derivatives)
averaged_name = 'avg';  % averaged data folder (inside preprocessed)
erp_name = 'erp';  % feature extraction folder (inside derivatives)

% Create the  erp data folder if it does not exist
for subject = subjects
    subject_erp_folder = fullfile(data_dir, erp_name, ...
                                       ['sub-', subject{1}]);
    if ~exist(subject_erp_folder, 'dir')
        mkdir(subject_erp_folder);
    end
end

%% ============================ 1. LOAD DATA =============================
disp("************* STARTING TIME FREQUENCY DECOMPOSITION *************");

% Define which channels should be used
channels = "";

HEP_avg_all = zeros(1,numel(subjects))

for sub = 1:numel(subjects)

    % Extract the subject
    subject = subjects{sub};

    fprintf('Loading Data of  subject %s number %i of %i\n', subject, sub, numel(subjects));

    % Load subject data
    %if windows
    subject_data = fullfile(data_dir, preprocessed_name, ['sub-', subject], [subject, '_preprocessed_MEDOFF_Rest.mat']);
    %else
    %subject_data = [data_dir, '/',  preprocessed_name, '/sub-', subject, "/", subject, '_preprocessed_MEDOFF_Rest.mat']; % MAC
    %end
    load(subject_data, 'SmrData');

 %% ============================ 2. ERP  =============================

    % Define the SR 
    SR = SmrData.SR;

    % Define the time window around Heartbeat
    time_win = [0.1 0.6]; % in sec
    time_wins_samples = round(time_win*SR); % in samples

    % Extract R-Peak Indicies 
    r_peak_indices = find(SmrData.WvDataCleaned(20,:));

    % Create Dataframe to store the ERP data

    %% 2a. Extract Epochs 
    for peaks = 1:numel(r_peak_indices)

        % Extract current Peak
        r_peak_idx = r_peak_indices(1,peaks);











    end


