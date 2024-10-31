%% Main modlue for HeadHeart 

% Author: Lisa Paulsen
% Contact: lisaspaulsen[at]web.de
% Created on: 1 October 2024
% Last update: 15 October 2024

%% REQUIRED TOOLBOXES
% Curve Fitting Toolbox
% Image Processing Toolbox 
% Signal Processing Toolbox
% Statistics and Machine Learning Toolbox

clear all
close all

%% Directory SetUp
% set the directory to best fit your path 

% Define if using Windows or Mac
windows = true; % if true then Windows if false then Mac

if windows
    % Lisas Path Lab on Windows
    data_dir = 'F:\HeadHeart\0_data';
    results_dir = 'F:\HeadHeart\2_results';
    plots_dir = 'F:\HeadHeart\3_plots';
else
    % Lisas Path Mac
    data_dir = '/Volumes/LP3/HeadHeart/0_data';
    results_dir = '/Volumes/LP3/HeadHeart/2_results';
    plots_dir = '/Volumes/LP3/HeadHeart/3_plots';
end

%% Analysis SetUp 

% initialize subjects 
% be aware that for sub 45 and 60 only MEDON is available
subjects_MedOff = {'SG041', 'SG043', 'SG044', 'SG046', 'SG047', 'SG050', 'SG052', 'SG056'};
subjects_MedOn = {'SG041', 'SG043', 'SG044', 'SG045', 'SG046', 'SG047', 'SG050', 'SG052', 'SG056', 'SG060'};
subjects = subjects_MedOn;

MedFlag = 'MedOn'; % MedOn or MedOff

%% PREPROCESSING 
 disp('********** Preprocessing the Data ************');
preprocess_physiological(subjects, data_dir, results_dir, MedFlag)


%% TIME FREQUENCY DECOMPOSITION



%% ERP 
erp_calculation(subjects, data_dir, results_dir, MedFlag)

 









