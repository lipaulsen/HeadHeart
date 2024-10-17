function [] = preprocess_physiological(subjects, data_dir, results_dir)
%% Main Preprocessing for HeadHeart 

% Author: Lisa Paulsen
% Contact: lisaspaulsen[at]web.de
% Created on: 1 October 2024
% Last update: 15 October 2024

%% REQUIRED TOOLBOXES
% Image Processing Toolbox 
% Signal Processing Toolbox
% Statistics and Machine Learning Toolbox

% This function preprocesses all physiological data (ECG + EEG) from the HeadHeart
% Project.

% Inputs:     Raw EEG + ECG data in .mat files
%
%
% Outputs:    Preprocessed data (EEG, ECG, PPG) in tsv files (ECG & PPG) and fif files (EEG, before and after ICA)
%             Participant metadata in .mat files
%             Excluded participants (with too many bad channels) in a list
%             Plots of preprocessing steps

% Steps:
%    1. LOAD DATA
%    2. PREPROCESS DATA
%        2a. Cutting data
%        2b. Format data
%        2c. Preprocess ECG data 
%        2d. Preprocess EEG and LFP data & save everzthing to .mat file

%% ============= SET GLOABAL VARIABLES AND PATHS =========================
% clear all
% close all

% Define if using Windows or Mac 
windows = true; % if windows true then windows if false than mac

% Define if plots are to be shown
show_plots = true;

% Define if manual data cleaning is to be done
manual_cleaning = true;

% % Only analyze one subject when debug mode is on
% debug = false;
% if debug
%     subjects = subjects(1);
% end

% Define preprocessing steps to perform
steps = {'Cutting', 'Preprocessing ECG', 'Preprocessing EEG'}; %'Formatting',

% Define whether scaling of the ECG data should be done
scaling = false;
if scaling
    scale_factor = 0.01;
end

% Define power line frequency
powerline = 50; % Hz

% Define cutoff frequencies for bandfiltering EEG data
low_frequency = 0.5; % Hz
high_frequency = 100; % Hz
low_frequency_ica = 1; % Hz

% Define the percentage of bad epochs that should be tolerated
bad_epochs_threshold = 30; % percent

% Define the artifact threshold for epochs
artifact_threshold = 100; % microvolts

% Define if the first and last 2 seconds of the data should be cut off
% to avoid any potential artifacts at the beginning and end of the experiment
cut_off_seconds = 2; % sec

% Define the file paths and subject folders to be created
data_dir = fullfile(data_dir);
rawdata_name = 'raw'; % folder with the raw data
preprocessed_name = 'preproc'; % folder containing the preprocessed data
averaged_name = 'avg'; % averaged data (within preprocessed)
results_dir = fullfile(results_dir);

% Create preprocessed and results folder for each subject if it does not exist
for i = 1:length(subjects)
    subject = subjects{i};
    % preporcess folder
    subject_preprocessed_folder = fullfile(data_dir, preprocessed_name, sprintf('sub-%s', subject));
    if ~exist(subject_preprocessed_folder, 'dir')
        mkdir(subject_preprocessed_folder);
    end
    % results folder
    subject_results_folder = fullfile(results_dir, preprocessed_name, sprintf('sub-%s', subject));
    if ~exist(subject_results_folder, 'dir')
        mkdir(subject_results_folder);
    end

end

% Create folder for the averages after preprocessing
avg_preprocessed_folder = fullfile(data_dir, preprocessed_name, averaged_name);
if ~exist(avg_preprocessed_folder, 'dir')
    mkdir(avg_preprocessed_folder);
end

avg_results_folder = fullfile(results_dir, averaged_name);
if ~exist(avg_results_folder, 'dir')
    mkdir(avg_results_folder);
end

% Create color palette for plots
colors = struct();
colors.ECG = {'#F0E442', '#D55E00'};  % Yellow and dark orange
colors.EEG = {'#56B4E9', '#0072B2', '#009E73'};  % Light blue, dark blue, and green
colors.others = {'#FFFFFF', '#6C6C6C', '#000000'};  % White, gray, and black


%% ============================ 1. LOAD DATA =================================

for sub = 1:numel(subjects)
   
    % Extract the subject
    subject = subjects{sub};

    fprintf('Processing subject %s number %i of %i\n', subject, sub, numel(subjects));

    % Load subject data
    if windows
        subject_data = [data_dir, '\',  rawdata_name, '\', subject, '_MEDOFF_Rest.mat']; % Windows
    else
        subject_data = [data_dir, '/',  rawdata_name, '/', subject, '_MEDOFF_Rest.mat']; % MAC
    end
    load(subject_data, 'SmrData');

    % Load the ECG Peak data
    if windows
        subject_data = [data_dir, '\matfiles-cleaned-peaks\justEvs\ECGPeak_', subject, '_MEDOFF_Rest.mat']; % Windows
    else
        subject_data = [data_dir, '/matfiles-cleaned-peaks/justEvs/ECGPeak_', subject, '_MEDOFF_Rest.mat']; % MAC
    end

    load(subject_data, 'EvTits', 'EvData');
    % Inject the Event Data into the Data Struct
    SmrData.EvTits = EvTits;
    SmrData.EvData = EvData';

    % Define the path to subject preprocessed folder
    subject_preprocessed_folder = fullfile(data_dir, preprocessed_name, sprintf('sub-%s', subject));

    % Define the path to subject results folder
    subject_results_folder = fullfile(results_dir, preprocessed_name, sprintf('sub-%s', subject));

    % Create Metadata .mat file for the subject to review what
    % preprocessing steps were taken

    % Create participant file with relevant preprocessing metadata
    participant_metadata = struct(...
        'subject', subject, ...
        'start_time', datetime("now"), ...
        'steps', steps, ...
        'manual_cleaning_of_peaks', manual_cleaning, ...
        'sampling_rate', SmrData.SR, ...
        'power_line', powerline, ...
        'low_frequency', low_frequency, ...
        'high_frequency', high_frequency, ...
        'cut_off_seconds', cut_off_seconds);

    % Save the metadata for the participant in JSON File
    if windows
        save_metadata = [subject_preprocessed_folder, '\',  subject, '_preprocessing_metadata_', datestr(datetime('today'), 'yyyy_mm_dd'), '.json']; % Windowws
    else
        save_metadata =[subject_preprocessed_folder, '/',  subject, '_preprocessing_metadata_', datestr(datetime('today'), 'yyyy_mm_dd'), '.json']; % MAC
    end
    writestruct(participant_metadata, save_metadata);

    % if you want to save it as a .mat file then change the ending in
    % save_metadata and uncomment this line
    %save(save_metadata, 'participant_metadata');


    %% ======================== 2. PREPROCESS DATA ============================

    % __________________________ 2a. Cutting Data _____________________________
    if ismember('Cutting', steps)

        disp('********** Cutting data **********');

        % As we are only looking at resting data we are only cutting 2 seconds
        % from the beggining of EEG and ECG Recording as to minimize start and
        % end artifacts


        % Before we can cut we need to align our detected ECG Peaks with the
        % ECG data

        % Calculate the time points of the ECG data
        ecg_timepoints = (0:length(SmrData.WvData)-1) / SmrData.SR; %LOOK AT THIS AGAIN!!!

        % Initialize a new ECG vector with 0s
        SmrData.WvData(20,:) = zeros(1, length(ecg_timepoints));

        % Loop through R-peak times to fill alignedECG
        for i = 1:length(SmrData.EvData)
            % Calculate the index corresponding to the R-peak time
            [~, idx] = min(abs(ecg_timepoints - SmrData.EvData(i))); % Find the nearest index
            %SmrData.WvData(20, idx) = ecg_data_bandpass(idx); % Assign the ECG value at that index
            SmrData.WvData(20, idx) = SmrData.WvData(16,idx); % Assign the ECG value at that index
            SmrData.WvTits{20,1} = 'RPeaks_data';
        end

        % Cut off of Data
        if cut_off_seconds > 0
            disp(['Removing first and last ', num2str(cut_off_seconds), ' seconds of data...']);

            num_samples_cut = SmrData.SR*cut_off_seconds;

            % Cut EEG + ECG data
            SmrData.WvDataCleaned = SmrData.WvData(:, num_samples_cut+1:end - num_samples_cut);

        else
            disp('No cropping of the data selected');
        end

        if show_plots
            % Figure that shows the alignment of the cropped data of peak time
            % with ECG times
            max_time = 20000; % time in samples

            f1 = figure;
            plot(SmrData.WvDataCleaned(16, 1:max_time), 'b', 'DisplayName', 'Original ECG');
            hold on;
            plot(SmrData.WvDataCleaned(20, 1:max_time), 'r', 'DisplayName', 'Aligned ECG with NaNs');
            xticks(0:2000:max_time); 
            xticklabels((0:2000:max_time) / SmrData.SR); 
            xlabel('Time (s)');
            ylabel('ECG Signal (mV)');
            legend("raw ECG data", "aligned Peaks");
            title("Aligned ECG Data with R-Peaks for sub " + subject + " for ", num2str(0) + " till " + num2str(max_time/SmrData.SR) + " seconds");
        
            % Save Graphics
            if windows
                exportgraphics(f1, [subject_preprocessed_folder, '\', subject, '_PeakDetectionCheck.png'], "Resolution",300) % WINDOWS
            else
                exportgraphics(f1, [subject_preprocessed_folder, '/', '_PeakDetectionCheck.png'], "Resolution",300) % MAC
            end
        end
    end

    % ___________________________ 2b. Format Data _____________________________
    if ismember('Formatting', steps)

        disp('********** Formatting data **********');

        % Scale ECG data to decrease the amplitude of the peaks
        if scaling
            disp(['scaling ECG data by ', num2str(scale_factor)]);
            SmrData.WvDataCleaned(16,:) = SmrData.WvDataCleaned(16,:) * scale_factor;
        end
    end

    %% _____________________ 2c. Preprocess ECG Data ___________________________
    if ismember('Preprocessing ECG', steps)

        disp('********** Preprocessing ECG Data **********');

        % Removal of the DC Offset
        disp('Removing DC Offset...');
        SmrData.WvDataCleaned(16,:) = SmrData.WvDataCleaned(16,:)-mean(SmrData.WvDataCleaned(16,:));

        % Apply signal cleaning to ECG data
        % 50 Hz powerline filter and 2nd-order two-pass Butterworth filters (0.5 Hz high-pass, 30 Hz low-pass)
        disp('Cleaning ECG data...');

        % Design 50Hz Notch Filter using the fieldtrip toolbox;
        ecg_notch_ft = ft_preproc_dftfilter(SmrData.WvDataCleaned(16,:), SmrData.SR);

        if show_plots
            % Plot the different results on top of the raw data to see the
            % difference
            % Plot Raw vs. FT Notch Filter
            max_time = 20000; % time in samples

            f2 = figure;
            subplot(2,1,1);
            plot(SmrData.WvData(16,1:max_time), 'k','LineWidth',1)
            hold on
            plot(SmrData.WvDataCleaned(16,1:max_time),'g','LineWidth',1)
            title(['Raw ECG data and DC Offset for sub ', subject]);
            xticks(0:2000:max_time); 
            xticklabels((0:2000:max_time) / SmrData.SR); 
            xlabel('Time (s)');
            ylabel('Amplitude (mV)');
            legend('raw trace','DC Offset trace')

            % Plot Raw vs. After DC Offset
            subplot(2,1,2);
            plot(SmrData.WvData(16,1:max_time), 'k','LineWidth',1)
            hold on
            plot(ecg_notch_ft(1,1:max_time),'b','LineWidth',1)
            title(['Raw ECG data and Notch filter using Fieldtrip for sub ', subject]);
            xticks(0:2000:max_time); 
            xticklabels((0:2000:max_time) / SmrData.SR); 
            xlabel('Time (s)');
            ylabel('Amplitude (mV)');
            legend('raw trace','filtered trace')
            
            % Save Graphics
            if windows
                exportgraphics(f2, [subject_preprocessed_folder, '\', subject, '_ECG-DCOffset-vs-Notch.png'], "Resolution",300) % WINDOWS
            else
                exportgraphics(f2, [subject_preprocessed_folder, '/', subject, '_ECG-DCOffset-vs-Notch.png'], "Resolution",300) % MAC
            end
           
        end

        % Design 2nd order twopass Butterworth filter (0.5 Hz high-pass, 30 Hz low-pass)
        ecg_data_bandpass  = ft_preproc_bandpassfilter(ecg_notch_ft(1,:), SmrData.SR, [0.5 30],  2, 'but','twopass'); % two pass butterworth filter
        % save filtered data in SmrData struct
        SmrData.WvDataCleaned(21,:) = ecg_data_bandpass;
        SmrData.WvTits{21,1} = 'ECG_filtered';

        if show_plots
            % Plot the Raw ECG Data against the bandpass filtered Data
            % (0-30Hz)
            max_time = 20000; % time in samples
            f3 = figure;
            % Plot Raw vs. After bandpass Filter
            subplot(1,1,1);
            plot(SmrData.WvData(16,1:max_time), 'k','LineWidth',1)
            hold on
            plot(SmrData.WvDataCleaned(21,1:max_time),'r','LineWidth',1)
            title(['Raw ECG data and bandpass (0.5-30Hz) Filter for sub ', subject]);
            xticks(0:2000:max_time); 
            xticklabels((0:2000:max_time) / SmrData.SR);
            xlabel('Time (in s)');
            ylabel('Amplitude (mV)');
            legend('raw trace','bandpass Filter')

            % Save Graphics
            if windows
                exportgraphics(f3, [subject_preprocessed_folder, '\', subject, '_ECG-Filter(0-30Hz).png'], "Resolution",300) % WINDOWS
            else
                exportgraphics(f3, [subject_preprocessed_folder, '/', subject, '_ECG-Filter(0-30Hz).png'], "Resolution",300) % MAC
            end
        end

        % Calculate the IBI (inter-beat interval) from filtered ECG signal
        disp('Calculating IBI and HR...');

        % Since the data has been cut (including the ECG Peaks) I need to
        % extract the current ECG Peaks from the cropped data

        % Find all non-zero Values
        peakIndices = find(SmrData.WvDataCleaned(20, :) ~= 0);
        SmrData.ECGPeak(1,:) = ecg_timepoints(peakIndices);


        SmrData.ECGcomp(1,:) = diff(SmrData.ECGPeak(1,:)); % extract time differences in s between R-Peaks
        SmrData.ECGcompTits{1,:} = 'IBI';

        % Calculate Heart Rate from ECG
        SmrData.ECGcomp(2,:) = 60 ./ SmrData.ECGcomp(1,:);
        SmrData.ECGcompTits{2,:} = 'HR';

        if show_plots
            %max_time = numel(SmrData.WvDataCleaned(20,:))
            f4 = figure;
            % Plot Raw vs. After bandpass Filter
            subplot(2,1,1);
            plot(SmrData.ECGcomp(1,:), 'b','LineWidth',1)
            yline(1, "--k",'LineWidth', 1);
            title(['IBI Length for sub ', subject]);
            xlabel('IBI Samples over time');
            ylabel('IBI length (in s)');
            legend('IBI')

            % Plot Raw vs. After high and low pass Filter
            subplot(2,1,2);
            plot(SmrData.ECGcomp(2,:), 'k','LineWidth',1)
            yline(60, '--b', 'Lower HR range', 'LabelHorizontalAlignment', 'right');
            title(['Heartrate (HR) in BPM for sub ', subject]);
            xlabel('HR Samples over time');
            ylabel('HR in BPM');
            legend('HR')

            % Save Graphics
            if windows
                exportgraphics(f4, [subject_preprocessed_folder, '\', subject, '_ECG-IBI-HR.png'], "Resolution",300) % WINDOWS
            else
                exportgraphics(f4, [subject_preprocessed_folder, '/', subject, '_ECG-IBI-HR.png'], "Resolution",300) % MAC
            end
        end



        % frq_area = fft(SmrData.WvData(16, 1:10240));
        % figure
        % plot(SmrData.SR/10240*(0:10240-1),abs(frq_area),"LineWidth",3)
        % title("Complex Magnitude of fft Spectrum")
        % xlabel("f (Hz)")
        % ylabel("|fft(X)|")



    end

    %% _____________________ 2c. Preprocess LFP+EEG Data ___________________________
    if ismember('Preprocessing EEG', steps)

        disp('********** Preprocessing LFP and EEG Data **********');
%%%% DC OFFSET????

        % Apply signal cleaning to LFP and EEG data
        % 50 Hz powerline filter and 4th-order two-pass Butterworth filters (0.1 Hz high-pass, 10 Hz low-pass)
        disp('Cleaning EEG & LFP data...');

        % 50Hz Line-Noise Removal using the fieldtrip toolbox;
        lfp_notch_ft = ft_preproc_dftfilter(SmrData.WvDataCleaned(1:15,:), SmrData.SR);

        % Design Butterworth High and Low pass filters
        lfp_data_highpass  = ft_preproc_highpassfilter(lfp_notch_ft(1:15,:), SmrData.SR, 0.5,  4, 'but','twopass'); % two pass butterworth filter
        lfp_data_lowpass  = ft_preproc_lowpassfilter(lfp_data_highpass(1:15,:), SmrData.SR, 100,  4, 'but','twopass'); % two pass butterworth filter
        % lfp_data_lowpass_10  = ft_preproc_lowpassfilter(lfp_data_highpass(1:15,:), SmrData.SR, 10,  4, 'but','twopass'); % two pass butterworth filter
        
        % Save cleaned Data into SmrData.WvDataCleaned Struct
        SmrData.WvDataCleaned(1:15, :) = lfp_data_lowpass;

        % Visualize filtering

        if show_plots
            chan = 11; % select which LFP or EEG Channel you want to look at
            max_time = 20000; % time in samples

            f5 = figure;
            subplot(2,1,1);
            plot(SmrData.WvData(chan,1:max_time), 'k','LineWidth',1)
            hold on
            plot(lfp_data_highpass(chan,1:max_time),'r','LineWidth',1)
            title("Raw EEG data and High Pass Filter (0.5 Hz) for sub " + subject + " and channel " + SmrData.WvTits{chan});
            xticks(0:2000:max_time); 
            xticklabels((0:2000:max_time) / SmrData.SR);
            xlabel('Time (in s)');
            ylabel('Amplitude (microvolts)');
            legend('raw trace','filtered data')

            subplot(2,1,2);
            plot(SmrData.WvData(chan,1:max_time), 'k','LineWidth',1)
            hold on
            plot(SmrData.WvDataCleaned(chan,1:max_time),'r','LineWidth',1)
            title("Raw EEG data and High & Low Pass Filter  (0.5-100Hz) for sub " + subject + " and channel " + SmrData.WvTits{chan});
            xticks(0:2000:max_time); 
            xticklabels((0:2000:max_time) / SmrData.SR);
            xlabel('Time (in s)');
            ylabel('Amplitude (microvolts)');
            legend('raw trace','filtered data')

            % Save Graphics
            if windows
                exportgraphics(f5, [subject_preprocessed_folder, '\', subject, '_EEG_HP-LP-Filter.png'], "Resolution",300) % WINDOWS
            else
                exportgraphics(f5, [subject_preprocessed_folder, '/', subject, '_EEG_HP-LP-Filter.png'], "Resolution",300) % MAC
            end
        end
    end

    % Save the preprocessed Data in Smr.Data Struct 
    %newFileName = fullfile(data_dir, sprintf('ECGPeak_%s.mat', SmrData.FileName));
    save_path = [subject_preprocessed_folder, '\', subject,  '_preprocessed_MEDOFF_Rest.mat'];
    save(save_path, 'SmrData');

    disp(['Processed and saved: ', save_path]);

end