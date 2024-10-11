%function [] = preprocess_physiological(subjects, data_dir, results_dir)
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
 %        2c. Preprocess ECG and PPG data & save to tsv files
 %        2d. Preprocess EEG data & save to fif files

%% ============= SET GLOABAL VARIABLES AND PATHS =========================
%clear all 
%close all


% Define if plots are to be shown
show_plots = false; 

% Define if manual data cleaning is to be done
manual_cleaning = true;

% Only analyze one subject when debug mode is on
debug = false;
if debug
    subjects = subjects(1);
end

% Define preprocessing steps to perform
steps = {'Cutting', 'Preprocessing ECG', 'Preprocessing EEG'}; %'Formatting', 

% Define whether scaling of the ECG data should be done
scaling = true;
if scaling
    scale_factor = 0.01;
end
    
% Define power line frequency
powerline = 50; % Hz

% Define cutoff frequencies for bandfiltering EEG data
low_frequency = 0.1; % Hz
high_frequency = 10; % Hz
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
    subject_data = [data_dir, '/',  rawdata_name, '/', subject, '_MEDOFF_Rest.mat'];
    load(subject_data, 'SmrData');

    % Load the ECG Peak data
    subject_data = [data_dir, '/matfiles-with-peak/justEvs/ECGPeak_', subject, '_MEDOFF_Rest.mat'];
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
        'low_frequency_ica', low_frequency_ica, ...
        'scaling', scaling, ...
        'scale_factor', scale_factor, ...
        'cut_off_seconds', cut_off_seconds, ...
        'bad_epochs_threshold', bad_epochs_threshold);

    % Save the metadata for the participant in JSON File
    save_metadata =[subject_preprocessed_folder, '/',  subject, '_preprocessing_metadata_', datestr(datetime('today'), 'yyyy_mm_dd'), '.json'];
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
        SmrData.WvDataCropped = SmrData.WvData(:, num_samples_cut+1:end - num_samples_cut);

    else
        disp('No cropping of the data selected');
    end

    if show_plots
        % Figure that shows the alignment of the cropped data of peak time
        % with ECG times
        figure;
        plot(SmrData.WvDataCropped(16, :), 'b', 'DisplayName', 'Original ECG');
        hold on;
        plot(SmrData.WvDataCropped(20, :), 'r', 'DisplayName', 'Aligned ECG with NaNs');
        xlabel('Time (s)');
        ylabel('ECG Signal');
        legend;
        title('Aligned ECG Data with R-Peaks');
    end

end

% ___________________________ 2b. Format Data _____________________________
if ismember('Formatting', steps)

    disp('********** Formatting data **********');

    % Scale ECG data to decrease the amplitude of the peaks 
    if scaling
    disp(['scaling ECG data by ', num2str(scale_factor)]);
    SmrData.WvDataCropped(16,:) = SmrData.WvDataCropped(16,:) * scale_factor;
    end
end

%% _____________________ 2c. Preprocess ECG Data ___________________________
if ismember('Preprocessing ECG', steps)

    disp('********** Preprocessing ECG Data **********');
    
    % Removal of the DC Offset
    disp('Removing DC Offset...');
    SmrData.WvDataCropped(16,:) = SmrData.WvDataCropped(16,:)-mean(SmrData.WvDataCropped(16,:));

    % Apply signal cleaning to ECG data
    % 50 Hz powerline filter and 2nd-order two-pass Butterworth filters (0.5 Hz high-pass, 30 Hz low-pass)
    disp('Cleaning ECG data...');

    % Design 50Hz Notch Filter using the fieldtrip toolbox;
    ecg_notch_ft = ft_preproc_dftfilter(SmrData.WvDataCropped(16,:), SmrData.SR);

    if show_plots 
        % Plot the different results on top of the raw data to see the
        % difference
        % Plot Raw vs. FT Notch Filter
        subplot(3,1,1);
        plot(SmrData.WvData(16,1:10000), 'k','LineWidth',1)
        hold on
        plot(ecg_notch_ft(1,1:10000),'b','LineWidth',1)
        title(['Raw data and Notch filter using Fieldtrip for sub ', subject]);
        xlabel('Time');
        ylabel('Amplitude');
        legend('raw trace','filtered trace')

        % Plot Raw vs. After DC Offset
        subplot(3,1,2);
        plot(SmrData.WvData(16,1:10000), 'k','LineWidth',1)
        hold on
        plot(SmrData.WvDataCropped(16,1:10000),'g','LineWidth',1)
        title('Raw data and DC Offset');
        xlabel('Time');
        ylabel('Amplitude');
        legend('raw trace','DC Offset trace')

        % Plot Raw vs. After bandpass Filter
        % subplot(3,1,3);
        % plot(SmrData.WvData(16,1:10000), 'k','LineWidth',1)
        % hold on
        % plot(ecg_data_bandpass(1,1:10000),'r','LineWidth',1)
        % title('Raw data and bandpass (0.5-30Hz) Filter');
        % xlabel('Time');
        % ylabel('Amplitude');
        % legend('raw trace','bandpass Filter')
    end 

    % Design 2nd order twopass Butterworth filter (0.5 Hz high-pass, 30 Hz low-pass)
    ecg_data_bandpass  = ft_preproc_bandpassfilter(ecg_notch_ft(1,:), SmrData.SR, [0.5 30],  2, 'but','twopass'); % two pass butterworth filter
    % save filtered data in SmrData struct 
    SmrData.WvDataCropped(21,:) = ecg_data_bandpass;
    SmrData.WvTits{21,1} = 'ECG_filtered';
    
    if show_plots
        % Plot Raw vs. After bandpass Filter
        subplot(2,1,1);
        plot(SmrData.WvData(16,1:10000), 'k','LineWidth',1)
        hold on
        plot(ecg_data_bandpass(1,1:10000),'r','LineWidth',1)
        title(['Raw data and bandpass (0.5-30Hz) Filter for sub ', subject]);
        xlabel('Time');
        ylabel('Amplitude');
        legend('raw trace','bandpass Filter')

        % Plot Raw vs. After high and low pass Filter
        % subplot(2,1,2);
        % plot(SmrData.WvData(16,1:10000), 'k','LineWidth',1)
        % hold on
        % plot(ecg_data_lowpass(1,1:10000),'b','LineWidth',1)
        % title('Raw data and high and low pass (0.5-30Hz) Filter');
        % xlabel('Time');
        % ylabel('Amplitude');
        % legend('raw trace','high and low pass Filter')
    end

    % Calculate the IBI (inter-beat interval) from filtered ECG signal 
    disp('Calculating IBI and HR...');

    % Since the data has been cut (including the ECG Peaks) I need to
    % extract the current ECG Peaks from the cropped data 
  
    % Find all non-zero Values 
    peakIndices = find(SmrData.WvDataCropped(20, :) ~= 0);
    SmrData.ECGPeak(1,:) = ecg_timepoints(peakIndices);


    SmrData.ECGcomp(1,:) = diff(SmrData.ECGPeak(1,:)); % extract time differences in s between R-Peaks
    SmrData.ECGcompTits{1,:} = 'IBI';

    % Calculate Heart Rate from ECG
    SmrData.ECGcomp(2,:) = 60 ./ SmrData.ECGcomp(1,:);
    SmrData.ECGcompTits{2,:} = 'HR';

       

frq_area = fft(SmrData.WvData(16, 1:10240));
figure
plot(SmrData.SR/10240*(0:10240-1),abs(frq_area),"LineWidth",3)
title("Complex Magnitude of fft Spectrum")
xlabel("f (Hz)")
ylabel("|fft(X)|")



end

%% _____________________ 2c. Preprocess LFP+EEG Data ___________________________
if ismember('Preprocessing EEG', steps)

    disp('********** Preprocessing LFP and EEG Data **********');


    % Apply signal cleaning to LFP and EEG data
    % 50 Hz powerline filter and 4th-order two-pass Butterworth filters (0.1 Hz high-pass, 10 Hz low-pass)
    disp('Cleaning ECG data...');

    % 50Hz Line-Noise Removal using the fieldtrip toolbox;
    lfp_notch_ft = ft_preproc_dftfilter(SmrData.WvDataCropped(1:15,:), SmrData.SR);

    % Design Butterworth High and Low pass filters
    lfp_data_highpass  = ft_preproc_highpassfilter(lfp_notch_ft(1:15,:), SmrData.SR, 0.5,  4, 'but','twopass'); % two pass butterworth filter
    lfp_data_lowpass  = ft_preproc_lowpassfilter(lfp_data_highpass(1:15,:), SmrData.SR, 100,  4, 'but','twopass'); % two pass butterworth filter
    lfp_data_lowpass_10  = ft_preproc_lowpassfilter(lfp_data_highpass(1:15,:), SmrData.SR, 10,  4, 'but','twopass'); % two pass butterworth filter
      

    % Visualize filtering 
    
    if show_plots
        chan = 9; % select which LFP or EEG Channel you want to look at

        subplot(3,1,1);
        plot(SmrData.WvDataCropped(chan,1:10000), 'k','LineWidth',1)
        hold on
        plot(lfp_data_highpass(chan,1:10000),'r','LineWidth',1)
        title(['Raw data and notch Filter for sub ', subject]);
        xlabel('Time');
        ylabel('Amplitude');
        legend('raw trace','filtered data')

        subplot(3,1,2);
        plot(SmrData.WvDataCropped(chan,:), 'k','LineWidth',1)
        hold on
        plot(lfp_data_lowpass(chan,:),'r','LineWidth',1)
        title(['Raw data and High pass (0.1-100Hz) Filter for sub ', subject]);
        xlabel('Time');
        ylabel('Amplitude');
        legend('raw trace','filtered data')

        subplot(3,1,3);
        plot(SmrData.WvDataCropped(chan,:), 'k','LineWidth',1)
        hold on
        plot(lfp_data_lowpass_10(chan,:),'r','LineWidth',1)
        title(['Raw data and High & Low pass (0.1-100Hz) Filter for sub ', subject]);
        xlabel('Time');
        ylabel('Amplitude');
        legend('raw trace','filtered data')
    end







end

    % % Function definitions
    % function plot_peaks(cleaned_signal, peaks, time_range, plot_title, sampling_rate)
    %     % Plot ECG or PPG signal with peaks
    % 
    %     % Time vector in samples
    %     min_sample = floor(time_range(1) * sampling_rate);
    %     max_sample = floor(time_range(2) * sampling_rate);
    %     time = min_sample:max_sample;
    % 
    %     % Plot setup
    %     figure;
    %     selected_signal = cleaned_signal(min_sample:max_sample);
    %     selected_peaks = peaks(peaks >= min_sample & peaks <= max_sample);
    %     selected_signal = selected_signal / 1000;  % Convert mV to V
    % 
    %     % Plot ECG or PPG
    %     if contains(plot_title, 'ECG')
    %         linecolor = colors.ECG{2};
    %         circlecolor = colors.ECG{1};
    %     elseif contains(plot_title, 'PPG')
    %         linecolor = colors.PPG{2};
    %         circlecolor = colors.PPG{1};
    %     end
    % 
    %     plot(time, selected_signal, 'Color', linecolor);
    %     hold on;
    %     scatter(selected_peaks, selected_signal(selected_peaks - min_sample + 1), 'MarkerFaceColor', circlecolor, 'MarkerEdgeColor', circlecolor);
    %     xlabel('Time (s)');
    %     ylabel(plot_title);
    %     title(plot_title);
    %     if show_plots
    %         hold off;
    %         grid on;
    %     end
    % end
    % 
    % function [filtered_data, epochs, reject_log] = preprocess_eeg(raw_data, low_freq, high_freq, autoreject)
    %     % Preprocess EEG data using MATLAB EEGLAB
    % 
    %     % Remove powerline noise using a notch filter
    %     disp('Removing line noise...');
    %     filtered_data = pop_eegfiltnew(raw_data, [], powerline, 1, 0, [], 1); % Band-stop filter
    % 
    %     % Re-reference to average
    %     disp('Rereferencing to average...');
    %     filtered_data = reref(filtered_data, []);
    % 
    %     % Bandpass filter
    %     disp('Filtering data...');
    %     filtered_data = pop_eegfiltnew(filtered_data, low_freq, high_freq);
    % 
    %     % Segment data into epochs (10s epochs)
    %     disp('Segmenting data into epochs...');
    %     tstep = 10; % in seconds
    %     events = make_fixed_length_events(filtered_data, tstep);
    %     epochs = pop_epoch(filtered_data, {}, [0 tstep]);
    % 
    %     % Auto-reject epochs if specified
    %     if autoreject
    %         disp('Detecting bad channels and epochs...');
    %         reject_log = pop_autorej(epochs, artifact_threshold, 'badchan', 'off', 'eegplot', 'off');
    %     else
    %         reject_log = [];
    %     end
    % end
    % 
    % function [ica, n_components] = run_ica(epochs)
    %     % Run ICA on EEG epochs
    % 
    %     disp('Running ICA...');
    %     ica = pop_runica(epochs, 'extended', 1, 'pca', 0.99);
    %     n_components = size(ica.icawinv, 1);
    % end
    % 
    % function [ica, eog_indices, ecg_indices, emg_indices] = ica_correlation(ica, epochs)
    %     % Semi-automatic ICA component selection using correlations
    % 
    %     disp('Correlating ICA components with artifacts...');
    %     [eog_indices, ~] = pop_eegcorr(ica, epochs, 1); % Eye movement (EOG)
    %     [ecg_indices, ~] = pop_eegcorr(ica, epochs, 2); % Cardiac (ECG)
    %     [emg_indices, ~] = pop_eegcorr(ica, epochs, 3); % Muscle activity (EMG)
    %     ica = pop_subcomp(ica, [eog_indices, ecg_indices, emg_indices], 0);
    % end

end