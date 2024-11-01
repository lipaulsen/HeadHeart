 %function [] = erp_calculation(subjects, data_dir, results_dir, MedFlag)

% TO DO

%  - Different Basline try out? yes longer 


%% Calculating ERPs for HeadHeart

% Author: Lisa Paulsen
% Contact: lisaspaulsen[at]web.de
% Created on: 1 October 2024
% Last update: 23 October 2024

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
%   2a. Extract Epochs
%   2b. Plotting ERPs for all Channels Seperately
%   2c. Plotting ERPs for all Channels overlapped
%   2d. Save the ERP Data
% 3. ERP Statistical Analysis
%   3d. P
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
erp_name = 'erp';  
baseline_name = '-400ms';

% Define the medications
medications = {'MedOn', 'MedOff'};

% Create folders if it does not exist
for med_name = medications
    for subject = subjects
        % Create the  erp data folder if it does not exist
        subject_erp_folder = fullfile(data_dir, erp_name,med_name{1}, ...
            sprintf(['sub-', subject{1}]));
        if ~exist(subject_erp_folder, 'dir')
            mkdir(subject_erp_folder);
        end
        % Create the  erp results data folder if it does not exist
        subject_erp_results_folder = fullfile(results_dir, erp_name, med_name{1},...
            sprintf(['sub-', subject{1}]));
        if ~exist(subject_erp_results_folder, 'dir')
            mkdir(subject_erp_results_folder);
        end
    end
end


%% ============================ 1. LOAD DATA =============================
disp("************* STARTING  ERP CALCULATION *************");

% Create a overall matrix to store the ERPs for all participants
%ERP_avg_all = zeros(1,numel(subjects));


for med = 1:2

    % Extract the medication info
    med_name = medications{med};

    for sub = 1:numel(subjects)

        % Extract the subject
        subject = subjects{sub};

        fprintf('Loading Data of subject %s number %i of %i\n', subject, sub, numel(subjects));

        % Load subject data
        subject_data = fullfile(data_dir, preprocessed_name, med_name, ['sub-', subject], [subject, '_preprocessed_', med_name, '_Rest.mat']);
        load(subject_data, 'SmrData');

        % Define the path to subject preprocessed folder
        subject_erp_folder = fullfile(data_dir, erp_name, med_name, sprintf('sub-%s', subject));

        % Define the path to subject preprocessed folder
        subject_erp_results_folder = fullfile(results_dir, erp_name, med_name, sprintf('sub-%s', subject));


        %% ============================ 2. ERP  =============================

        fprintf('Extracting ERPs for subject %s\n', subject);

        % Define the SR
        SR = SmrData.SR;

        % Define the time window around Heartbeat
        time_windows = [-0.1 0.8;  -0.05 0.4; -0.04 0.1]; % in sec ; -0.1 0.6; -0.04 0.04]

        for win = 1:size(time_windows, 1)
            
            fprintf('Extracting ERPs for subject %s  with window %i\n', subject, win);
            % Get current window
            time_win = time_windows(win, :);

            % Transform the timewindow into samples
            time_wins_samples = round(time_win*SR); % in samples

            % Define the baseline window
            baseline_win = round([-0.4 -0.1]*SR); % 400 to 100ms before rPeak as baseline

            % Extract R-Peak Indicies
            r_peak_indices = find(SmrData.WvDataCleaned(20,:));

            % Create Dataframe to store the ERP data
            num_channels = 15;  % Number of EEG/LFP channels. All channels = 15
            epoch_length =  time_wins_samples(2) - time_wins_samples(1) + 1; % Length of the epoch in samples
            num_trials = numel(r_peak_indices);  % Number of R-peaks (trials)
            ERP_epochs = zeros(num_channels, epoch_length, num_trials); % Matrix that is channels x epoch samples x trials (15x1435x325 in my case)

            %% 2a. Extract Epochs
            for peaks = 1:numel(r_peak_indices)

                % Extract current Peak
                r_peak_idx = r_peak_indices(peaks);

                % Define the epoch range
                epoch_start = r_peak_idx + time_wins_samples(1); % start before r-peak
                epoch_end = r_peak_idx + time_wins_samples(2); % end after r-peak

                % Check that Epoch is within the data
                if epoch_start > 0 && epoch_end <= size(SmrData.WvDataCleaned, 2)
                    % Extract and store the epoch data for channels
                    ERP_epochs(:, :, peaks) = SmrData.WvDataCleaned(1:15, epoch_start:epoch_end);
                else
                    % Warn if the epoch is outside of the data
                    warning('Trial %d out of bounds, skipping...\n', peaks);
                end

                % Baseline correction
                % it takes for every trial the 400ms to 100ms BEFORE the r-peak (so
                % baseline is 300ms) and uses the average of those baselines of all
                % trials and then subtract it
                baseline_start = r_peak_idx + baseline_win(1);
                baseline_end = r_peak_idx + baseline_win(2);% Pre-stimulus baseline
                if baseline_start > 0 && baseline_end <= size(SmrData.WvDataCleaned, 2)
                    % Extract and store the epoch data for channels
                    baseline = mean(SmrData.WvDataCleaned(1:15, baseline_start:baseline_end), 2); % Mean over baseline period for each channel
                else
                    % Warn if the epoch is outside of the data
                    warning('Trial %d out of bounds, skipping...\n', peaks);
                    baseline = 0;
                end
                ERP_epochs(:, :, peaks) = ERP_epochs(:, :, peaks) - baseline;  % Subtract baseline

            end


            % Average over all trials
            ERP_average = mean(ERP_epochs, 3);  % Averaging across the 3rd dimension (trials)

            % Save the ERP for the current subject
            ERP.(med_name).(subject) = 

            %% 2b. Plotting ERPs for all Channels Seperately

            fprintf('Plotting ERPs for subject %s\n', subject);
            f1 = figure; % initialize Figure
            for chan = 1:15
                row = ceil(chan / 5); % Calculate the row number
                col = mod(chan - 1, 5) + 1; % Calculate the column number
                subplot(3, 5, (row - 1) * 5 + col)

                % Plot the ERP per channel for 1 subj
                plot((time_wins_samples(1):time_wins_samples(2)) / SR * 1000, ERP_average(chan, :));
                hold on
                xline(0, "--", 'Color', 'k', 'LineWidth', 1); % vertical line at time-lock

                % Set X-Ticks dynamically to the window size
                if win == 1
                    xTicks = round(time_wins_samples(1) / SR * 1000:100:time_wins_samples(2) / SR * 1000); % Set x-ticks in ms
                elseif win == 2
                    xTicks = round(time_wins_samples(1) / SR * 1000:50:time_wins_samples(2) / SR * 1000); % Set x-ticks in ms
                else win == 3;
                    xTicks = round(time_wins_samples(1) / SR * 1000:10:time_wins_samples(2) / SR * 1000); % Set x-ticks in ms
                end
                set(gca, 'XTick', xTicks);

                axis tight

                % Set Labels
                xlabel('Time (ms)');
                ylabel('Amplitude (uV)');
                title(sprintf('Channel %s', SmrData.WvTits{chan}));

            end
            sgtitle(sprintf('ERP for Subject %s - All Channels with %s', subject, med_name)); % Major Title
            set(f1, 'Position', [100, 100, 1920, 1080]);

            gr1 = fullfile(subject_erp_results_folder, ['\', subject, '_ERP_sep-channels_win',  ...
                num2str(time_win(1)), 'till', num2str(time_win(2)),'s_', med_name, '_BSL_', baseline_name,'.png']);
            exportgraphics(f1, gr1, "Resolution",300);

            %% 2c. Plotting ERPs for all Channels overlapped

            f2 = figure; % Create a new figure
            colors = lines(15); % Get distinct colors for each channel
            subplot(2,1,1)
            for chan = 1:7
                hold on;
                % Plot the averaged HEP for the current channel
                plot((time_wins_samples(1):time_wins_samples(2)) / SR * 1000, ERP_average(chan, :), 'Color', colors(chan, :), 'DisplayName', SmrData.WvTits{chan});
            end
            xline(0, 'k--', 'LineWidth', 1.5); % Dashed vertical line at time = 0
            % Set X-Ticks
            if win == 1
                xTicks = round(time_wins_samples(1) / SR * 1000:100:time_wins_samples(2) / SR * 1000); % Set x-ticks in ms
                legend('Location','northeast', 'FontSize',6);
            elseif win == 2
                xTicks = round(time_wins_samples(1) / SR * 1000:50:time_wins_samples(2) / SR * 1000); % Set x-ticks in ms
                legend('Location','northeast', 'FontSize',6);
            else win == 3;
                xTicks = round(time_wins_samples(1) / SR * 1000:10:time_wins_samples(2) / SR * 1000); % Set x-ticks in ms
                legend('Location','northwest', 'FontSize',6);
            end
            set(gca, 'XTick', xTicks);
            axis tight
            xlabel('Time (ms)');
            ylabel('Amplitude (uV)');
            title(sprintf('ERP - EEG for Subject %s %s', subject, med_name));

            subplot(2,1,2)
            for chan = 8:15
                hold on;
                % Plot the averaged HEP for the current channel
                plot((time_wins_samples(1):time_wins_samples(2)) / SR * 1000, ERP_average(chan, :), 'Color', colors(chan, :), 'DisplayName', SmrData.WvTits{chan});

            end
            xline(0, 'k--', 'LineWidth', 1.5); % Dashed vertical line at time = 0
            % Set X-Ticks
            if win == 1
                xTicks = round(time_wins_samples(1) / SR * 1000:100:time_wins_samples(2) / SR * 1000); % Set x-ticks in ms
                legend('Location','northeast', 'FontSize',6);
            elseif win == 2
                xTicks = round(time_wins_samples(1) / SR * 1000:50:time_wins_samples(2) / SR * 1000); % Set x-ticks in ms
                legend('Location','northeast', 'FontSize',6);
            else win == 3;
                xTicks = round(time_wins_samples(1) / SR * 1000:10:time_wins_samples(2) / SR * 1000); % Set x-ticks in ms
                legend('Location','northwest', 'FontSize',6);
            end
            set(gca, 'XTick', xTicks);
            axis tight
            % Set Labels
            xlabel('Time (ms)');
            ylabel('Amplitude (uV)');
            title(sprintf('ERP - LFP for Subject %s %s', subject, med_name));

            gr2 = fullfile(subject_erp_results_folder, ['\', subject, '_ERP_LFP-EEG_win',  num2str(time_win(1)),...
                'till', num2str(time_win(2)),'s_', med_name, '_BSL_', baseline_name,'.png']);
            exportgraphics(f2, gr2, "Resolution",300)



            %% 2d. Save Subject ERP Data

            % Save the matrix as .mat file and the plots as .png
            fprintf('Saving ERPs and Plots for subject %s\n', subject);


            save_path = fullfile(subject_erp_folder, ['\', subject, '_ERP_win', num2str(time_win(1)), 'till', ...
                num2str(time_win(2)),'s-', med_name, '_BSL_', baseline_name,'.mat']);
            save(save_path, 'ERP_average');

        end
    end
end

%% ======================== 3. ERP Stats Analysis  =========================


% Load the Data of MedOff and MedOn Data 



