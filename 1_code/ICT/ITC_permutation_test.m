function [p_values, fdr_corrected_p] = ITC_permutation_test(original_ITC, ChDta, ibi_series, numPerms, freq_bins, time_bins, SR)
    % INPUTS:
    % original_ITC - Original inter-trial coherence values [Freq x Time]
    % ibi_series - Vector of interbeat intervals (IBIs) for R-peaks
    % numPerms - Number of permutations (e.g., 1000)
    % freq_bins, time_bins - The frequency and time bins for analysis
    
    % OUTPUTS:
    % p_values - Raw p-values (Freq x Time)
    % fdr_corrected_p - FDR-corrected p-values (Freq x Time)

    % Initialize storage for permutation-based ITC
    permuted_ITCs = zeros([numPerms, size(original_ITC, 1), size(original_ITC, 2)]);

    BandWidth=2; % BandWidth in Hz;
    Qfac     =2; % Attenuation in db(-Qfac)

    tWidth   = 1.5;
    tOffset  = 0.4;
    FltPassDir = 'onepass';
    
    % Generate Permutations
    for perm = 1:numPerms
        % Step 1: Generate surrogate R-peaks
        surrogate_rpeaks = generate_surrogate_rpeaks(ibi_series);
        
        % Step 2: Calculate ITC with the surrogate R-peaks (one per channel)
        permuted_ITCs(perm, :, :) = calculate_ITC_with_surrogate(surrogate_rpeaks, freq_bins, time_bins);
    end
    
    % Step 3: Compute the p-values by comparing original ITC to permuted ITC distribution
    p_values = calculate_p_values(original_ITC, permuted_ITCs);
    
    % Step 4: Apply FDR correction across time-frequency bins
    fdr_corrected_p = fdr_correction(p_values, 0.05);  % Set FDR threshold, e.g., 0.05
end

function surrogate_rpeaks = generate_surrogate_rpeaks(ibi_series)
    % Function to create surrogate R-peaks based on the given IBIs
    % Random shifts within -500 to +500 ms
    time_shifts = (rand(size(ibi_series)) - 0.5) * 1000; % Shift in ms
    surrogate_rpeaks = ibi_series + time_shifts;
end

% Function to time-lock neural data (e.g., EEG/LFP) to surrogate R-peaks
function surrogate_data = time_lock_to_surrogate(ChDta, surrogate_rpeaks, SR, freq_bins, tWidth, tOffset)
    
    % Create a new matrix to store time-locked data for surrogate R-peaks
    nRpeaks = length(surrogate_rpeaks); % Number of surrogate R-peaks
    surrogate_data = zeros(nRpeaks, length(freq_bins), length(time_bins)); % Matrix to store time-locked data for surrogate R-peaks
    
    % Loop through each surrogate R-peak to extract data around it
    for peak_idx = 1:nRpeaks
        % Extract a window around the surrogate R-peak
        start_idx = surrogate_rpeaks(peak_idx) - tOffset; % Start index for the segment
        end_idx = surrogate_rpeaks(peak_idx) + (tWidth-tOffset);   % End index for the segment
        
        % Ensure indices are within bounds of the neural data
        if start_idx > 0 && end_idx <= length(ChDta)
            data_window = ChDta(start_idx:end_idx); % Segment of data around the surrogate R-peak
            
            % Perform time-frequency decomposition 
            surrogate_data(peak_idx, :, :) = compute_time_frequency_decomp(data_window, freq_bins, time_bins);
        else
            continue; % Skip if the window is out of bounds
        end
    end
end

% Function for time-frequency decomposition 
function tfrSurrData = compute_time_frequency_decomp(data_window, freq_bins, BandWidth, Qfac, FltPassDir)
[nEvs,nData]=size(data_window);
nFrs=length(freq_bins);
tfrSurrData=zeros(nEvs,nFrs,nData);
for iev=1:nEvs
    dx=squeeze(data_window(ich,iev,:));
    for ifr=1:nFrs
        vfr=Frqs(ifr);
        df=IIRPeak_Flt(dx,SR,vfr,BandWidth,Qfac,FltPassDir);
        tfrSurrData(iev,ifr,:)=hilbert(df);
    end
end
end 

function perm_ITC = calculate_ITC_with_surrogate(surrogate_rpeaks, freq_bins, time_bins)
    % Custom function to calculate ITC with surrogate R-peaks
    % (Compute ITC across time and frequency bins here, as in original ITC)
    [FrsTmItc] = Get_PSI_ByTrials_ITC(Ch1EvsFrsTmPha,SR,tCircMean);
    % Dummy data, assuming `freq_bins` and `time_bins` as input dimensions
    perm_ITC = rand(size(freq_bins, 1), size(time_bins, 2));  % Replace with actual calculation
end

function p_values = calculate_p_values(original_ITC, permuted_ITCs)
    % Compare original ITC values to the null distribution and compute p-values
    p_values = zeros(size(original_ITC));
    for f = 1:size(original_ITC, 1)
        for t = 1:size(original_ITC, 2)
            p_values(f, t) = mean(squeeze(permuted_ITCs(:, f, t)) >= original_ITC(f, t));
        end
    end
end

function fdr_corrected_p = fdr_correction(p_values, alpha)
    % Perform FDR correction on the p-values using the Benjamini-Hochberg method
    sorted_p = sort(p_values(:));
    m = length(sorted_p);
    fdr_threshold = (1:m)'/m * alpha;
    below_threshold = sorted_p <= fdr_threshold;
    max_idx = find(below_threshold, 1, 'last');
    if ~isempty(max_idx)
        fdr_level = sorted_p(max_idx);
        fdr_corrected_p = p_values <= fdr_level;
    else
        fdr_corrected_p = false(size(p_values));
    end
end
