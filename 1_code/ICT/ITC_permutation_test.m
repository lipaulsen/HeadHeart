function [p_values, clusPos_Z_Stat, clusPval_Z_Stat] = ITC_permutation_test(original_ITC, ChDta, ibi_series, numPerms, freq_bins, time_bins, SR)
    % INPUTS:
    % original_ITC - Original inter-trial coherence values [Freq x Time]
    % ibi_series - Vector of interbeat intervals (IBIs) for R-peaks
    % numPerms - Number of permutations (e.g., 1000)
    % freq_bins, time_bins - The frequency and time bins for analysis
    
    % OUTPUTS:
    % p_values - Raw p-values (Freq x Time)

    % Initialize storage for permutation-based ITC
    permuted_ITCs = zeros([numPerms, size(original_ITC, 1), size(original_ITC, 2)]);

    BandWidth=2; % BandWidth in Hz;
    Qfac     =2; % Attenuation in db(-Qfac)
    tCircMean=0.05;

    tWidth   = 1.5;
    tOffset  = 0.4;
    FltPassDir = 'onepass';
    
    % Generate Permutations
    for perm = 1 %:numPerms
        % Step 1: Generate surrogate R-peaks
        time_shifts = rand(size(ibi_series)) - 0.5; % in sec
        surrogate_rpeaks(perm,:) = cumsum(ibi_series) + time_shifts;
     
        % Time Lock the surrogate R Peaks to the Channel Data
         tfrSurrData = time_lock_to_surrogate(ChDta, surrogate_rpeaks, SR, freq_bins, tWidth, tOffset,  BandWidth, Qfac, FltPassDir);

        % Step 2: Calculate ITC with the surrogate R-peaks (one per channel)
        [tfrSurrItc(perm, :, :)] = Get_PSI_ByTrials_ITC(tfrSurrData,SR,tCircMean);
     
        fprintf('perm = %d', perm)
    end
    
    % Step 3: Compute the p-values by comparing original ITC to permuted ITC distribution
    clusPval_Z_Stat, clusPos_Z_Stat, clusPval_clusSize, clusPos_clusSize = calculate_p_values(original_ITC, permuted_ITCs);
    
end


% Function to time-lock neural data (e.g., EEG/LFP) to surrogate R-peaks
function  tfrSurrData = time_lock_to_surrogate(ChDta, surrogate_rpeaks, SR, freq_bins, tWidth, tOffset, BandWidth, Qfac, FltPassDir)

dtTime = 1/SR;
[EventTms,EvData,TmAxis]=GetEvTimeAndData(surrogate_rpeaks,ChDta,dtTime,tWidth,tOffset);


% Perform time-frequency decomposition
[nEvs,nData]=size(EvData);
nFrs=length(freq_bins);
tfrSurrData=zeros(nEvs,nFrs,nData);
for iev=1:nEvs
    dx=squeeze(EvData(iev,:));
    for ifr=1:nFrs
        vfr=freq_bins(ifr);
        df=IIRPeak_Flt(dx,SR,vfr,BandWidth,Qfac,FltPassDir);
        tfrSurrData(iev,ifr,:)=hilbert(df);
    end
end
end


function clusPval_Z_Stat, clusPos_Z_Stat, clusPval_clusSize, clusPos_clusSize = calculate_p_values(tfrSurrItc)

    data1 = tfrSurrItc;
    data2 =  zeros(size(tfrSurrItc)); % do stats against the null hypothesis 
    oneDim = false;
    diff = shiftdim(mean(data2-data1),1); % find the mean difference, important to use shiftdim for 1-d data
    numSub = size(data1,1);

    % Define significance threshold and alpha level
    THRESH_SUPRACLUSTER = 0.05; % Threshold for suprathreshold clusters
    ALPHA = 0.05; % Alpha level for final significance testing

    permuteSign = (rand(numSub, 1, 1, numPerms) < 0.5) * 2 - 1;    % creates -1 or 1 to shuffle the order of condition subtraction for each subject and permutation ...
    permuteSign = repmat(permuteSign, [1, size(data1,2), size(data1,3), 1]); % ...but leaves the time-frequency association for each subject intact by just replicating it
    diff_perm   = repmat(data1-data2, [1,1,1,numPerms]) .* permuteSign; % shuffle signs
    diff_perm   = squeeze(mean(diff_perm, 1)); % take the mean across subjects
    if oneDim
        diff_perm   = permute(diff_perm, [2,1]);  % re-order dimensions to: diff_perm(NUMPERMS x TIME/FREQ)
    else
        diff_perm   = permute(diff_perm, [3,1,2]);  % re-order dimensions to: diff_perm(NUMPERMS x TIME, FREQ)
    end



    % Step 1: Compute z-scores for original data
    diff_sum_perm_mean = squeeze(mean(diff_perm)); % Mean of the permutation distribution
    diff_sum_perm_std = squeeze(std(diff_perm)); % Standard deviation of permutation distribution

    zscores = (diff - diff_sum_perm_mean) ./ diff_sum_perm_std;
    

    % Step 2: Compute p-values from z-scores (two-tailed)
    p_orig = 2 * (1 - normcdf(abs(zscores), 0, 1));

    % Step 3: Calculate z-scores for permutations
    zscores_perm = bsxfun(@rdivide, bsxfun(@minus, diff_perm, diff_sum_perm_mean), diff_sum_perm_std);
    p_perm = 2 * (1 - normcdf(abs(zscores_perm), 0, 1));

    % Step 4: Prepare inputs for getSignifClusters
    % Create a logical significance matrix for the original data
    p_sig = p_orig < THRESH_SUPRACLUSTER;

    % Call getSignifClusters to find clusters
    [clusPval_Z_Stat, clusPos_Z_Stat, clusPval_clusSize, clusPos_clusSize] = ...
        getSignifClusters(p_sig, zscores, p_perm, zscores_perm, THRESH_SUPRACLUSTER, ALPHA);

    % The output clusPos_Z_Stat and clusPos_clusSize will indicate positions of significant clusters
    % clusPval_Z_Stat and clusPval_clusSize contain p-values for each detected cluster
end

