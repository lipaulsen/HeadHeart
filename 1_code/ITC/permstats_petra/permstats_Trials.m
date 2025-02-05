function [clusPos_Z_Stat, clusPval_Z_Stat] = permstats(data1, data2, numPerms)
% PERMSTATS Cluster-based permutation testing for paired comparisons
% INPUTS
% data1    - data of condition 1 (SUBJECT x FREQ x TIME)
%            OR: (SUBJECT x TIME/FREQ) if you don't want to correct 2-d data 
%            such as time-frequency data but only over TIME or FREQs 
% data2    - data of condition 2, or zeros if you want to test against 0
% numPerms - the number of permutations to run (should be >1000)
%
% OUTPUTS 
% clusPos_Z_Stat  - Positions of significant clusters
% clusPval_Z_Stat - P-values of the orginal supra-thresholds clusters 
%                   after correction

if ndims(data1)==2
    oneDim = true;  % test only 1-d data (SUBJECT x TIME or FREQ)
    add_forNan = data1 + data2;
    nanSubj    = isnan(sum(add_forNan,2)); 
    data1(nanSubj,:) = [];
    data2(nanSubj,:) = [];   
elseif ndims(data2)==3
    oneDim = false; % test 2-d data (SUBJECT x TIME x FREQ)
    add_forNan = data1 + data2;
    nanSubj    = isnan(nanmean(nanmean(add_forNan,3),2));
    data1(nanSubj,:,:) = [];
    data2(nanSubj,:,:) = [];
end

diff = shiftdim(mean(data2-data1),1); % find the mean difference, important to use shiftdim for 1-d data
numSub = size(data1,1);

permuteSign = (rand(numSub, 1, 1, numPerms) < 0.5) * 2 - 1;    % creates -1 or 1 to shuffle the order of condition subtraction for each subject and permutation ... 
permuteSign = repmat(permuteSign, [1, size(data1,2), size(data1,3), 1]); % ...but leaves the time-frequency association for each subject intact by just replicating it
diff_perm   = repmat(data1-data2, [1,1,1,numPerms]) .* permuteSign; % shuffle signs
diff_perm   = squeeze(mean(diff_perm, 1)); % take the mean across subjects
if oneDim
    diff_perm   = permute(diff_perm, [2,1]);  % re-order dimensions to: diff_perm(NUMPERMS x TIME/FREQ)
else
    diff_perm   = permute(diff_perm, [3,1,2]);  % re-order dimensions to: diff_perm(NUMPERMS x TIME, FREQ)
end

%% Cluster based stats
diff_sum_perm_mean = squeeze(mean(diff_perm)); % get the mean of the permutation distribution (TIME x FREQ)
diff_sum_perm_std  = squeeze(std(diff_perm));  % get the std of the permutation distribution (TIME x FREQ)
if oneDim % if there's only a TIME or FREQ dimension:
    zscores = (diff' - diff_sum_perm_mean) ./ diff_sum_perm_std;
    diffPerm_mean = diff_sum_perm_mean;
    diffPerm_std  = diff_sum_perm_std;
else  % otherwise for TIME * FREQ
    zscores = (diff-diff_sum_perm_mean) ./ diff_sum_perm_std; % zscore the real difference relative to the mean and std of the permutation difference
    diffPerm_mean(1,:,:) = diff_sum_perm_mean;
    diffPerm_std(1,:,:)  = diff_sum_perm_std;
end
p_orig = 2 * (1 - normcdf(abs(zscores), 0, 1)); % get p-values from the zscore, abs to make it 2-tailed

% calculate: (diff_perm - diffPerm_mean) / diffPerm_std
zscores_perm = bsxfun(@rdivide, bsxfun(@minus, diff_perm, diffPerm_mean), diffPerm_std);
p_perm =  2 * (1 - normcdf(abs(zscores_perm), 0, 1)); % get p-values from the zscore, abs to make it 2-tailed

preCluster_thresh = .05;
alpha = .05;
% preCluster_thresh = .2;
% alpha = .2;
% Obtain multiple-comparison corrected p-values for each suptra-threshold
% cluster (clusPval_Z_Stat), and the position of significant clusters
% (clusPos_Z_Stat)
% clusPval_Z_Stat and clusPos_Z_Stat are based on the sum of z-scores
% clusPval_clusSize & clusPos_clusSize are based on the cluster size
% irrespective of the z-scores. In most cases the Z_Stat or clusSize
% result is identical.
[clusPval_Z_Stat, clusPos_Z_Stat, clusPval_clusSize, clusPos_clusSize] = getSignifClusters(p_orig, zscores, p_perm, zscores_perm, preCluster_thresh, alpha);
