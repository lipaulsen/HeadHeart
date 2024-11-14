function [FrsTmItc]=Get_PSI_ByTrials_ITC(Ch1EvsFrsTmPha,SR,tCircMean)

[nEvs,nFrs,nData]=size(Ch1EvsFrsTmPha);
nCircMeanH=int32(tCircMean*SR)/2+1;

FrsTmItc = zeros(nFrs, nData); % ITC matrix set up
% Loop over frequencies to calculate ITC for each frequency and time point
for ifr=1:nFrs
    phasedf=squeeze(Ch1EvsFrsTmPha(:, ifr, :));  %Extract phase data for channel 1 at the current frequency
    if nCircMeanH > 1
        for itm=nCircMeanH:(nData-nCircMeanH)
            % Take phase data around each time point for smoothing
            phx1=phasedf(:,(itm-nCircMeanH+1):(itm+nCircMeanH));
            phx2=circ_mean(phx1,[],2); % Calculate circular mean across trials
            phasedf(:,itm)=phx2; % Replace with smoothed values
        end
    end
    for itm=1:nData
        dx=phasedf(:,itm); % Extract all trials' phases for this time point
        vs=vector_strength(dx); % Calculate vector strength for ITC
        FrsTmItc(ifr,itm)=vs; % Store ITC value in the result matrix
    end
end







% % Apply smoothing only where possible (i.e., from nCircMeanH to nData - nCircMeanH)
%     for itm = (nCircMeanH + 1):(nData - nCircMeanH)
%         % Smoothing over a sliding window
%         phase_window = phasedf(:, (itm - nCircMeanH):(itm + nCircMeanH));
%         phase_smoothed = circ_mean(phase_window, [], 2);  % Smoothing window
%         FrsTmItc(ifr, itm) = vector_strength(phase_smoothed);  % ITC calculation
%     end
% 
%     % For the initial part (1 to nCircMeanH), use available data for partial smoothing
%     % Here we take the first `nCircMeanH` time points and perform smoothing with available data
%     for itm = 1:nCircMeanH
%         phase_window = phasedf(:, 1:(itm + nCircMeanH));  % Use available data for initial smoothing
%         phase_smoothed = circ_mean(phase_window, [], 2);   % Smoothing
%         FrsTmItc(ifr, itm) = vector_strength(phase_smoothed);  % ITC calculation
%     end
% 
%     % For the final part (nData - nCircMeanH + 1 to nData), use available data for partial smoothing
%     for itm = (nData - nCircMeanH + 1):nData
%         phase_window = phasedf(:, (itm - nCircMeanH):end);  % Use available data for final smoothing
%         phase_smoothed = circ_mean(phase_window, [], 2);
%         FrsTmItc(ifr, itm) = vector_strength(phase_smoothed);  % ITC calculation
%     end