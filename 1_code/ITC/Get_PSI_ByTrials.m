function [FrsTmPsi]=Get_PSI_ByTrials(Ch1EvsFrsTmPha,Ch2EvsFrsTmPha,SR,tCircMean)
%output alternative: [FrsTmPsi,FrsTmPha]

[nEvs,nFrs,nData]=size(Ch1EvsFrsTmPha);

nCircMeanH=int32(tCircMean*SR)/2+1;

FrsTmPsi=zeros(nFrs,nData);
FrsTmPha=zeros(nFrs,nData);

for ifr=1:nFrs

    phasedf = angle(exp(1i*(squeeze(Ch1EvsFrsTmPha(:,ifr,:)) - squeeze(Ch2EvsFrsTmPha(:,ifr,:)))));
    %phasedf = squeeze(Ch1EvsFrsTmPha(:,ifr,:)) - squeeze(Ch2EvsFrsTmPha(:,ifr,:));

    if nCircMeanH > 1
        for itm=nCircMeanH:(nData-nCircMeanH)
            phx1=phasedf(:,(itm-nCircMeanH+1):(itm+nCircMeanH));
            phx2=circ_mean(phx1,[],2);
            phasedf(:,itm)=phx2;
        end
    end
    for itm=1:nData
        dx=phasedf(:,itm);
        vs=vector_strength_CCC(dx);
        FrsTmPsi(ifr,itm)=vs;
        FrsTmPha(ifr,itm)=circ_mean(dx)*180/pi;
    end
end


% --- Input: Phase data from two channels
% Ch1EvsFrsTmPha: [nTrials x nFreqs x nTimes]
% Ch2EvsFrsTmPha: [nTrials x nFreqs x nTimes]

% [nTrials, nFreqs, nTimes] = size(Ch1EvsFrsTmPha);
% 
% CCC = zeros(nFreqs, nTimes);     % Cross-channel coherence (vector strength)
% PSI = zeros(nFreqs, nTimes);     % Phase slope index
% 
% for ifr = 1:nFreqs
%     for itm = 1:nTimes
%         % === CCC calculation ===
%         pha1 = squeeze(Ch1EvsFrsTmPha(:, ifr, itm));  % [nTrials x 1]
%         pha2 = squeeze(Ch2EvsFrsTmPha(:, ifr, itm));
%         dphi = angle(exp(1i * (pha1 - pha2)));        % Wrap to [-π, π]
%         CCC(ifr, itm) = abs(mean(exp(1i * dphi)));    % Vector strength
%     end
% end
% 
% for itm = 1:nTimes
%     dphi_trials = zeros(nTrials, nFreqs);  % [nTrials x nFreqs]
%     for tr = 1:nTrials
%         pha1 = squeeze(Ch1EvsFrsTmPha(tr, :, itm));  % [1 x nFreqs]
%         pha2 = squeeze(Ch2EvsFrsTmPha(tr, :, itm));
%         dphi = angle(exp(1i * (pha1 - pha2)));       % Phase difference
%         dphi_unwrapped = unwrap(dphi);               % Unwrap across freqs
%         dphi_trials(tr, :) = dphi_unwrapped;
%     end
% 
%     % === PSI calculation === (slope of phase diff across freqs)
%     for ifr = 1:nFreqs
%         x_freq = 1:nFreqs;  % or actual frequency vector if available
%         y = dphi_trials(:, :);  % Each row = trial, each col = freq
%         % Linear fit per trial
%         psi_slope = zeros(nTrials, 1);
%         for tr = 1:nTrials
%             p = polyfit(x_freq, dphi_trials(tr, :), 1);
%             psi_slope(tr) = p(1);  % Slope = PSI
%         end
%         PSI(ifr, itm) = mean(psi_slope);  % Average PSI at this time and freq
%     end
% end
% 
% skdjflk


