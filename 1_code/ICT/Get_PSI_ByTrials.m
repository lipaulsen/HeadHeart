function [FrsTmPsi,FrsTmPha]=Get_PSI_ByTrials(Ch1EvsFrsTmPha,Ch2EvsFrsTmPha,SR,tCircMean)

if ndims(Ch1EvsFrsTmPha) ==3
    [nEvs,nFrs,nData]=size(Ch1EvsFrsTmPha);
elseif ndims(Ch1EvsFrsTmPha) == 2
    [nFrs,nData]=size(Ch1EvsFrsTmPha);
end

nCircMeanH=int32(tCircMean*SR)/2+1;

FrsTmPsi=zeros(nFrs,nData);
FrsTmPha=zeros(nFrs,nData);

for ifr=1:nFrs
    
    if ndims(Ch1EvsFrsTmPha) ==3
       phasedf=squeeze(Ch1EvsFrsTmPha(:,ifr,:))-squeeze(Ch2EvsFrsTmPha(:,ifr,:));
    elseif ndims(Ch1EvsFrsTmPha) == 2
       phasedf=squeeze(Ch1EvsFrsTmPha(ifr,:))-squeeze(Ch2EvsFrsTmPha(ifr,:));
    end

    if nCircMeanH > 1
        for itm=nCircMeanH:(nData-nCircMeanH)
            phx1=phasedf(:,(itm-nCircMeanH+1):(itm+nCircMeanH));
            phx2=circ_mean(phx1,[],2);
            phasedf(:,itm)=phx2;
        end
    end
    for itm=1:nData
        dx=phasedf(:,itm);
        vs=vector_strength(dx);
        FrsTmPsi(ifr,itm)=vs;
        FrsTmPha(ifr,itm)=circ_mean(dx)*180/pi;
    end
end
