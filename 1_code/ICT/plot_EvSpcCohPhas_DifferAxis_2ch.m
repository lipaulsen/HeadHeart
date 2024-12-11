function u = plot_EvSpcCohPhas_DifferAxis_2ch(DataInfo)   
   stPowColorAxis=DataInfo.stPowerColorAxis;
   PowMin=DataInfo.PowClMin;
   PowMax=DataInfo.PowClMax;
   stCohColorAxis=DataInfo.stCohColorAxis;
   CohClxMin=DataInfo.CohClxMin;
   CohClxMax=DataInfo.CohClxMax;
   Frqs=DataInfo.Frqs;
   nEvs=DataInfo.nEvs; 
   tSmSpc=DataInfo.tSmSpc;
   tSmPsi=DataInfo.tSmPsi;
   tSmItc=DataInfo.tSmItc;
   nFrqs=length(Frqs);
   SpecsTtl=DataInfo.SpecsTtl;
   Ch2EvsData=DataInfo.Ch2EvsData;
   Ch1EvsData=DataInfo.Ch1EvsData;

   stMapsLines=DataInfo.stMapsLines; % 1- plot Maps, other - plot Lines
   LinesFrq=DataInfo.LinesFrq;
   LinesClx=DataInfo.LinesClx;
   ncnt=DataInfo.FrqMean;   
   
   nLns=length(LinesFrq);
   for il=1:nLns
       stx=sprintf("%3.1fhz",Frqs(LinesFrq(il)));
       LinesTtl(il)={stx};
   end
   
    FigTtl=DataInfo.fName;

    nSmSpc=floor(tSmSpc*DataInfo.SR)+1;
    if nSmSpc > 1
        for ifr=1:nFrqs
            dx=squeeze(DataInfo.PowData1(ifr,:));
            dx=smooth(dx,nSmSpc);
            DataInfo.PowData1(ifr,:)=dx;
            dx=squeeze(DataInfo.PowData2(ifr,:));
            dx=smooth(dx,nSmSpc);
            DataInfo.PowData2(ifr,:)=dx;
        end
        FigTtl=FigTtl+", tSmSpc="+num2str(tSmSpc);
    end    

    nSmPsi=floor(tSmPsi*DataInfo.SR)+1;
    if nSmPsi > 1
        for ifr=1:nFrqs
            dx=squeeze(DataInfo.CohTrsPsi(ifr,:));
            dx=smooth(dx,nSmPsi);
            DataInfo.CohTrsPsi(ifr,:)=dx;

            dx=squeeze(DataInfo.CohTimPsi(ifr,:));
            dx=smooth(dx,nSmPsi);
            DataInfo.CohTimPsi(ifr,:)=dx;
        end
        FigTtl=FigTtl+", tSmPsi="+num2str(tSmSpc);
    end    
    
    nSmItc=floor(tSmItc*DataInfo.SR)+1;
    if nSmItc > 1
        for ifr=1:nFrqs
            dx=squeeze(DataInfo.CohTrsItc1(ifr,:));
            dx=smooth(dx,nSmItc);
            DataInfo.CohTrsItc1(ifr,:)=dx;

            dx=squeeze(DataInfo.CohTrsItc2(ifr,:));
            dx=smooth(dx,nSmItc);
            DataInfo.CohTrsItc2(ifr,:)=dx;
        end
        FigTtl=FigTtl+", tSmITC="+num2str(tSmItc);
    end 


    if DataInfo.stSpecLog > 0
        DataInfo.PowData1=log10(DataInfo.PowData1);
        DataInfo.PowData2=log10(DataInfo.PowData2);
        FigTtl=FigTtl+", Log";
    end

    if DataInfo.stZscoreSpc > 0
        DataInfo.PowData1=zscore(DataInfo.PowData1,0,2);
        DataInfo.PowData2=zscore(DataInfo.PowData2,0,2);
        FigTtl=FigTtl+", zSc";
        stPowColorAxis=1; PowMin=-2; PowMax=2; 
    end

    if stPowColorAxis < 0
        PowMin=min(min(DataInfo.PowData1));
        PowMax=max(max(DataInfo.PowData1));
    end    

    if DataInfo.stZscorePSI > 0
        DataInfo.CohTrsPsi=zscore(DataInfo.CohTrsPsi,0,2);
        DataInfo.CohTimPsi=zscore(DataInfo.CohTimPsi,0,2);
        FigTtl=FigTtl+", zScPSI";
    end
    if stCohColorAxis <= 0
        CohClxMin=min(min(DataInfo.CohTrsPsi(10:45,:)));
        CohClxMax=max(max(DataInfo.CohTimPsi(10:45,:)));
    end    

   u=figure;
   set(u,'Name',FigTtl);
   rscreen=get(0,'ScreenSize');
   dy=rscreen(4)/2;
%   set(u,'Position',[1 dy-10 rscreen(3)/2 dy-60] )
   set(u,'Position',DataInfo.FigSize);
   set(groot, 'DefaultTextInterpreter', 'none');   
   % 
% 
    %% Plot PSD 
    % txt=sprintf('PSD for %s', DataInfo.Title1);
    % subplot(7,1,1);
    % ChsEvsFrsPds = squeeze(mean(mean(DataInfo.ChsEvsFrsTmSpc.^2, 2), 4)); % Average over Time and Events 
    % plot(Frqs, ChsEvsFrsPds(1, :), 'b', 'LineWidth', 1);
    % hold on;
    % if DataInfo.Title2 ~= "ECG"
    %     plot(Frqs, ChsEvsFrsPds(2, :), 'k', 'LineWidth', 1); 
    %     txt=sprintf('PSD for %s and %s', DataInfo.Title1,  DataInfo.Title2);
    % end
    % xlabel('Frequency (Hz)'); ylabel('Power (Magnitude^2)');
    %  if numel(findall(gca,'type','line')) == 2; legend(DataInfo.Title1, DataInfo.Title2); else; legend(DataInfo.Title1);end
    % title(txt);
    % hold off;


    %% Plot the ERP and Average ECG 

    %subplot(7,1,2);
    subplot(3,1,1);
    % here comes the ERP Code
    yyaxis left
    ave1=mean(Ch1EvsData,1);
    plot(DataInfo.TmAxis,ave1,'k','LineWidth',1); hold on;
    set(gca, 'YColor', 'k');
    ylabel('Amplitude (in Uv)');
    txt=sprintf('%s ERP n=%d and %s Average', DataInfo.Title1,  nEvs,  DataInfo.EcgTitle);
    if DataInfo.Title2 ~= "ECG"
        ave3=mean(Ch2EvsData,1);
        plot(DataInfo.TmAxis,ave3,'-b','LineWidth',1);
        txt=sprintf('%s and %s ERP n=%d and %s Average', DataInfo.Title1, DataInfo.Title2,  nEvs,  DataInfo.EcgTitle);
    end
    if min(ave1) && max(ave1) > min(ave3) && max(ave3); ylim([min(ave1), max(ave1)]); else; ylim([min(ave3), max(ave3)]); end
    % plot the avg ECG 
    yyaxis right
    ave2=mean(DataInfo.EvEcgData,1);
    plot(DataInfo.TmAxis,ave2,'r','LineWidth',1);
    ylim([min(ave2), max(ave2)]);
    xline(0, "--k", 'LineWidth', 2);
    if numel(findall(gca,'type','line')) == 3; legend(DataInfo.Title1, DataInfo.Title2, 'ECG'); else; legend(DataInfo.Title1, 'ECG');end
    set(gca, 'YColor', 'r');
    
    % if the second channel is not ECG then plot the ERP too
  
    xlim([DataInfo.TmAxis(1) DataInfo.TmAxis(end)]);
    title(txt); %  ylim([ yxmin yxmax ]);
    hold off;
%%
% 
   %subplot(2,3,1); 
   % subplot(7,1,6);
   % 
   % txt=sprintf('%s, %s, %s=%d',DataInfo.Title1,SpecsTtl,DataInfo.EvTitle,nEvs);
   % xpow=DataInfo.PowData1;
   % if stMapsLines == 1
   %     imagesc(DataInfo.TmAxis,Frqs,xpow); axis xy; 
   %     colormap(DataInfo.ColorMap);
   % 
   %     if stPowColorAxis > 0; clim([PowMin PowMax]); end
   %     colorbar;
   %     xline(0, "--k", 'LineWidth', 2);
   % else
   %     hold on;
   %     for il=1:nLns
   %         nst=LinesFrq(il)-ncnt; if nst < 1; nst=1; end
   %         nen=LinesFrq(il)+ncnt; if nen > nFrqs; nst=nFrqs; end
   %         dx=mean(xpow(nst:nen,:),1);
   %         plot(DataInfo.TmAxis,dx,LinesClx(il));
   %     end
   %     legend(LinesTtl); legend("boxoff");
   %     if stPowColorAxis > 0; ylim([PowMin PowMax]); end
   %     xlim([DataInfo.TmAxis(1) DataInfo.TmAxis(end)]);
   %     grid on;
   % end
   % title(txt); 
   % 
   % % subplot(2,3,4); 
   % subplot(7,1,7);
   % txt=sprintf('%s, %s, %s=%d',DataInfo.Title2,SpecsTtl,DataInfo.EvTitle,nEvs);
   % xpow=DataInfo.PowData2;
   % if stMapsLines == 1
   %     imagesc(DataInfo.TmAxis,Frqs,xpow); axis xy; 
   %     colormap(DataInfo.ColorMap);
   %     if stPowColorAxis > 0; clim([PowMin PowMax]); end
   %     colorbar;
   %     xline(0, "--k", 'LineWidth', 2);
   % else
   %     hold on;
   %     for il=1:nLns
   %         nst=LinesFrq(il)-ncnt; if nst < 1; nst=1; end
   %         nen=LinesFrq(il)+ncnt; if nen > nFrqs; nst=nFrqs; end
   %         dx=mean(xpow(nst:nen,:),1);
   %         plot(DataInfo.TmAxis,dx,LinesClx(il));
   %     end
   %     legend(LinesTtl); legend("boxoff");
   %     if stPowColorAxis > 0; ylim([PowMin PowMax]); end
   %     xlim([DataInfo.TmAxis(1) DataInfo.TmAxis(end)]);
   %     grid on;
   % end
   % title(txt);

   %% Plot Trial ITC Data
   xttl=DataInfo.CohItcTtl;
   xaxi=DataInfo.CohItcAxis;

   % subplot(2,3,2);
   % subplot(7,1,3);
   subplot(3,1,2);
   txt=sprintf('%s, %s',DataInfo.Title1,xttl);
   xdta=DataInfo.CohTrsItc1; 
   
   if stMapsLines == 1
       imagesc(xaxi,Frqs,xdta); axis xy;
       colormap(DataInfo.ColorMap)
        hold on
        if DataInfo.permstats
            itc1_zscores_thresh = (DataInfo.Itc1_clusPos_Z_Stat > 15) | (DataInfo.Itc1_clusPos_Z_Stat < -15);
            contour(xaxi,Frqs, itc1_zscores_thresh, 1, 'linecolor', 'k', 'linewidth', 0.9);
        end
       if stCohColorAxis > 0
           %clim([CohClxMin CohClxMax]);
       end
       colorbar;
       xline(0, "--k", 'LineWidth', 2);
   else
       hold on;
       for il=1:nLns
           nst=LinesFrq(il)-ncnt; if nst < 1; nst=1; end
           nen=LinesFrq(il)+ncnt; if nen > nFrqs; nst=nFrqs; end
           dx=mean(xdta(nst:nen,:),1);
           plot(xaxi,dx,LinesClx(il));
       end
       legend(LinesTtl); legend("boxoff");
       if stCohColorAxis > 0; ylim([CohClxMin CohClxMax]); end
       xlim([xaxi(1) xaxi(end)]);
       grid on;
   end
   ylabel('Frequency (in Hz)');
   title(txt);

%% Plot ITC for 2nd Channel
   % 
   % % subplot(7,1,4);
   % subplot(3,1,3);
   % txt=sprintf('%s, %s',DataInfo.Title2,xttl);
   % xdta=DataInfo.CohTrsItc2;
   % 
   % if stMapsLines == 1
   %     imagesc(xaxi,Frqs,xdta); axis xy;
   %     colormap(DataInfo.ColorMap);
   %     hold on
   %     if DataInfo.permstats  
   %         itc2_zscores_thresh = (DataInfo.Itc2_clusPos_Z_Stat > 15) | (DataInfo.Itc2_clusPos_Z_Stat < -15);
   %             contour(xaxi,Frqs,itc2_zscores_thresh, 1, 'linecolor', 'k', 'linewidth', 1.1);
   %     end
   %     if stCohColorAxis > 0
   %         %clim([CohClxMin CohClxMax]);
   %     end
   %     colorbar;
   %     xline(0, "--k", 'LineWidth', 2);
   % else
   %     hold on;
   %     for il=1:nLns
   %         nst=LinesFrq(il)-ncnt; if nst < 1; nst=1; end
   %         nen=LinesFrq(il)+ncnt; if nen > nFrqs; nst=nFrqs; end
   %         dx=mean(xdta(nst:nen,:),1);
   %         plot(xaxi,dx,LinesClx(il));
   %     end
   %     legend(LinesTtl); legend("boxoff");
   %     if stCohColorAxis > 0; ylim([CohClxMin CohClxMax]); end
   %     xlim([xaxi(1) xaxi(end)]);
   %     grid on;
   % end
   % ylabel('Frequency (in Hz)');
   % title(txt);

   %% Plot Trial PSI Data
   xttl=DataInfo.CohTrsTtl;
   xaxi=DataInfo.CohTrsAxis;


   % subplot(2,3,2);
   % subplot(7,1,5);
   subplot(3,1,3);
   txt=sprintf('%s-%s, %s',DataInfo.Title1,DataInfo.Title2,xttl);
   xdta=DataInfo.CohTrsPsi;
   if stMapsLines == 1
       imagesc(xaxi,Frqs,xdta); axis xy;
       colormap(DataInfo.ColorMap);
       hold on
       if DataInfo.permstats
    ccc_zscores_thresh = (DataInfo.Ccc_zscores > 15) | (DataInfo.Ccc_zscores < -15);
           contour(xaxi,Frqs,ccc_zscores_thresh, 1, 'linecolor', 'k', 'linewidth', 1.1); 
           %imagesc(xaxi,Frqs,ccc_zscores_thresh);axis xy;
       end
       if stCohColorAxis > 0
           %clim([CohClxMin CohClxMax]);
       end
       colorbar;
       xline(0, "--k", 'LineWidth', 2);
   else
       hold on;
       for il=1:nLns
           nst=LinesFrq(il)-ncnt; if nst < 1; nst=1; end
           nen=LinesFrq(il)+ncnt; if nen > nFrqs; nst=nFrqs; end
           dx=mean(xdta(nst:nen,:),1);
           plot(xaxi,dx,LinesClx(il));
       end
       legend(LinesTtl); legend("boxoff");
       if stCohColorAxis > 0; ylim([CohClxMin CohClxMax]); end
       xlim([xaxi(1) xaxi(end)]);
       grid on;
   end
   xlabel('Time (in s)'); ylabel('Frequency (in Hz)');
   title(txt);

   % 
   % %subplot(2,3,3); 
   % subplot(6,1,6);
   % txt=sprintf('%s-%s, PhasDf  Trials',DataInfo.Title1,DataInfo.Title2);
   % xdta=DataInfo.CohTrsPha;
   % if stMapsLines == 1
   %     imagesc(xaxi,Frqs,xdta); axis xy; 
   %     colormap(DataInfo.ColorMap);
   %     clim([-180 180]);
   %     colorbar;
   % else
   %     hold on;
   %     for il=1:nLns
   %         nst=LinesFrq(il)-ncnt; if nst < 1; nst=1; end
   %         nen=LinesFrq(il)+ncnt; if nen > nFrqs; nst=nFrqs; end
   %         dx=mean(xdta(nst:nen,:),1);
   %         plot(xaxi,dx,LinesClx(il));
   %     end
   %     legend(LinesTtl); legend("boxoff");
   %     ylim([-180 180]);
   %     xlim([xaxi(1) xaxi(end)]);
   %     grid on;
   % end
   % title(txt);   

% %  Time PSI Data
%    xttl=DataInfo.CohTimTtl;
%    xaxi=DataInfo.CohTimAxis;
% 
%    %subplot(2,3,5);
%    subplot(7,1,7);
%    txt=sprintf('%s-%s, %s',DataInfo.Title1,DataInfo.Title2,xttl);
%    xdta=DataInfo.CohTimPsi; 
%    if stMapsLines == 1
%        imagesc(xaxi,Frqs,xdta); axis xy; 
%        colormap(DataInfo.ColorMap);
%        if stCohColorAxis > 0
%            % clim([CohClxMin CohClxMax]);
%        end
%        colorbar;
%        xline(0, "--k", 'LineWidth', 2);
%    else
%        hold on;
%        for il=1:nLns
%            nst=LinesFrq(il)-ncnt; if nst < 1; nst=1; end
%            nen=LinesFrq(il)+ncnt; if nen > nFrqs; nst=nFrqs; end
%            dx=mean(xdta(nst:nen,:),1);
%            plot(xaxi,dx,LinesClx(il));
%        end
%        legend(LinesTtl); legend("boxoff");
%        if stCohColorAxis > 0; ylim([CohClxMin CohClxMax]); end
%        xlim([xaxi(1) xaxi(end)]);
%        grid on;
%    end
%    title(txt); 
%    xlabel(FigTtl);
% 
%    subplot(2,3,6); 
%    txt=sprintf('%s-%s, PhasDf  Time',DataInfo.Title1,DataInfo.Title2);
%    xdta=DataInfo.CohTimPha;
%    if stMapsLines == 1
%        imagesc(xaxi,Frqs,xdta); axis xy; 
%        colormap(DataInfo.ColorMap);
%        clim([-180 180]);
%        colorbar;
%    else
%        hold on;
%        for il=1:nLns
%            nst=LinesFrq(il)-ncnt; if nst < 1; nst=1; end
%            nen=LinesFrq(il)+ncnt; if nen > nFrqs; nst=nFrqs; end
%            dx=mean(xdta(nst:nen,:),1);
%            plot(xaxi,dx,LinesClx(il));
%        end
%        legend(LinesTtl); legend("boxoff");
%        ylim([-180 180]);
%        xlim([xaxi(1) xaxi(end)]);
%        grid on;
%    end
%    title(txt); 
% 