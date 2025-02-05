
clear;
clc; 

%FilesDir="D:\OneDrive - Nexus365\Alek_WorkDocs\DataFiles\ReadinessPoteltials\";
FilesDir="/Volumes/LP3/HeadHeart/0_data/preproc/"; %Mac
%FilesDir="F:\LP3\HeadHeart\0_data\preproc\"; % windows

Lucia_Data_ECG={
'raw/';
'SG041_MedOff_Rest_wECG.mat';
'SG041_MedOn_Rest_wECG.mat';
'SG043_MedOff_Rest_wECG.mat';
'SG043_MedOn_Rest_wECG.mat';
'SG044_MedOff_Rest_wECG.mat';
'SG044_MedOn_Rest_wECG.mat';
'SG045_MedOn_Rest_wECG.mat';
'SG046_MedOff_Rest_wECG.mat';
'SG046_MedOn_Rest_wECG.mat';
'SG047_MedOff_Rest_wECG.mat';
'SG047_MedOn_Rest_wECG.mat';
'SG050_MedOff_Rest_wECG.mat';
'SG050_MedOn_Rest_wECG.mat';
'SG052_MedOff_Rest_wECG.mat';
'SG052_MedOn_Rest_wECG.mat';
'SG056_MedOff_Rest_wECG.mat';
'SG056_MedOn_Rest_wECG.mat';
'SG060_MedOn_Rest_wECG.mat';
};


% [SG41 , L3 and R3; C3, C4
%  SG43, R1, C4
%  SG44, L1 and R3, C3, C4
%  SG46, L4, C4
%  SG47 R4, L3 , C3 C4
%  Sg50 L3,R3, C3, C4
%  SG52 R2, C4
%  SG56 L4, R1, C3, C4
%  ]
%%------------------------------------------------------------------------


FileNames=Lucia_Data_ECG; GrNames="Lucia_Data_ECG"; EvChansTtl = {"EvECGP_Cl"}; ECGChansTtl = {"ECG"};
    % 'F3'	'F4'	'C3'	'C4'	'P3'	'P4'	'Pz'	
    % 'L1'	'L2'	'L3'	'L4'	'R1'	'R2'	'R3'	'R4'	
    % 'ECG'	'Ax1'	'Ax2'	'EvECG'	
    % for SG041 L3_25hz  L3_12hz '
    % R2_26hz'	'R2_16hz'

%% Initilize which Channels to use 


 EegLfpChTts= {"L3"; "C3"};

%%
nSubs=length(FileNames)-1;

iSingSub=-1;  % if > 0 takes only 1 subject i.e. isb, <=0 all subs;

sbSt =1;
sbEn =nSubs;
if iSingSub > 0; sbSt=iSingSub; sbEn=iSingSub; end    

[nRow,nCol]=size(EegLfpChTts);
if nRow == 2; EegLfpChTts=[EegLfpChTts(1,:) EegLfpChTts(2,:) ]; end

%%       

stShuffle=-3; % 1 Shuffle 1st chan
             % 2 Shuffle 2nd chan
             % 3 Shuffle both chans
             % Other: NOT shuffle


NewSR=300;
stfr=0.2;   enfr=30; dfr=0.2;
Frqs=stfr:dfr:enfr;
if NewSR < 2*enfr; NewSR=(enfr+2)*2; end

tWidth   = 1.5;
tOffset  = 0.4;
tSmtEvWv = 0.1;
stZscoreWave=1;

DcRemHpFlt=0.1;  % global dc remove
Fhp=0.1;  % high pass filter
Flp=-80;  % high pass filter

stIIRPeakSpec=1;  % 1-IIRPeak, other Wavlet
FltPassDir='onepass'; % onepass  twopass

BandWidth=2; % BandWidth in Hz; 
Qfac     =2; % Attenuation in db(-Qfac)    
WaveletnCyc=6;
WaveletgWidth=3;
        
tSmSpc=0.05;
tSmPsi =0.05;
tSmItc =0.05;
tCircMean=0.05; % for By TRials calc
tCohTm=0.05; 
tCohStep=0.05;

stMapsLinesEPs =1;  % = 1 Plots Maps, Other - Lines
stMapsLinesCoh =1;  % = 1 Plots Maps, Other - Lines

stPlotPSIMap=1;

stSpecLog   =1;
stZscoreSpc =-1;
stZscorePSI =-1;
stSetColorAxis=1;
stColorMap=1;

%%
nLines=4;
LinesClr=["k","b","g","r"];
LinesCrd=zeros(nLines,2);
ilx=1; LinesCrd(ilx,1)=-tOffset+0.1;      LinesCrd(ilx,2)=-tOffset/2;
ilx=2; LinesCrd(ilx,1)=LinesCrd(ilx-1,2); LinesCrd(ilx,2)=0;
ilx=3; LinesCrd(ilx,1)=LinesCrd(ilx-1,2); LinesCrd(ilx,2)=0.5;
ilx=4; LinesCrd(ilx,1)=LinesCrd(ilx-1,2); LinesCrd(ilx,2)=LinesCrd(ilx,1)+1;

%%

% cedpath = getenv('CEDS64ML'); 
% addpath(cedpath);
% CEDS64LoadLib( cedpath );           

for isb=sbSt:sbEn
%% 
    xFilePath=FileNames{1,1};
    xFileName=FileNames{isb+1,1};
    FileNameFull=FilesDir+xFilePath+xFileName;    
    FileNameFull=char(FileNameFull);
    fprintf('---  Subject: %d,  File:  %s\n',isb,FileNameFull);
%%        
    %[fid,WaveChans,EventChans,ChanList]=
    load(FileNameFull,"SmrData");
    ChanList = SmrData.WvTits(1:16);
    
%%    
    [nAllChans,xx]=size(ChanList);
    % LOOk UP These Three
    ChanListNms=1:nAllChans;
    ChanListTyp=ChanList.ChanTypeN; % This ones imp
    ChanListTts=ChanList;   ChanListTts=ChanListTts';
%%    
    nAllChans  =length(ChanListTts);
    if strcmp(ChanListTts, 'ECG2');  ChanListTts = strrep(ChanListTts, 'ECG2', 'ECG');end
%     
    [ChanNumsWv,stState]=Get_ChanNumbersFromTitles(ChanListTts,ChanListNms,EegLfpChTts);
    if stState <= 0; error("Error in Get_ChanNumbersFromTitles"); end
    [ChanNumsEv,stState]=Get_ChanNumbersFromTitles(ChanListTts,ChanListNms,EvChansTtl);
    if stState <= 0; error("Error in Get_ChanNumbersFromTitles"); end
    [ChanNumsECG,stState]=Get_ChanNumbersFromTitles(ChanListTts,ChanListNms,ECGChansTtl);
    if stState <= 0; error("Error in Get_ChanNumbersFromTitles"); end

    nConds=length(ChanNumsEv);
    nChansWv =length(ChanNumsWv);    
%% Read Event Timings
    EvChanTyp=2;
    [EventTms]=Smr64_ReadEventChan(fid,ChanNumsEv,EvChanTyp); % check this length
    nEvs=length(EventTms);
%% Read  LFP/EEG Signal
    clear ChanDataWv;
    ich=0;
    for ix=1:nChansWv
        ic=ChanNumsWv(ix);
        if ( (ChanListTyp(ix) == 1) || (ChanListTyp(ix) == 9) )
            ich=ich+1;
            [MasData,SR]=Smr64_ReadWaveChan(fid,ic);
            ChanDataWv(ich).Data=double(MasData');
            ChanDataWv(ich).Title=EegLfpChTts{ich};
            ChanDataWv(ich).SR=SR;
        end
    end
    if ich ~= nChansWv
        error("Error in -> nChansWv ~= ich"); 
    end
    dtTime=1.0/ChanDataWv(1).SR;
    SR=1/dtTime;
   
         
%%
    nChsWv=length(ChanDataWv);
    if nChsWv ~= 2 
        warning('Number of Chans must be = 2');
        return;
    end
    
    clear ChsEvsData ChsTtl;
    for ich=1:nChsWv
        ChDta=ChanDataWv(ich).Data;
        ChTtl=ChanDataWv(ich).Title;
        SR   =ChanDataWv(ich).SR;
        ChsTtl(ich,1)={ChTtl};
        if NewSR > 0
            FsOrigin=SR;
            if  FsOrigin ~=  NewSR
                [fsorig, fsres] = rat(FsOrigin/NewSR);      
                ChDta=resample(ChDta,fsres,fsorig);   
                dtTime=1/NewSR;
            end
            SR=1.0/dtTime;
        end    
        if DcRemHpFlt > 0
            ChDta=ft_preproc_highpassfilter(ChDta,SR,DcRemHpFlt,2,'but',FltPassDir); % twopass onepass
        end        
        if Fhp > 0 
            ChDta=ft_preproc_highpassfilter(ChDta,SR,Fhp,2,'but',FltPassDir); % twopass
        end    
        if Flp > 0
            ChDta=ft_preproc_lowpassfilter(ChDta,SR,Flp,2,'but',FltPassDir);
        end

        if stShuffle == 1 && ich == 1
            nrnd=randperm(length(ChDta));
            ChDta=ChDta(nrnd);
        end

        if stShuffle == 2 && ich == 2
            nrnd=randperm(length(ChDta));
            ChDta=ChDta(nrnd);
        end
        
        if stShuffle == 3
            nrnd=randperm(length(ChDta));
            ChDta=ChDta(nrnd);
        end             
                
        stPlot.Plot=1;
        stPlot.fName=xFileName;
        stPlot.WvEvTtl=ChTtl;
        [EventTms,EvData,TmAxis]=GetEvTimeAndData(EventTms,ChDta,dtTime,tWidth,tOffset,stPlot);        
        
        [nEvsGood,nData]=size(EvData);
        if nEvsGood ~= nEvs
            disp("****  Warning:  nEvsGood ~= nEvs"); 
        end
        nEvs=nEvsGood;        
        ChsEvsData(ich,:,:)=EvData; 
    end % end of for ich=1:Nchs 
    disp("ChsEvsData is OK");   

%% Read in seperate ECG Chan Data 

    [MasData,EvEcgSR]=Smr64_ReadWaveChan(fid,ChanNumsECG);
    ECGDataWv.Data=double(MasData');
    ECGDataWv.Title=ECGChansTtl{1};
    ECGDataWv.SR=EvEcgSR;
    ecg_dtTime=1.0/ECGDataWv.SR;
    EcgSR=1/ecg_dtTime;
    if NewSR > 0
        FsOrigin=EcgSR;
        if  FsOrigin ~=  NewSR
            [fsorig, fsres] = rat(FsOrigin/NewSR);
            ECGData=resample(ECGDataWv.Data,fsres,fsorig);
            ecg_dtTime=1/NewSR;
        end
        EcgSR=1.0/ecg_dtTime;
    end
    if DcRemHpFlt > 0
        ECGData=ft_preproc_highpassfilter(ECGData,EcgSR,DcRemHpFlt,2,'but',FltPassDir); % twopass onepass
    end
    if Fhp > 0
        ECGData=ft_preproc_highpassfilter(ECGData,EcgSR,Fhp,2,'but',FltPassDir); % twopass
    end
    if Flp > 0
        ECGData=ft_preproc_lowpassfilter(ECGData,EcgSR,Flp,2,'but',FltPassDir);
    end
    
    stPlot.Plot=-1;
    stPlot.fName=xFileName;
    stPlot.WvEvTtl=ECGChansTtl;
    [EventTms,EvEcgData,TmAxis]=GetEvTimeAndData(EventTms,ECGData,ecg_dtTime,tWidth,tOffset,stPlot);
    disp("ECG Data is OK")

    CEDS64Close(fid);
%%    
%stZscoreWave=1;
    SpecTransTtl= "";
    if stZscoreWave > 0
        ChsEvsData=zscore(ChsEvsData,0,3); 
        SpecTransTtl=SpecTransTtl+", zScWv";
    end    
%%    
stMapsLinesEPs=1;
    clear EvTts;
    for iev=1:nEvs; EvTts(iev,1)={string(iev)}; end
    
    DataInfo.fName=xFileName+", EvCh: "+EvChansTtl+SpecTransTtl; 
    DataInfo.TmAxis=TmAxis;
    DataInfo.Data=ChsEvsData;
    DataInfo.WvTitle=ChsTtl;
    DataInfo.EvTts=EvTts;
    DataInfo.nRow=nRow;
    DataInfo.nCol=nCol;  
    stMinMax=-1; yxmin=-10; yxmax=10;
    if stZscoreWave >= 1; stMinMax=-1; yxmin=-2; yxmax=2; end
    DataInfo.styxminmax=stMinMax;
    DataInfo.yxmin=yxmin;
    DataInfo.yxmax=yxmax;

    nSmtEvWv = floor(tSmtEvWv*SR);    
    if nSmtEvWv > 1
        EvWvsTransTtls=sprintf(",  tSmt=%6.2f",tSmtEvWv);
        DataInfo.fName=DataInfo.fName+EvWvsTransTtls; 
        for ich=1:nChsWv
            for iev=1:nEvs
                dx=squeeze(DataInfo.Data(ich,iev,:));
                dx=smooth(dx,nSmtEvWv);
                DataInfo.Data(ich,iev,:)=dx;
            end
        end
    end
    
    DataInfo.stMapsLines=stMapsLinesEPs;       
    DataInfo.FigSize=[525 563 499 401]; %Monitor 1-> 4 623 652 493; Mon 2-> 1926 623 652 493
    [avechs]=plot_ChsEvs_Maps_Lines(DataInfo);    
%%    
    %[evaves,evstds]=plot_ChsEvs(DataInfo); 
%% Get Spectras
    if stIIRPeakSpec == 1
        ChsEvsFrsTm=Get_IIRPeak_ChsEvsFrsTm(ChsEvsData,ChsTtl,SR,Frqs,BandWidth,Qfac,FltPassDir);
        SpecTransTtl=SpecTransTtl+", IIRpeak, "+FltPassDir;
    else
        FtType='wavelet';    %  'mtmfft' 'mtmconvol' 'wavelet' 'tfr' 'hilbert'
        ChsEvsFrsTm=Get_FT_ChsEvsFrsTm(ChsEvsData,ChsTtl,SR,Frqs,WaveletnCyc,WaveletgWidth,FtType);
        SpecTransTtl=SpecTransTtl+", "+FtType;
    end
    [nChsWv,nEvs,nFrs,nData]=size(ChsEvsFrsTm);
    ChsEvsFrsTmSpc=zeros(nChsWv,nEvs,nFrs,nData);
    ChsEvsFrsTmPha=zeros(nChsWv,nEvs,nFrs,nData);
    for ich=1:nChsWv
        for iev=1:nEvs
            for ifr=1:nFrs
                xlb=squeeze(ChsEvsFrsTm(ich,iev,ifr,:));
                df=abs(xlb);
                ChsEvsFrsTmSpc(ich,iev,ifr,:)=df;
                ChsEvsFrsTmPha(ich,iev,ifr,:)=angle(xlb);
            end
        end
    end % end of for ich=1:nChsWv       
    disp("Spectra are OK  -------");      
%%  Calc PSI by trials and by Time
    CohTrsTtl="PSI Trials";
    if tCircMean  > 0  CohTrsTtl=sprintf("%s, tMean=%5.2f",CohTrsTtl,tCircMean); end
    Ch1EvsFrsTmPha=squeeze(angle(ChsEvsFrsTm(1,:,:,:)));
    Ch2EvsFrsTmPha=squeeze(angle(ChsEvsFrsTm(2,:,:,:)));

    [FrsTmPsiTrial,FrsTmPhaTrial]=Get_PSI_ByTrials(Ch1EvsFrsTmPha,Ch2EvsFrsTmPha,SR,tCircMean);
    disp("Get_PSI_ByTrials is OK");

    CohTimTtl="PSI Time";
    CohTimTtl=sprintf("%s, tCoh= %5.2f, tStep=%5.2f",CohTimTtl,tCohTm,tCohStep);
    [FrsTmPsiTime,FrsTmPhaTime,TmAxisCoh]=Get_PSI_ByTime(Ch1EvsFrsTmPha,Ch2EvsFrsTmPha,SR,TmAxis,tCohTm,tCohStep);
    disp("Get_PSI_ByTime is OK");


%% Calc Itc for one Channel by Trials 
    CohItcTtl="ITC Trials";
    if tCircMean  > 0  CohItcTtl=sprintf("%s, tMean=%5.2f",CohItcTtl,tCircMean); end

    [FrsTmItc1]=Get_PSI_ByTrials_ITC(Ch1EvsFrsTmPha,SR,tCircMean);
    [FrsTmItc2]=Get_PSI_ByTrials_ITC(Ch2EvsFrsTmPha,SR,tCircMean);
    fprintf("Get_ICT_ByTrials for %s and %s is OK \n", (ChsTtl{1}), (ChsTtl{2}))

%%
    clear plotinfo;
    plotinfo.fName=xFileName+", EvCh:"+EvChansTtl;
    plotinfo.dtTime=1.0/SR;
    plotinfo.nEvs=nEvs;
    plotinfo.EvTitle="";
    plotinfo.Title1=ChsTtl{1};
    plotinfo.Title2=ChsTtl{2};
    
    % include Ecg data
    plotinfo.EcgTitle=ECGChansTtl{1};
    plotinfo.EvEcgData=EvEcgData;

    % include Power data for PSD
    plotinfo.ChsEvsFrsTmSpc = ChsEvsFrsTmSpc
    
    plotinfo.SpecsTtl=SpecTransTtl;
    if stSpecLog > 0; plotinfo.SpecsTtl=plotinfo.SpecsTtl+", Log"; end
    
    %Initialize the basic EventData to plotinfo
    plotinfo.Ch1EvsData=squeeze(ChsEvsData(1,:,:));
    plotinfo.Ch2EvsData=squeeze(ChsEvsData(2,:,:));
    
    ChsFrsTmSpc=ChsEvsFrsTmSpc(1,:,:,:);
    ChsFrsTmSpc=squeeze(mean(ChsFrsTmSpc,2));
    plotinfo.PowData1=ChsFrsTmSpc;

    ChsFrsTmSpc=ChsEvsFrsTmSpc(2,:,:,:);
    ChsFrsTmSpc=squeeze(mean(ChsFrsTmSpc,2));
    plotinfo.PowData2=ChsFrsTmSpc;
    plotinfo.TmAxis=TmAxis;

    stPowerColorAxis=-1; PowClxMin=0; PowClxMax=10;
stZscoreSpc=-1;    
    if stZscoreSpc == 1
        plotinfo.PowData1=zscore(plotinfo.PowData1,0,2);
        plotinfo.PowData2=zscore(plotinfo.PowData2,0,2);
        stPowerColorAxis=1;
        PowClxMin=-2;
        PowClxMax=2;
        plotinfo.SpecsTtl=plotinfo.SpecsTtl+",  zSc";
    end
    
    plotinfo.CohTrsTtl=CohTrsTtl;
    plotinfo.CohTrsPsi=FrsTmPsiTrial;
    plotinfo.CohTrsPha=FrsTmPhaTrial;
    plotinfo.CohTrsAxis=TmAxis;
    
    plotinfo.CohTimTtl=CohTimTtl;
    plotinfo.CohTimPsi=FrsTmPsiTime;
    plotinfo.CohTimPha=FrsTmPhaTime;
    plotinfo.CohTimAxis=TmAxisCoh;

    % Intro the ITC for plotting
    plotinfo.CohItcTtl=CohItcTtl;
    plotinfo.CohTrsItc1=FrsTmItc1;
    plotinfo.CohTrsItc2=FrsTmItc2;
    plotinfo.CohItcAxis=TmAxis;
   
    
    stCohColorAxis=1; CohClxMin=0; CohClxMax=1;
    if stZscorePSI > 0
        plotinfo.CohTrsTtl=plotinfo.CohTrsTtl+",  zSc";
        plotinfo.CohTimTtl=plotinfo.CohTimTtl+",  zSc";
        plotinfo.CohTrsPsi=zscore(plotinfo.CohTrsPsi,0,2);
        plotinfo.CohTimPsi=zscore(plotinfo.CohTimPsi,0,2);
        stCohColorAxis=1; CohClxMin=-2; CohClxMax=2;
    end
    plotinfo.Frqs       =Frqs;
    plotinfo.tSmSpc=tSmSpc;
    plotinfo.tSmPsi=tSmPsi;
    plotinfo.tSmItc=tSmItc;
    plotinfo.SR=SR;
    plotinfo.stSpecLog=stSpecLog;
    plotinfo.stPowerColorAxis=stPowerColorAxis;
    plotinfo.PowClMin=PowClxMin;
    plotinfo.PowClMax=PowClxMax;
    plotinfo.stZscoreSpc=stZscoreSpc;
    plotinfo.stZscorePSI=stZscorePSI;
    plotinfo.stCohColorAxis=stCohColorAxis;
    plotinfo.CohClxMin=CohClxMin;
    plotinfo.CohClxMax=CohClxMax;
    plotinfo.ColorMap='jet'; % hsv jet;
    
    plotinfo.LinesFrq=[12 25 35];
    plotinfo.LinesClx=['k' 'r' 'g'];
    plotinfo.FrqMean=2;
    
    plotinfo.stMapsLines=stMapsLinesCoh; % 1- plot Maps, other - plot Lines
    plotinfo.FigSize=[1013 66 866 894]; % 45 611 918 484];
    u = plot_EvSpcCohPhas_DifferAxis_2ch(plotinfo);
    
    gr1 = fullfile('F:\HeadHeart\2_results\itc' ,strjoin({char(xFileName{1}), char(EegLfpChTts{1}), 'PSD-ERP_ITC_', char(EegLfpChTts{1}), '_PSI_', char(EegLfpChTts{1}), 'vs.', char(EegLfpChTts{2}),'.png'}, ''));
    exportgraphics(u,gr1, 'Resolution', 300)
    %
    %plotinfo.stMapsLines=2; % 1- plot Maps, other - plot Lines
    %plot_EvSpcCohPhas_DifferAxis_2ch(plotinfo);    
%%    
end % for isb=sbSt:sbEn

CEDS64CloseAll();
unloadlibrary ceds64int;

disp("*****  DONE   *****");