% Power Analysis


%%
% MEDICATION
% only one can be true at all times
MedOn = true;
MedOff = false;

% SUBJECT STATUS
% only one can be true at all times
newsubs = false;
oldsubs = false;
allsubs = true;

% get the channel info into the shape of cells
AllSubsChansRaw = cellfun(@(x) strsplit(x, ', '), {subject_info.channels_raw}, 'UniformOutput', false);
AllSubsChansStn = cellfun(@(x) strsplit(x, ', '), {subject_info.channels}, 'UniformOutput', false);
AllSubsOnlyStn = cellfun(@(x) strsplit(x, ', '), {subject_info.STN}, 'UniformOutput', false);
AllSubsOnlyEEG = cellfun(@(x) strsplit(x, ', '), {subject_info.EEG}, 'UniformOutput', false);

% filter which subjects and which channels you want
if MedOn == true & newsubs == true % Only New Subs that are MedOn
    subjects = string({subject_info([subject_info.new] == 1 & [subject_info.MedOn] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.new] == 1 & [subject_info.MedOn] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.new] == 1 & [subject_info.MedOn] == 1);
    FltSubsOnlyStn = AllSubsOnlyStn([subject_info.new] == 1 & [subject_info.MedOn] == 1);
    FltSubsOnlyEEG = AllSubsOnlyEEG([subject_info.new] == 1 & [subject_info.MedOn] == 1);
elseif MedOff == true & newsubs == true  % Only New Subs that are MedOff
    subjects = string({subject_info([subject_info.new] == 1 & [subject_info.MedOff] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.new] == 1 & [subject_info.MedOff] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.new] == 1 & [subject_info.MedOff] == 1);
    FltSubsOnlyStn = AllSubsOnlyStn([subject_info.new] == 1 & [subject_info.MedOff] == 1);
    FltSubsOnlyEEG = AllSubsOnlyEEG([subject_info.new] == 1 & [subject_info.MedOff] == 1);
elseif MedOn == true & oldsubs == true  % Only Old Subs that are MedOn
    subjects = string({subject_info([subject_info.new] == 0 & [subject_info.MedOn] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.new] == 0 & [subject_info.MedOn] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.new] == 0 & [subject_info.MedOn] == 1);
    FltSubsOnlyStn = AllSubsOnlyStn([subject_info.new] == 0 & [subject_info.MedOn] == 1);
    FltSubsOnlyEEG = AllSubsOnlyEEG([subject_info.new] == 0 & [subject_info.MedOn] == 1);
elseif MedOff == true & oldsubs == true  % Only Old Subs that are MedOff
    subjects = string({subject_info([subject_info.new] == 0 & [subject_info.MedOff] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.new] == 0 & [subject_info.MedOff] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.new] == 0 & [subject_info.MedOff] == 1);
    FltSubsOnlyStn = AllSubsOnlyStn([subject_info.new] == 0 & [subject_info.MedOff] == 1);
    FltSubsOnlyEEG = AllSubsOnlyEEG([subject_info.new] == 0 & [subject_info.MedOff] == 1);
elseif MedOn == true & allsubs == true  % All Subs that are MedOn
    subjects = string({subject_info([subject_info.MedOn] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.MedOn] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.MedOn] == 1);
    FltSubsOnlyStn = AllSubsOnlyStn([subject_info.MedOn] == 1);
    FltSubsOnlyEEG = AllSubsOnlyEEG([subject_info.MedOn] == 1);
elseif MedOff == true & allsubs == true % All Subs that are MedOff
    subjects = string({subject_info([subject_info.MedOff] == 1).ID});
    FltSubsChansStn = AllSubsChansStn([subject_info.MedOff] == 1);
    FltSubsChansRaw = AllSubsChansRaw([subject_info.MedOff] == 1);
    FltSubsOnlyStn = AllSubsOnlyStn([subject_info.MedOff] == 1);
    FltSubsOnlyEEG = AllSubsOnlyEEG([subject_info.MedOff] == 1);
end

%=========================================================================
%% Parameters

% Downsample Parameters
NewSR=300;
stfr=0.5;   enfr=30; dfr=0.2;
Frqs=stfr:dfr:enfr;
if NewSR < 2*enfr; NewSR=(enfr+2)*2; end

% Flag if only EEG, STN or all channels
allchans = true;
onlyeeg = false;
onlystn = false;


% Hilbert TFR Parameters
BandWidth=2; % BandWidth in Hz;
Qfac     =2; % Attenuation in db(-Qfac)
WaveletnCyc=6;
WaveletgWidth=3;
FltPassDir='twopass'; % onepass

% Baseline und Epoch Parameter
ChsCmxEvFrTm =[]; %
baseline_win = [-0.3 -0.1]; % Baseline Time Window

% Define TFR Time Window
tWidth   = 0.8;
tOffset  = 0.4;

% Generating MedName
if MedOn == true
    medname = 'MedOn';
elseif MedOff == true
    medname = 'MedOff';
end

Fhp = 2;

% Use BPReref Data
BPReref = true; BPRerefTit = 'BPReref';
BPRerefHi = true; BPRerefHiTit = 'BPRerefHi';
BPRerefLw = false; BPRerefLwTit = 'BPRerefLow';
BPRerefBest = false; BPRerefBestTit = 'BPRerefBest';

%% Loop

for sub = 1:numel(subjects) % BE AWARE THAT THIS EXCLUDES PATIENTS WITH ARRITHYMIAS
    % Extract the subject
    subject = subjects{sub};

    fprintf('Loading Data of subject %s number %i of %i\n', subject, sub, numel(subjects));

    pattern = fullfile(data_dir, 'preproc', 'all', [subject, '_', 'preprocessed', '_', medname, '_BPReref_', '*']);
    files = dir(pattern);
    filename = fullfile(files(1).folder, files(1).name);
    load(filename, 'SmrData');
    % Load subject data
    % subject_data = fullfile(data_dir, preprocessed_name, medname, ['sub-', subject], [subject, '_preprocessed_', medname, '_Rest.mat']);
    % load(subject_data, 'SmrData');

    % pattern = fullfile(data_dir, 'ecg', 'ss' ,[subject, '_EpochECGEvData_', medname, '*']);
    % files = dir(pattern);
    % filename = fullfile(files(1).folder, files(1).name);
    % load(filename, 'EvECG');

    % Load the the cleaned ECG R Peaks Data
    fprintf('Loading ECG Data\n');
    pattern = fullfile(data_dir, 'ecg', 'ss' ,[subject, '_EpochECGEvData_', medname, '*']);
    files = dir(pattern);
    filename = fullfile(files(1).folder, files(1).name);
    load(filename, 'EvECG');

    SR = SmrData.SR;
    EventTms = SmrData.EvData.EvECGP_Cl;

    if allchans
        channels = FltSubsChansStn{sub};
    elseif onlyeeg
        channels = FltSubsOnlyEEG{sub};
    elseif onlystn
        channels = FltSubsOnlyStn{sub};
    end

    % KS29 has no EEG recordings in MedOn so we delete those values
    if strcmp(medname,'MedOn') & strcmp(subject,'KS29')
        channels = {FltSubsChansStn{sub}{end-1:end}};
        SmrData.WvDataBPRerefHi(1:6,:) = [];
        SmrData.WvDataBPRerefLow (1:6,:) = [];
    end

    for c = 1:numel(channels)
        channel = channels{c};


        fprintf('****************** EPOCH for %s %s...****************\n', subject, channel);

        if BPReref & BPRerefHi
            ChDta = SmrData.WvDataBPRerefHi(c, :);
        elseif BPReref & BPRerefLw
            ChDta = SmrData.WvDataBPRerefLow(c, :);
        elseif BPReref & BPRerefBest
            ChDta = SmrData.WvDataBPRerefLow(c, :); % Hier filter oder vielleicht doch schon im Preprocessing später!!
        else
            ChDta = SmrData.WvDataCleaned(c, :);
        end

        % HIGH PASS FILTER
        if Fhp > 0
            ChDta=ft_preproc_highpassfilter(ChDta,SR,Fhp,4,'but', 'twopass'); % twopass
        end

        % DOWNSASMPLE
        if NewSR > 0
            FsOrigin=SR;
            if  FsOrigin ~=  NewSR
                [fsorig, fsres] = rat(FsOrigin/NewSR);
                ChDta=resample(ChDta,fsres,fsorig);
                dtTime=1/NewSR;
            end
            NewSR=1.0/dtTime;
        end

        % Epoch the Data
        [EventTms,EvData,TmAxis]=GetEvTimeAndData(EventTms,ChDta,dtTime,tWidth,tOffset);
        [nEvs,nData]=size(EvData);

        fprintf('****************** Baseline Correction for %s %s med: %s ...****************\n', subject, channel, medname);
        % My baseline is the time window -0.3 to -0.1 s
        % before my Rpeak of every trial

        % % Find the the indices for the basseline window
        % bidx = find(TmAxis' >= baseline_win(1) & TmAxis' <= baseline_win(2));
        %
        % % for every trial calc the mean of the baseline win
        % % and subtract that from the entire epoch
        % for t = 1:nEvs
        %     baseline_mean = mean(EvData(t, bidx(1):bidx(end)),2);
        %     EvData(t,:) = EvData(t,:)-baseline_mean;
        % end

        % for every trial calculate the mean of the entire epoch
        % and subtract that from the trial
        for t = 1:nEvs
            epoch_mean = mean(EvData(t,:), 2);   % average over all time points
            EvData(t,:) = EvData(t,:) - epoch_mean;
        end


        fprintf('****************** TIME FREQ DECOMP for %s %s, med: %s ...****************\n', subject, channel, medname);
        %TFR = []; % time frequency representation
        ChsCmxEvFrTm = [];

        % Get IIR Peak
        for iev=1:nEvs
            dx=squeeze(EvData(iev,:));
            for ifr=1:length(Frqs)
                vfr=Frqs(ifr);
                df=IIRPeak_Flt(dx,NewSR,vfr,BandWidth,Qfac,FltPassDir);
                ChsCmxEvFrTm(iev,ifr,:)=hilbert(df); % ChannelxFreqxTime
            end
        end

        % EXTRACTION OF POWER AND PHASE
        [nEvs, nFrs,nData]=size(ChsCmxEvFrTm);
        ChsAllFrsTmSpc=zeros(nEvs,nFrs,nData);
        ChsAllFrsTmPha=zeros(nEvs,nFrs,nData);
        for iev=1:nEvs
            for ifr=1:nFrs
                xlb=squeeze(ChsCmxEvFrTm(iev,ifr,:));
                df=abs(xlb);
                ChsAllFrsTmSpc(iev,ifr,:)=df; % Power (eventxfreqxpower)
                ChsAllFrsTmPha(iev,ifr,:)=angle(xlb); %Phase (eventxfreqxphase)
            end
        end

        f2 = figure;
        imagesc(TmAxis,Frqs,squeeze(ChsAllFrsTmSpc(1,:,:))), axis xy
        title(['Power Plot, Sub: ', subject, ' med: ', medname, ' chan: ', channel])
        colorbar;

        gr2 = fullfile( results_dir, 'power/2Hz/ss' , ['POWER_', subject, '_', medname, '_', channel, '_EPOCH= -', num2str(tOffset), '-', num2str(tWidth), '_BL=epoch-average' '.png']);
        exportgraphics(f2, gr2, "Resolution", 300);

    end
end


