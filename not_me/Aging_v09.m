tic

%%% SCRIPT TO STUDY AGING PROCESS IN LEARNING IN MODULES DATAS
all_attempts=zeros(1,4);
for dataset = [ 1 4 3 2  ]
    clearvars -except all_attempts dataset

    disp('...loading')
    if dataset==1
        load D:\MATLAB\DATABASE\mkdb_ref.mat
        %          mkid= [ mkid(1:59,:); mkid(59,:); mkid(60:61,:)];
        if ~strcmp(mkid.name{60},'fuckit')
            mkid2=mkid;
            mkid2.id(60,:)= 60; mkid2.name{60}='fuckit';
            mkid2(61:end,:) = [];
            mkid2 = [mkid2; mkid(60:end,:)];
            mkid=mkid2; mkid2=[];
        end

    elseif dataset==2
        load D:\MATLAB\DATABASE\mkdb_ref_cov.mat
    elseif dataset==3
        load D:\MATLAB\DATABASE\mkdb_ref_j.mat
        if ~strcmp(mkid.name{5},'fuckit')
            mkid2=mkid;
            mkid2.id(5,:)= 5; mkid2.name{5}='fuckit';
            mkid2(6:end,:) = [];
            mkid2 = [mkid2; mkid(5:end,:)];
            mkid=mkid2; mkid2=[];
        end
    elseif dataset==4
        load D:\MATLAB\DATABASE\mkdb_ref_rh.mat
        mkid.birthdate=cell(size(mkid,1),1);
        %         mkid.birthdate(1)={'2007-05-06'};
        %         mkid.birthdate(3)={'2007-06-06'};
        mkid.birthdate(8)={'2015-07-03'};
    end

    % UPDATING MKID
    % DATE OF BIRTH (NO ULYSSE AND TOO YOUNG)
    timeratio=24*60*60*1000;
    refdate=datenum(1970,01,01); % day*hours*min*10msec
    %
    % mkBirth=[datenum(1995,12,14,0,0,1); datenum(1997,10,11,0,0,1) ; datenum(1999,01,15,0,0,1); datenum(2000,08,25,0,0,1) ...
    %     ; datenum(2001,08,01,0,0,1) ; datenum(2009,07,12,0,0,1)  ; datenum(2009,07,23,0,0,1)  ; datenum(2011,01,05,0,0,1) ...
    %     ; datenum(2011,04,06,0,0,1)  ;  datenum(2014,11,13,0,0,1);  NaN; datenum(2007,05,29,0,0,1) ...
    %     ; datenum(2007,06,30,0,0,1)  ; datenum(2007,10,24,0,0,1)  ; datenum(2009,04,10,0,0,1)  ; datenum(2009,11,11,0,0,1)...
    %     ; datenum(2010,11,29,0,0,1)  ; datenum(2011,07,17,0,0,1) ; datenum(2012,01,23,0,0,1) ; datenum(2012,02,06,0,0,1) ...
    %     ; datenum(2013,02,12,0,0,1) ; datenum(2013,03,19,0,0,1) ; datenum(2013,05,14,0,0,1) ; datenum(2013,05,21,0,0,1)...
    %     ; datenum(2014,12,05,0,0,1) ; datenum(2015,06,05,0,0,1); NaN ; NaN ; datenum(2016,03,22,0,0,1) ; datenum(2017,03,28,0,0,1)...
    %     ; datenum(2018,02,21,0,0,1); NaN; NaN; datenum(2019,02,21,0,0,1);]; % ADD GANDHI BIRTH DATE

    % T1 = datenum(2019,11,29);
    % mkAge=[];
    % % T2 = datenum(2019,10,28);
    % for n=1:numel(mkBirth)
    %     T2 = mkBirth(n);
    %     mkAge(n,1)=(T1-T2)/365;
    %     try
    %         mkid.Birth(n,:) = datestr(mkBirth(n));
    %     catch
    %         mkid.Birth(n,:) = NaN;
    %     end
    % end

    % mkid.Gender=ones(size(mkid.id)); mkid.Gender([1:10 29 31])=2;
    mkname=mkid.name; mkgender = mkid.gender;

    delid= [ ];
    % delid= [ 27 28 32 33 ]; % Without  Human RFID
    % delid= [4 10 11 25 28 32 33];
    % delid= [ 9 11 25 28 32 33]; % WITHOUT LASSA
    % delid= [ 27 28 32 33 ]; % WITHOUT HUMANS TEST RFID
    mkname(delid,:)=[];
    mkgender(delid,:)=[];
    % mkAge(delid,:)=[];

    %%% CSRT: Start: first CSRT_T01 (88) TO first: TAV_T01 (123) -1
    %%% TAV: Start: first TAV_T01 (123) TO first: DMS_T01 (85) -1
    %%% CSRT: Start: first DMS_T01 (85) TO first: DMS_T04 (123)
    % %
    % % % END OF LEARNING DMS:
    % % % DMS_T04= {ID=3080 Battery=4 Level=33 Session=31 Task=101}

    mk_dms=[]; mk_csrt=[]; mk_tav=[]; firstdate=[]; lastdate=[];

    if dataset ~=3
        t_tp = [3 10]; %3 % 14
        t_csrt = [14 21];
        t_tav = [26 29];
        t_dms = [30 33]; %33
    else
        t_tp = [3 14];
        t_csrt = [14 21];
        t_tav = [26 28];
        t_dms = [30 33]; %33
    end

    % t_dms = [82 83];
    %  t_dms = [126 132]; % SOCIAL ?


    % PROGRESSION ID % FOR HM
    % t_tp = [4 14];
    % t_csrt = [14 21];
    % t_tav = [26 29];
    % t_dms = [30 34];


    % % PROGRESSION ID
    % t_tp = [4 14];
    % t_csrt = [14 21];
    % t_tav = [26 29];
    % t_dms = [30 33];

    % t_soss = [30 33];
    % t_pal = [30 33];
    % t_eco = [30 33];
    % t_dom = [30 33];


    % ADD POSSIBILITY TO FILTER TASK !!!!
    % ADD FEEDBACK on Pallier Name


    %
    % % OLD TASKID
    % t_tp = [80 82];  % t_tp = [77 88];
    % t_tp = [81 4];
    % t_csrt = [88 98]; % t_csrt=[88 123];
    % t_tav = [123 85];
    % t_dms = [85 101]; % t_dms = [85 103];
    %
    % % % HM
    % % t_tp = [80 114];
    % % t_csrt = [88 97];
    % % t_tav = [123 129];
    % % t_dms = [85 106];
    check=[];
    NMK=size(mkid,1);

    mmRT1 = ones(NMK,1).*NaN;
    mmRT2 = ones(NMK,1).*NaN;
    mmRT3 = ones(NMK,1).*NaN;

    % ONLY CONSOLIDATED
    mTP = [];
    mmTP = ones(NMK,1).*NaN;
    TPp = [  5 6 7 8  9 ;
        4 110 111 112 113 ];

    mCSRT = [];
    mmCSRT = ones(NMK,1).*NaN;
    mmCSRTi = ones(NMK,1).*NaN;
    CSRTp = [ 10 11 ;
        97 98 ];

    mDMS = [];
    mmDMS = ones(NMK,1).*NaN;
    DMSp = [ 2 ;
        99];

%     mDMS = [];
%     mmDMS = ones(NMK,1).*NaN;
%     DMSp = [ 6 ;
%         103];

    mTAV= [];
    mmTAV = ones(NMK,1).*NaN;
    TAVp = [   2   ; %BEST 2
         124]; % BEST 124

    mSOSS= [];
    mmSOSS = ones(NMK,1).*NaN;
    SOSSp = [  4 5 6 7 ;
        118 119 120 121 ];
    %
    mPAL= []; % PAL1
    mmPAL = ones(NMK,1).*NaN;
    PALp = [ 4 5 6  ;
        106 107 108 ];


    %     mPAL= []; % PAL 2
    %     mmPAL = ones(NMK,1).*NaN;
    %     PALp = [ 3 5 6 7  ;
    %         186 187 188 189];

% TRAINING
  mECO= []; % GAIN 
    mmECO = ones(NMK,1).*NaN;
    ECOp = [ 2;
        196 ];

    mECOl= []; % Perte
    mmECOl = ones(NMK,1).*NaN;
    ECOlp = [ 3;
        197 ];

    mECOp= []; % Probaba
    mmECOp = ones(NMK,1).*NaN;
    ECOpp = [ 4;
        198 ];

%     % TEST
%     mECO= []; % GAIN
%     mmECO = ones(NMK,1).*NaN;
%     ECOp = [ 2;
%         206 ];
% 
%     mECOl= []; % Perte
%     mmECOl = ones(NMK,1).*NaN;
%     ECOlp = [ 3;
%         207 ];
% 
%     mECOp= []; % Probaba
%     mmECOp = ones(NMK,1).*NaN;
%     ECOpp = [ 4;
%         208 ];

RTtaskp = [ CSRTp DMSp, SOSSp ]; 

    for mk=1:NMK
        disp(mkid.name{mk})

        mk=mkid.id(mk);
        try
            %         if mk<=5
            %            doerror
            %         end
            mkid_save=mkid;

            if dataset==1
                eval(['load D:\MATLAB\DATABASE\monkdb_ID' num2str(mk) '.mat'])
                eval(['load D:\MATLAB\DATABASE\monkdb_ID_step' num2str(mk) '.mat'])
            elseif dataset==2
                eval(['load D:\MATLAB\DATABASE\monkdb_ID' num2str(mk) '_cov.mat'])
                eval(['load D:\MATLAB\DATABASE\monkdb_ID_step' num2str(mk) '_cov.mat'])
            elseif dataset==3
                eval(['load D:\MATLAB\DATABASE\monkdb_ID' num2str(mk) '_j.mat'])
                eval(['load D:\MATLAB\DATABASE\monkdb_ID_step' num2str(mk) '_j.mat'])
            elseif dataset==4
                eval(['load D:\MATLAB\DATABASE\monkdb_ID' num2str(mk) '_rh.mat'])
                eval(['load D:\MATLAB\DATABASE\monkdb_ID_step' num2str(mk) '_rh.mat'])
            end

            mkid=mkid_save;

%%%%% CALCULATE RT
 % RT GAIN
%             keyboard
            rtattempt = attempt(ismember(attempt.task,RTtaskp(2,:)),:) ;
            rtstep = step(ismember(step.attempt,rtattempt.id),:) ;
%             mask = strcmp(rtstep.type,'StartStep') ;
 mask = strcmp(rtstep.name,'answer') ;
            rata=rtstep.instant_end(mask)-rtstep.instant_begin(mask) ;
            nr = round(numel(rata)*0.05);
            rata2=rata;
            rata2=rmoutliers(rata,'quartiles');

            mmRT1(mk)=nanmean(rata2)/10;
            mmRT2(mk)=nanmedian(rata2)/10;
            mmRT3(mk)=nanstd(rata2)/10;

%             irata=sort(rata); irata=mean(irata(1:nr));
      
%             med = nanmedian(rata) ;
%                         rata(rata<500)=[]; % removing to fast responses


%             rata2=rata-irata;
%             rata2(rata2<=0)=[];

            %             subplot(1,2,1); histogram(rata);
            %             subplot(1,2,2); histogram(rata2);
            % keyboard
            %             mmRT1(mk)=nanmean(rata2(rata2<med));
            %             mmRT1(mk)=nanmedian(rata2);
            %             mmRT2(mk)=nanmean(rata2(rata2>med));
% 
%             gmdl = fitgmdist(rata2,3);
%             [mu]=sort(gmdl.mu);
% 
%             mmRT1(mk)=mu(1)/10;
%             mmRT2(mk)=mu(2)/10;
%             mmRT3(mk)=mu(3)/10;

%             mmRT1(mk)=nanmean(rata2);
%             mmRT2(mk)=nanmedian(rata2);
%             mmRT3(mk)=nanstd(rata2);

            %%% CALCULATE SUCCES RATE FOR ALL TASK
            limstart = 100;
            limtrial = 500 + limstart; %250 1000 3000
            
            alltask = unique(attempt.task);

            for i = 1: numel(alltask)

                at=attempt;
                mask = find(at.task==alltask(i));

                try
                    mask= mask(limstart:limtrial) ;
                    
                    %                     mask=mask(idx);
                catch
                   if numel(mask)>(limstart *2)
                    mask= mask(limstart:end) ;
                   end
disp([ '... not enought trial in task ' num2str(alltask(i)) ]);
                    disp(numel(mask));
                end

                at=at(mask,:);
% keyboard
                alltask(i,2) = mean(strcmp(at.result,'success'));
                alltask(i,3) = numel(mask);
                alltask(i,4) = mean(strcmp(at.result,'error'));
                alltask(i,5) = mean(strcmp(at.result,'prematured'));
                alltask(i,6) = mean(strcmp(at.result,'stepomission'));
            end



            % TP
            for z = 1 : size(TPp,2)
                idx = find(alltask(:,1)==TPp(2,z)) ;
                if ~isempty(idx)
                    mTP{1}(mk,z) =  alltask (idx,2);
                    mTP{2}(mk,z) =  alltask (idx,3);
                else
                    mTP{1}(mk,z) = NaN;
                    mTP{2}(mk,z) = NaN;
                end
            end
            mmTP(mk)=nanmean(mTP{1}(mk,:));

            % CSRT Attention
            for z = 1 : size(CSRTp,2)
                idx = find(alltask(:,1)==CSRTp(2,z)) ;
                if ~isempty(idx)
                    mCSRT{1}(mk,z) =  1- (alltask (idx,4) + alltask (idx,6)) ;
                    mCSRT{2}(mk,z) =  alltask (idx,3);
                else
                    mCSRT{1}(mk,z) = NaN;
                    mCSRT{2}(mk,z) = NaN;
                end
            end
            mmCSRT(mk)=nanmean(mCSRT{1}(mk,:));


            % CSRT Inhibition
            for z = 1 : size(CSRTp,2)
                idx = find(alltask(:,1)==CSRTp(2,z)) ;
                if ~isempty(idx)
                    mCSRTi{1}(mk,z) =  1-alltask (idx,5);
                    mCSRTi{2}(mk,z) =  alltask (idx,3);
                else
                    mCSRTi{1}(mk,z) = NaN;
                    mCSRTi{2}(mk,z) = NaN;
                end
            end
            mmCSRTi(mk)=nanmean(mCSRTi{1}(mk,:));

            % DMS
            for z = 1 : size(DMSp,2)
                idx = find(alltask(:,1)==DMSp(2,z)) ;
                if ~isempty(idx)
                    mDMS{1}(mk,z) =  alltask (idx,2);
                    mDMS{2}(mk,z) =  alltask (idx,3);
                else
                    mDMS{1}(mk,z) = NaN;
                    mDMS{2}(mk,z) = NaN;
                end
            end
            mmDMS(mk)=nanmean(mDMS{1}(mk,:));


            % TAV
            for z = 1 : size(TAVp,2)
                idx = find(alltask(:,1)==TAVp(2,z)) ;
                if ~isempty(idx)
                    mTAV{1}(mk,z) =  alltask (idx,2);
                    mTAV{2}(mk,z) =  alltask (idx,3);
                else
                    mTAV{1}(mk,z) = NaN;
                    mTAV{2}(mk,z) = NaN;
                end
            end
            mmTAV(mk)=nanmean(mTAV{1}(mk,:));


            % PAL
            for z = 1 : size(PALp,2)
                idx = find(alltask(:,1)==PALp(2,z)) ;
                if ~isempty(idx)
                    mPAL{1}(mk,z) =  alltask (idx,2);
                    mPAL{2}(mk,z) =  alltask (idx,3);
                else
                    mPAL{1}(mk,z) = NaN;
                    mPAL{2}(mk,z) = NaN;
                end
            end
            mmPAL(mk)=nanmean(mPAL{1}(mk,:));

            % SOSS
            for z = 1 : size(SOSSp,2)
                idx = find(alltask(:,1)==SOSSp(2,z)) ;
                if ~isempty(idx)
                    mSOSS{1}(mk,z) =  alltask (idx,2);
                    mSOSS{2}(mk,z) =  alltask (idx,3);
                else
                    mSOSS{1}(mk,z) = NaN;
                    mSOSS{2}(mk,z) = NaN;
                end
            end
            mmSOSS(mk)=nanmean(mSOSS{1}(mk,:));


            % ECO GAIN
            for z = 1 : size(ECOp,2)
                idx = find(alltask(:,1)==ECOp(2,z)) ;
                if ~isempty(idx)
                    mECO{1}(mk,z) =  alltask (idx,2);
                    mECO{2}(mk,z) =  alltask (idx,3);
                else
                    mECO{1}(mk,z) = NaN;
                    mECO{2}(mk,z) = NaN;
                end
            end
            mmECO(mk)=nanmean(mECO{1}(mk,:));

            % ECO LOSS
            for z = 1 : size(ECOlp,2)
                idx = find(alltask(:,1)==ECOlp(2,z)) ;
                if ~isempty(idx)
                    mECOl{1}(mk,z) =  alltask (idx,2);
                    mECOl{2}(mk,z) =  alltask (idx,3);
                else
                    mECOl{1}(mk,z) = NaN;
                    mECOl{2}(mk,z) = NaN;
                end
            end
            mmECOl(mk)=nanmean(mECOl{1}(mk,:));

            % ECO GAIN
            for z = 1 : size(ECOpp,2)
                idx = find(alltask(:,1)==ECOpp(2,z)) ;
                if ~isempty(idx)
                    mECOp{1}(mk,z) =  alltask (idx,2);
                    mECOp{2}(mk,z) =  alltask (idx,3);
                else
                    mECOp{1}(mk,z) = NaN;
                    mECOp{2}(mk,z) = NaN;
                end
            end
            mmECOp(mk)=nanmean(mECOp{1}(mk,:));


           
            % % % %   FIND BACK PROGRESS IDs
            p_tp=[]; p_csrt=[]; p_tav=[]; p_dms=[];
            TP=[]; TAV=[]; CSRT=[]; DMS=[];
            for n=1:2
                % PER TASK ID
                %             id=find(progress.task==t_tp(n),1,'first'); p_tp(n)=progress.id(id);
                %             id=find(progress.task==t_csrt(n),1,'first'); p_csrt(n)=progress.id(id);
                %             id=find(progress.task==t_tav(n),1,'first'); p_tav(n)=progress.id(id);
                %             id=find(progress.task==t_dms(n),1,'first'); p_dms(n)=progress.id(id);

                % PER LEVEL
                try
                    id=find(progress.level==t_tp(n),1,'first'); p_tp(n)=progress.id(id);
                catch;  p_tp=[NaN NaN]; end
                try
                    id=find(progress.level==t_csrt(n),1,'first'); p_csrt(n)=progress.id(id);
                catch;  p_csrt=[NaN NaN]; end
                try
                    id=find(progress.level==t_tav(n),1,'first'); p_tav(n)=progress.id(id);
                catch;  p_tav=[NaN NaN]; end
                try
                    id=find(progress.level==t_dms(n),1,'first'); p_dms(n)=progress.id(id);
                catch;  p_dms=[NaN NaN]; end

            end

            % CHECKING DATA INTEGRITY
            for k=1:diff(t_tp)+1
                id=find(progress.level==(t_tp(1)+k-1),1,'first'); if ~isempty(id); tmp=find(attempt.progression==progress.id(id),1,'first'); ...
                        if ~isempty(tmp);check{1}(mk,k)=tmp; else; check{1}(mk,k)=NaN; end; else; check{1}(mk,k)=NaN; end
            end
            for k=1:diff(t_csrt)+1
                id=find(progress.level==(t_csrt(1)+k-1),1,'first'); if ~isempty(id); tmp=find(attempt.progression==progress.id(id),1,'first');...
                        if ~isempty(tmp); check{2}(mk,k)=tmp; else; check{2}(mk,k)=NaN; end; else; check{2}(mk,k)=NaN; end
            end
            for k=1:diff(t_tav)+1
                id=find(progress.level==(t_tav(1)+k-1),1,'first'); if ~isempty(id); tmp=find(attempt.progression==progress.id(id),1,'first');...
                        if ~isempty(tmp); check{3}(mk,k)=tmp; else; check{3}(mk,k)=NaN; end; else; check{3}(mk,k)=NaN; end
            end
            for k=1:diff(t_dms)+1
                id=find(progress.level==(t_dms(1)+k-1),1,'first'); if ~isempty(id); tmp=find(attempt.progression==progress.id(id),1,'first');...
                        if ~isempty(tmp); check{4}(mk,k)=tmp; else; check{4}(mk,k)=NaN; end; else; check{4}(mk,k)=NaN; end
            end


            % FIND WHEN MONKEY FINISH TO LEARN DMS
            %               keyboard
            for n=1:2
                %TP
                tmp=find(attempt.progression==p_tp(n),1,'first');
                if ~isempty(tmp)
                    TP(n)=tmp;
                    if n==1
                        firstdate(mk,1) =((attempt.instant_begin(tmp))/timeratio) + refdate; % CONVERTION FROM MYSQL TIME TO MATLAB TIME !
                    else
                        lastdate(mk,1) =((attempt.instant_begin(tmp))/timeratio) + refdate; % CONVERTION FROM MYSQL TIME TO MATLAB TIME !
                    end
                else
                    TP(n)=NaN;
                    if n==1
                        firstdate(mk,1)=NaN;
                    else
                        lastdate(mk,1)=NaN;
                    end
                end

                %CSRT
                tmp=find(attempt.progression==p_csrt(n),1,'first');
                if ~isempty(tmp)
                    CSRT(n)=tmp;
                    if n==1
                        firstdate(mk,2) =((attempt.instant_begin(tmp))/timeratio) + refdate;
                    else
                        lastdate(mk,2) =((attempt.instant_begin(tmp))/timeratio) + refdate;
                    end
                else
                    CSRT(n)=NaN;
                    if n==1
                        firstdate(mk,2)=NaN;
                    else
                        lastdate(mk,2)=NaN;
                    end
                end

                %TAV
                tmp=find(attempt.progression==p_tav(n),1,'first');
                if ~isempty(tmp)
                    TAV(n)=tmp;
                    if n==1
                        firstdate(mk,3) =((attempt.instant_begin(tmp))/timeratio) + refdate;
                    else
                        lastdate(mk,3) =((attempt.instant_begin(tmp))/timeratio) + refdate;
                    end
                else
                    TAV(n)=NaN;
                    if n==1
                        firstdate(mk,3)=NaN;
                    else
                        lastdate(mk,3)=NaN;
                    end
                end

                % DMS
                tmp=find(attempt.progression==p_dms(n),1,'first');
                if ~isempty(tmp)
                    DMS(n)=tmp;
                    if n==1
                        firstdate(mk,4) =((attempt.instant_begin(tmp))/timeratio) + refdate;
                    else
                        lastdate(mk,4) =((attempt.instant_begin(tmp))/timeratio) + refdate;
                    end
                else
                    DMS(n)=NaN;
                    if n==1
                        firstdate(mk,4)=NaN;
                    else
                        lastdate(mk,4)=NaN;
                    end
                end

            end

            mk_tp(mk)=diff(TP);
            mk_csrt(mk)=diff(CSRT);
            mk_tav(mk)=diff(TAV);
            mk_dms(mk)=diff(DMS);

            ALLTP(mk,:) = TP;

        catch
%                                 keyboard
            disp( ['Bad data mk: ', num2str(mk), ' : ', mkid.name{mk} ] )
            mk_tp(mk)=NaN;         mk_csrt(mk)=NaN;         mk_tav(mk)=NaN;         mk_dms(mk)=NaN;
            firstdate(mk,:)=zeros(1,4)*NaN; lastdate(mk,:)=zeros(1,4)*NaN;
            ALLTP(mk,:)=[NaN NaN];
        end
        all_attempts(dataset)=all_attempts(dataset)+size(attempt,1);

    end

    mkAge=[]; mkAge_last=[]; mkAge_first=[]; mkDateFirst=cell(1,NMK);
    % CALCULATED AGE OF THE MONKEY AT FIRST TRIAL OF EACH TASK
    for n=1:NMK
        n=mkid.id(n);
        try
            T1= datestr(datevec(firstdate(n,2)),'yyyy-mm-dd'); % AGE at FIRST TRIAL
            mkDateFirst{n}=T1;
            T2 = datevec(mkid.birthdate(n));
        catch
            mkAge_last(n,k)=NaN;
            mkAge_first(n,k)=NaN;
            mkAge(n,k)=NaN;
            continue
        end

        for k=1:size(firstdate,2)
            T1= datevec(firstdate(n,k)); % AGE at FIRST TRIAL
            mkAge_first(n,k)=etime(T1,T2)/60/60/24/365;
            T1= datevec(lastdate(n,k));  % AGE at LAST TRIAL
            mkAge_last(n,k)=etime(T1,T2)/60/60/24/365;

            mkAge(n,k)= min([mkAge_first(n,k) mkAge_last(n,k)]);

        end
    end
    mkid.datefirst=mkDateFirst';

    % mkAge=(mkAge_first+mkAge_last)/2;
    % mkAge=mkAge_first;
    % mkAge=mkAge_last;

    % CLEAN PERF
    if dataset ==1
        deldel= [ 27 28 32 33 59 38 40 45 65]; % Without  Human RFID or ID with not enought trials
        mmTP( deldel )=NaN; mmTP(mmTP>=0.97|mmTP==0)=NaN;
        mmCSRT( deldel )=NaN; mmCSRT(mmCSRT>=0.97|mmCSRT==0)=NaN;
        mmCSRTi( deldel )=NaN; mmCSRTi(mmCSRTi>=0.97|mmCSRTi==0)=NaN;
        mmDMS( deldel )=NaN; mmDMS(mmDMS>=0.97|mmDMS==0)=NaN;
        mmPAL( deldel)=NaN; mmPAL(mmPAL>=0.97|mmPAL==0)=NaN;
        mmECO( deldel )=NaN; mmECO(mmECO>=0.97|mmECO==0)=NaN;
        mmSOSS( deldel)=NaN; mmSOSS(mmSOSS>=0.97|mmSOSS==0)=NaN;
        mmTAV( deldel )=NaN; mmTAV(mmTAV>=0.97|mmTAV==0)=NaN;
        mmECOl( deldel )=NaN; mmECOl(mmECOl>=0.97|mmECOl==0)=NaN;
        mmECOp( deldel )=NaN; mmECOp(mmECOp>=0.97|mmECOp==0)=NaN;
        mmRT1(deldel) = NaN;
        mmRT2(deldel) = NaN;
        mmRT3(deldel) = NaN;

    else

        mmTP(mmTP>=0.97|mmTP==0)=NaN;
        mmCSRT(mmCSRT>=0.97|mmCSRT==0)=NaN;
        mmCSRTi(mmCSRTi>=0.97|mmCSRTi==0)=NaN;
        mmDMS(mmDMS>=0.97|mmDMS==0)=NaN;
        mmPAL(mmPAL>=0.97|mmPAL==0)=NaN;
        mmECO(mmECO>=0.97|mmECO==0)=NaN;
        mmSOSS(mmSOSS>=0.97|mmSOSS==0)=NaN;
        mmTAV(mmTAV>=0.97|mmTAV==0)=NaN;
        mmECOl(mmECOl>=0.97|mmECOl==0)=NaN;
        mmECOp(mmECOp>=0.97|mmECOp==0)=NaN;

    end

    mmAge = nanmean(mkAge_first,2);

    mmALL= table;
    mmALL.Age = mmAge;
    mmALL.TP = mmTP;
    mmALL.TAV= mmTAV;
    mmALL.CSRT= mmCSRT;
    mmALL.CSRTi= mmCSRTi;
    mmALL.DMS= mmDMS;
    mmALL.ECO= mmECO;
    mmALL.ECOl= mmECOl;
    mmALL.ECOp= mmECOp;
    mmALL.SOSS = mmSOSS;
    mmALL.PAL = mmPAL;
    mmALL.RT1 = mmRT1;
    mmALL.RT2 = mmRT2;
    mmALL.RT3 = mmRT3;

    %     mmALL.ECOl= mmECOl;
    %     mmALL.ECOp= mmECOp;

    % SAVE BIOSIMIA_DATA
    save([ 'bioaging_' num2str(dataset)],'mmALL', 'mmAge', 'mkid','all_attempts')
end
toc

%%
clc
close all
mA= [];
ntrials=0;
for dataset = [1 2 3 4]

    clearvars -except mA dataset ntrials
    load([ 'bioaging_' num2str(dataset)])
    mA = [mA; mmALL];
    ntrials=sum(all_attempts) + ntrials;

end

% REMOVING PAL
% mA.PAL=[];

% ADDING CONTRAST TAV-DMS
mA.DmT = mA.TAV-mA.DMS;

% figure(dataset)
clc
% THE 9 ones
nema=mA.Properties.VariableNames;
mdl=cell(1,size(nema,2));
for i=2:size(nema,2)
    %         if i==1; i=i+1; end
    mini=table;
    eval(['mini= table( mA.Age, mA.' nema{i} ');' ])
    mini = renamevars(mini, {'Var1', 'Var2'}, {'Age', nema{i} });
    mini(mini.Age==0,:)=[];
%     subplot(4,4,i-1)
figure(i)
    %         eval(['mdl = fitglm(mini,''' nema{i} ' ~ Age + Age^2  '' ,''Distribution'',''normal'')'])
    eval(['mdl{i} = stepwiseglm(mini,''' nema{i} ' ~ Age + Age^2  '', ''Criterion'',''aic'');'])
    eval(['mdlm = fitglm(mini,''' nema{i} ' ~ Age   '' ,''Distribution'',''normal'');'])
%     mdl{i}=mdlm;
    disp( [ nema{i} ' models: '])
    [ypred,yci] = predict(mdl{i}, [0:1:25 ]'  );
    eval(['scatter(mini.Age,mini.' nema{i} ',10, ''ok'' );']); lsline; hold on;
    plot(0:25,ypred,'-r');
    plot(0:25,yci(:,1),':r'); plot(0:25,yci(:,2),':r'); axis square
    eval([' N = sum(~isnan(mini.' nema{i} '));' ] )
    title({ [ nema{i} ': R = ' num2str(round(mdl{i}.Rsquared.Adjusted,2)) , ' ; slope = ' num2str(round(mdlm.Coefficients.Estimate(2),3)) ...
        '; AIC = ' num2str(round(mdlm.ModelCriterion.AIC,2)) ] , ['n= ' num2str(N)] })
    xlabel('Age'); ylabel('Performance')
    if strcmp(nema{i}, 'RT1') || strcmp(nema{i}, 'RT2') || strcmp(nema{i}, 'RT3') || strcmp(nema{i}, 'DmT')
        axis auto
    else
        ylim([0:1])

    end

    hold off

    % if i ==4
    %     keyboard
    % end

    %             waitforbuttonpress
end

% mA.DmT = mA.TAV-mA.DMS;

fitglm(mA,'Age ~ DmT')

%
%     % MDL TO DEFINE COGNTION
% mdlraw = fitglm(mA,' CSRT ~ TAV  ')
% resid= mdlraw.Residuals.Raw;
% fitglm(mmAge,resid)
% plot(mmAge,resid,'ok'); lsline

% MDL TO FIND AGING
mA2=mA;
try

    if 0
    mA2.TP =[];
    mA2.PAL =[];
mA2.ECO =[];
mA2.ECOl =[];
mA2.ECOp =[];
mA2.RT1 =[];
mA2.RT2 =[];
mA2.RT3 =[];
    end

catch
    disp('...error')
end
% mdlraw = stepwiseglm(mA, 'Age ~ TP  + CSRT + CSRTi + TAV + DMS + SOSS + ECO + PAL', 'upper', 'interactions', 'Criterion','bic')
% mdlraw = stepwiseglm(mA2, 'Age ~  CSRT + CSRTi + TAV + DMS + SOSS + PAL', 'upper', 'linear', 'Criterion','aic')
% mdlraw = stepwiseglm(mA2, 'Age ~ 1', 'upper', 'linear', 'Criterion','bic')
mdlraw = stepwiseglm(mA2, 'Age ~ CSRTi+ CSRTi^2 + DMS + DMS^2 + CSRT + CSRT^2', 'upper', 'linear', 'Criterion','Deviance')

nullmdl=fitglm(mA2, 'Age ~ 1' );
deviance = mdlraw.Deviance;
null_deviance = nullmdl.Deviance;
% variance_explained = 1 - (deviance / null_deviance);

rsquared = mdlraw.Rsquared.Ordinary;

% fitglm(mA2, 'Age ~ CSRT + CSRT^2')
% mdlraw = stepwiseglm(mmALL, 'Age ~ 1', 'upper', 'linear', 'Criterion','bic')

figure
qqplot(mdlraw.Residuals.Pearson)

mdl1=fitglm(mA2, 'Age ~ CSRTi ' ); rsquared1 = mdl1.Rsquared.Ordinary
mdl2=fitglm(mA2, 'Age ~ CSRTi+ DMS + DMS^2' ); rsquared2 = mdl2.Rsquared.Ordinary
mdl3=fitglm(mA2, 'Age ~ CSRTi+ DMS + DMS^2 + CSRT + CSRT^2' ); rsquared3 = mdl3.Rsquared.Ordinary


mdl4a = fitglm(mA2, 'Age ~ CSRTi+ DMS + DMS^2 + CSRT + CSRT^2 + RT3 + RT3^2' ) ;
rsquared4 = mdl4a.Rsquared.Ordinary 

mdl4b = fitglm(mA2, 'Age ~ CSRTi+ DMS + DMS^2 + CSRT + CSRT^2 + SOSS ' ) ;
rsquared4 = mdl4b.Rsquared.Ordinary 

mdl5 = fitglm(mA2, 'Age ~ CSRTi+ DMS + DMS^2 + CSRT + CSRT^2 + SOSS + SOSS^2 + RT3 + RT3^2 ' ) ;
rsquared5 = mdl5.Rsquared.Ordinary 

%%
nspecies = zeros(1,125);
nspecies([2:7 97:114 117:118 123:125]) = 3; % RHESUS
nspecies([9:54]) = 2; % FASCICU
nspecies([57:82 85:87 90:96 119:122]) = 1; % TONK

if 0

    %%%%%
    aALL= table2array(mA2);
    aALL (:,1)=[];
    [coeff,score,latent,tsquared,explained,mu] = pca(aALL);

    allpca = score;
    mask= isnan(mean(allpca,2));
    allpca(mask,:)=[];
    Age= mA2.Age;
    Age(mask)=[];

    allpca= [Age allpca];
    allpca=array2table(allpca);

    mdlpca = stepwiseglm(allpca, 'allpca1 ~ 1', 'upper', 'linear');

end