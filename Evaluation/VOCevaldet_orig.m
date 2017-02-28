function [ap,npos,nd,total_tp,total_fp] = VOCevaldet_orig(VOCopts,id,cls,draw)

% load test set
tic;
cp=VOCopts.annocachepath;
if exist(cp,'file')
    fprintf('%s: pr: loading ground truth\n',cls);
    load(cp,'gtids','recs');
else
    %[gtids,t]=textread(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s %d');
    [gtids,t]=textread(VOCopts.imgsetpath,'%s %d');
    for i=1:length(gtids)
        % display progress
        if toc>1
            fprintf('%s: pr: load: %d/%d\n',cls,i,length(gtids));
            drawnow;
            tic;
        end 
        % read annotation
        recs(i)=PASreadrecord(fullfile(VOCopts.annopath,sprintf('%s.xml',gtids{i})));
    end
    save(cp,'gtids','recs');
end

fprintf('%s: pr: evaluating detections\n',cls);

% hash image ids
hash=VOChash_init(gtids);
        
% extract ground truth objects

npos=0;
gt(length(gtids))=struct('BB',[],'diff',[],'det',[]);
for i=1:length(gtids)
    % extract objects of class
    clsinds=strmatch(cls,{recs(i).objects(:).class},'exact');
    gt(i).BB=cat(1,recs(i).objects(clsinds).bbox)';
    gt(i).diff=[recs(i).objects(clsinds).difficult];
    gt(i).det=false(length(clsinds),1);
    npos=npos+sum(~gt(i).diff);
end

result_file=sprintf('%s_det_val_%s.txt','comp3',cls);
filename=fullfile(VOCopts.resultDir,result_file);

[ids,b1,b2,b3,b4,confidence]=textread(filename,'%s %f %f %f %f %f');
BB=[b1 b2 b3 b4]';


%Rename to make it 11 length long
%for m=1:length(ids)
%    namme=ids{m};
%    itn_v=namme(5:end);
%    namme=strcat('img_',sprintf('%07d',str2double(itn_v)));
%    ids{m}=namme;
%end 

% sort detections by decreasing confidence
[sc,si]=sort(-confidence);
ids=ids(si);
BB=BB(:,si);
confidence=confidence(si);

% load results
%[ids,confidence,b1,b2,b3,b4]=textread(sprintf(VOCopts.detrespath,id,cls),'%s %f %f %f %f %f');
%BB=[b1 b2 b3 b4]';

% sort detections by decreasing confidence
%[sc,si]=sort(-confidence);
%ids=ids(si);
%BB=BB(:,si);

% assign detections to ground truth objects
nd=length(confidence);
tp=zeros(nd,1);
fp=zeros(nd,1);
tic;
for d=1:nd
    % display progress
    if toc>1
        fprintf('%s: pr: compute: %d/%d\n',cls,d,nd);
        drawnow;
        tic;
    end
    
    % find ground truth image
    i=VOChash_lookup(hash,ids{d});
   %  i=strmatch(ids{d},gtids,'exact');
    if isempty(i)
        error('unrecognized image "%s"',ids{d});
    elseif length(i)>1
        error('multiple image "%s"',ids{d});
    end

    % assign detection to ground truth object if any
    %%{ 
        I=imread(fullfile(VOCopts.imgpath,sprintf('%s.jpg',gtids{i})));
  
        im_width=size(I,2);
        im_height=size(I,1);

        if(im_height < im_width)
             im_=imresize(I,[256,NaN]);
         else 
             im_=imresize(I,[NaN,256]);
        end 
   %%} 
    % assign detection to ground truth object if any
     bb=BB(:,d);
    
     bb(1)= round(bb(1) * im_width/size(im_,2));
     bb(3)= round(bb(3) * im_width/ size(im_,2));
     bb(2)= round(bb(2) * im_height/size(im_,1));
     bb(4)= round(bb(4) * im_height/size(im_,1));
                  
    
    
    
    ovmax=-inf;
    for j=1:size(gt(i).BB,2)
        bbgt=gt(i).BB(:,j);
        bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
        iw=bi(3)-bi(1)+1;
        ih=bi(4)-bi(2)+1;
        if iw>0 & ih>0                
            % compute overlap as area of intersection / area of union
            ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
               (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
               iw*ih;
            ov=iw*ih/ua;
            if ov>ovmax
                ovmax=ov;
                jmax=j;
            end
        end
    end
    % assign detection as true positive/don't care/false positive
    if ovmax>=VOCopts.minoverlap
        if ~gt(i).diff(jmax)
            if ~gt(i).det(jmax)
                tp(d)=1;            % true positive
		gt(i).det(jmax)=true;
            else
                fp(d)=1;            % false positive (multiple detection)
            end
        end
    else
        fp(d)=1;                    % false positive
    end
end

% compute precision/recall

total_fp=sum(fp);
total_tp=sum(tp);
fprintf('True Detections %d\n',total_tp);
fprintf('False Detections %d\n',total_fp);

fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/npos;
prec=tp./(fp+tp);

ap=VOCap(rec,prec);
fprintf('Average Precision %f\n',ap);

if draw
    % plot precision/recall
    plot(rec,prec,'-');
    grid;
    xlabel 'recall'
    ylabel 'precision'
    title(sprintf('class: %s, subset: %s, AP = %.3f',cls,'val',ap));
end
