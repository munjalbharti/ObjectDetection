
function [ap,npos,nd,total_tp,total_fp] = VOCevaldet_orig_disp(VOCopts,id,cls,draw,pre)
 
% load test set

mkdir(fullfile(VOCopts.resultDir,pre));
 
tic;
cp=VOCopts.annocachepath;
if exist(cp,'file')
    fprintf('%s: pr: loading ground truth\n',cls);
    load(cp,'gtids','recs');
else
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
 
%[ids,b1,b2,b3,b4,confidence2,perct2, contbytotal]=textread(filename,'%s %f %f %f %f %f %f %f');
[ids,b1,b2,b3,b4,confidence]=textread(filename,'%s %f %f %f %f %f');

BB=[b1 b2 b3 b4]';
%confidence=perct2 .* contbytotal;
%confidence=perct2;
%perct=perct2;
 
%high_perct_ind=find(perct > 0.3);
%ids=ids(high_perct_ind);
%BB=BB(:,high_perct_ind);
%perct=perct(high_perct_ind);
%confidence=confidence(high_perct_ind);
 
% sort detections by decreasing confidence
[sc,si]=sort(-confidence);
ids=ids(si);
BB=BB(:,si);
%perct=perct(si);
confidence=confidence(si);
 
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
    %i=strmatch(ids{d},gtids,'exact');
   
    if isempty(i)
        error('unrecognized image "%s"',ids{d});
    elseif length(i)>1
        error('multiple image "%s"',ids{d});
    end
 
        % read image
    I=imread(fullfile(VOCopts.imgpath,sprintf('%s.jpg',gtids{i})));
    % draw detection bounding box and ground truth bounding box (if any)
    f=figure;
    imshow(I);
    hold on;
 
    im_width=size(I,2);
    im_height=size(I,1);
    
    if(im_height < im_width)
         im_=imresize(I,[256,NaN]);
     else 
         im_=imresize(I,[NaN,256]);
    end 
    %    im_= I;
    
    % assign detection to ground truth object if any
        bb=BB(:,d);
    
         bb(1)= round(bb(1) * im_width/size(im_,2));
         bb(3)= round(bb(3) * im_width/ size(im_,2));
         bb(2)= round(bb(2) * im_height/size(im_,1));
         bb(4)= round(bb(4) * im_height/size(im_,1));
                  
    ovmax=-inf;
    jmax=0;
    
    for j=1:size(gt(i).BB,2)
        bbgt=gt(i).BB(:,j);
       if gt(i).diff(j)
            plot(bbgt([1 3 3 1 1]),bbgt([2 2 4 4 2]),'k','LineStyle' ,':','linewidth',2);       
       else 
            plot(bbgt([1 3 3 1 1]),bbgt([2 2 4 4 2]),'k','linewidth',2);      
       end 
       
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
    
 
       if(jmax ~= 0)
            %this detection was assigned to this ground truth %colored with
            %blue
            bbgt=gt(i).BB(:,jmax);
            plot(bbgt([1 3 3 1 1]),bbgt([2 2 4 4 2]),'b','linewidth',2);
       end 
        
    % assign detection as true positive/don't care/false positive
    if ovmax>=VOCopts.minoverlap
        if ~gt(i).diff(jmax)
            if ~gt(i).det(jmax)
                tp(d)=1;            % true positive
                gt(i).det(jmax)=true;
        
                 plot(bb([1 3 3 1 1]),bb([2 2 4 4 2]),'g','linewidth',2);
                 titl=sprintf('TP:Correct Detection, IOU:%f CONF:%f',ovmax,confidence(d));
                 name= sprintf('%s_%s_%d_tp',ids{d},cls,d);
            else
                fp(d)=1;            % false positive (multiple detection)
                plot(bb([1 3 3 1 1]),bb([2 2 4 4 2]),'r','LineStyle',':','linewidth',2);
                titl=sprintf('FP:Multiple Detection, IOU:%f CONF:%f',ovmax,confidence(d));
                name= sprintf('%s_%s_%d_fpm',ids{d},cls,d);
            
            end
        else 
                plot(bb([1 3 3 1 1]),bb([2 2 4 4 2]),'y','LineStyle',':','linewidth',2);
                titl=sprintf('NA:Difficult Detection, IOU:%f CONF:%f',ovmax,confidence(d));
                name= sprintf('%s_%s_%d_dif',ids{d},cls,d);
            
        end
    else
        fp(d)=1;                    % false positive
     
        plot(bb([1 3 3 1 1]),bb([2 2 4 4 2]),'r','linewidth',2);
        titl=sprintf('FP: False Detection, IOU:%f CONF:%f PER:%f',ovmax,confidence(d));
        name= sprintf('%s_%s_%d_fp',ids{d},cls,d);
    end
    
    title(titl);
    %savefig(f,fullfile(VOCopts.resultDir,pre,sprintf('%s.fig',name)));
    saveas(f,fullfile(VOCopts.resultDir,pre, sprintf('%s.png',name)));
    close(f);
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
 
if draw
    % plot precision/recall
    plot(rec,prec,'-');
    grid;
    xlabel 'recall'
    ylabel 'precision'
    title(sprintf('class: %s, subset: %s, AP = %.3f',cls,'val',ap));
end

