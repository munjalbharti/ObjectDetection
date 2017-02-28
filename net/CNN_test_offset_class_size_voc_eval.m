function [] = CNN_test_offset_class_size_voc_eval()
 
        close all;
        clear;
 
        % Do not forget to run Setup first!
        run ..\RGBObjectDetectionSetUp.m
 
        opts.baseDir = 'E:\Bharti\Code\Thesis\data\' ;
 
        opts.expDir=[opts.baseDir filesep 'rgb_object_detection-offset-class-size-trained'];
       
        opts.epoch=205; %124;
        opts.resultDir=[opts.expDir filesep 'Results2' filesep 'val' filesep sprintf('%d',opts.epoch)];
        opts.modelPath = [opts.expDir filesep 'net-det-on-seg_ssc256 -l-removed.mat'] ;
        
        opts.gpus=[1];
       
        
        method=2;
 
%method 1
switch(method)
    case 1 
           opts.threshold=20; 
           post_process_pref=sprintf('cc_th_%d',opts.threshold);
           
    case 2  
           opts.threshold=500;
           opts.bin_size=[16,16];
           opts.non_m_win_size=5;
           post_process_pref=sprintf('hough_th_%d_bin_%d_win_%d',opts.threshold,opts.bin_size(1),opts.non_m_win_size);
        
    case 3
           opts.threshold=20;
           opts.bandwidth=50;
           opts.iteration=2;
           post_process_pref=sprintf('meanshift_th_%d_bw_%d_itr_%d_result_seg',opts.threshold,opts.bandwidth,opts.iteration);
end 
 
      
        if not (exist(fullfile(opts.resultDir,post_process_pref),'dir')==7)
                mkdir(fullfile(opts.resultDir,post_process_pref));
        end
        
     opts.classes={...
    'aeroplane'
    'bicycle'
    'bird'
    'boat'
    'bottle'
    'bus'
    'car'
    'cat'
    'chair'
    'cow'
    'diningtable'
    'dog'
    'horse'
    'motorbike'
    'person'
    'pottedplant'
    'sheep'
    'sofa'
    'train'
    'tvmonitor'};
 opts.minoverlap=0.5;
opts.nclasses=length(opts.classes);
VOCPATH=[opts.baseDir filesep 'VOC2012'];
opts.annopath=[VOCPATH  filesep 'Annotations'];
opts.imgpath=[VOCPATH filesep 'JPEGImages' ];
opts.imgsetpath= fullfile(VOCPATH, 'ImageSets', 'Main' , 'val.txt');
opts.annocachepath=[VOCPATH  filesep 'AnnotationsCache' filesep 'val_anno.mat'];
 
 
   
 net=get_test_net(opts);     
 predVar = net.getVarIndex('prediction') ;
 inputVar = 'data' ;
 
 cp=opts.annocachepath;
if exist(cp,'file')
    fprintf('loading ground truth\n');
    load(cp,'gtids','recs');
else
     [gtids,~]=textread(opts.imgsetpath,'%s %d');
    for i=1:length(gtids)
        recs(i)=PASreadrecord(sprintf(opts.annopath,gtids{i}));
    end
    save(cp,'gtids','recs');
end   

gt_all_classes={};
for k=1:opts.nclasses
   
    gt(length(gtids))=struct('BB',[],'diff',[],'det',[]);    
    for i=1:length(gtids)
        % extract objects of each class   
         class_name=opts.classes{k};
         clsinds=strmatch(class_name,{recs(i).objects(:).class},'exact');
         gt(i).BB=cat(1,recs(i).objects(clsinds).bbox)';
         gt(i).diff=[recs(i).objects(clsinds).difficult];
         gt(i).det=false(length(clsinds),1);        
    end 
   gt_all_classes{k}=gt;
end


 
 
 
    
 for i=1:length(gtids)
             fprintf('Evaluating image %d\n',i);
          
             name=sprintf('%s.png',gtids{i});
             rgb= imread(fullfile(opts.imgpath,sprintf('%s.jpg',gtids{i})));
     
             orig_img =single(rgb) ;
        
             
              im_height=size(orig_img,1);
             im_width=size(orig_img,2);
             
             if(im_height < im_width)
                  im_=imresize(orig_img,[256,NaN]);
             else 
                  im_=imresize(orig_img,[NaN,256]);
             end 
             
             if ~isempty(opts.gpus)
                im_ = gpuArray(im_) ;
             end
   
         
          
 
             tStart=tic;
       
             net.eval({inputVar, im_}) ;
             
             tElapsed = toc(tStart);
             fprintf('Time elapsed %f sec for image %d\n',tElapsed,i);
   
             prediction= gather(net.vars(predVar).value)  ;   
             offset=prediction(:,:,[22,23])*256;
            
             switch method
                 case 1
                     [centers_y,centers_x,votes]=find_centers_connected_components(offset,opts.threshold);      
                 case 2
                     [centers_y,centers_x,votes]=find_centers_hough_voting_fast(offset,opts.bin_size,opts.threshold, opts.non_m_win_size);           
                 case 3 
                     [centers_y,centers_x,votes]= find_centers_mean_shift_fast(offset,opts.threshold,opts.bandwidth,opts.iteration);
             end
             
               
           
             
               prob= prediction(:,:,[1:21]);
               bbox_size_w=prediction(:,:,[24]) * 256;
               bbox_size_h=prediction(:,:,[25]) * 256;
               
               [prob_pred,class_pred] = max(prob,[],3) ;
     
 
    Result=struct('min_x',[],'min_y',[],'max_x',[],'max_y',[],'classes',[],'confidence',[]);
    total_detections=0;
 
           for k=1:length(centers_y)
               
                   %one center will correspond to bin_size centers at
                   %higher resolution
                   
          %centers_x = all_centers_x(:,(bin_size(2)/2)+1);              
                    
      
                 ind = find(votes(:,1)==centers_x(k) & votes(:,2)==centers_y(k));
                 % ind = sub2ind([size(offset,1),size(offset,2)],y_pos,x_pos);
                 [y_pos,x_pos] = ind2sub([size(offset,1),size(offset,2)],ind);
                
              
                  width= round(median(bbox_size_w(ind)));
                  height= round(median(bbox_size_h(ind)));
                    
                  min_cx=centers_x(k)-floor(width/2)+1;
                  min_cy =centers_y(k)-floor(height/2)+1;
                    
                    
                  max_cx =  min_cx + width - 1;
                  max_cy =  min_cy + height - 1;
               
                  if(max_cx > size(offset,2))
                       max_cx = size(offset,2);
                  end
                      
                  if(min_cx < 1)
                        min_cx = 1;
                  end
                      
                  if(min_cy < 1)
                        min_cy = 1;
                  end
                     
                   if(max_cy > size(offset,1))
                         max_cy = size(offset,1);
                   end
                
        
                   classes_box=class_pred([min_cy:max_cy], [min_cx:max_cx]);
          
                   total_pixels= length(classes_box(:));
                   
                   classes_count2 = accumarray(class_pred(ind),1);
                   total_contri_pixels=length(ind);
                   per_class_pixels2= (classes_count2 /total_contri_pixels) ;
                     %for background
                   classes_count2(1)=-inf ;
               
                   [count2,class2]= max(classes_count2);
                
       
                
                if(class2 ~=1 ) 
                      %Add this detection for results
                      total_detections=total_detections+1;
                     
                      Result(1).min_x(total_detections) = min_cx;
                      Result(1).min_y(total_detections) = min_cy;
                      Result(1).max_x(total_detections) = max_cx;
                      Result(1).max_y(total_detections) = max_cy;
                      
                      Result(1).classes(total_detections) = class2;
                      Result(1).confidence(total_detections) = per_class_pixels2(class2) * total_contri_pixels/total_pixels;
                      
                      
                end
               
              
           
           end
               
           
           
             
               f=figure;
               imshow(uint8(orig_img));
               hold on;
             
            
             for j=1:size(recs(i).objects(:),1)
              
                  bbgt=cat(1,recs(i).objects(j).bbox)';
                  diff=recs(i).objects(j).difficult;
                  if(diff)
                    plot(bbgt([1 3 3 1 1]),bbgt([2 2 4 4 2]),'k','LineStyle',':','linewidth',2); 
                  else 
                    plot(bbgt([1 3 3 1 1]),bbgt([2 2 4 4 2]),'k','linewidth',2); %true
                   end 
             end 
             hold on ;
             
             BB=[Result(1).min_x' Result(1).min_y' Result(1).max_x' Result(1).max_y']';
            
 
            % sort detections by decreasing confidence
            [sc,si]=sort(-Result(1).confidence);
            
            BB=BB(:,si);
             
 
              for d=1:total_detections
                    bb=BB(:,d); 
                    %Rescaling to original size image
                    bb(1)= round(bb(1) * im_width/size(im_,2));
                    bb(3)= round(bb(3) * im_width/ size(im_,2));
                    bb(2)= round(bb(2) * im_height/size(im_,1));
                    bb(4)= round(bb(4) * im_height/size(im_,1));
                    
                    
                    class_ass=Result(1).classes(d)-1;
                    gt=gt_all_classes{class_ass};
                   
                    ovmax=-inf;
                    jmax=0;
    
                 for j=1:size(gt(i).BB,2)
                     bbgt=gt(i).BB(:,j);
                     bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
                     iw=bi(3)-bi(1)+1;
                     ih=bi(4)-bi(2)+1;
                     if iw>0 & ih>0                
                                        % compute overlap as area of intersection / area of union
                         ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+(bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-iw*ih;
                         ov=iw*ih/ua;
                         if ov>ovmax
                            ovmax=ov;
                            jmax=j;
                         end
                     end
                 end

    
  
                if ovmax>=opts.minoverlap
                      if ~gt(i).diff(jmax)
                            if ~gt(i).det(jmax)
                                gt(i).det(jmax)=true;
              
                                gt_all_classes{class_ass}=gt;
                                bbgt=gt(i).BB(:,jmax);
                                plot(bbgt([1 3 3 1 1]),bbgt([2 2 4 4 2]),'b','linewidth',2); %true
                                 plot(bb([1 3 3 1 1]),bb([2 2 4 4 2]),'g','linewidth',2);
                            else
                                 plot(bb([1 3 3 1 1]),bb([2 2 4 4 2]),'r','LineStyle',':','linewidth',2); %multiple detections
                            end
                         else
                             plot(bb([1 3 3 1 1]),bb([2 2 4 4 2]),'y','LineStyle',':','linewidth',2); %difficult detection
                      end 
                else
                      plot(bb([1 3 3 1 1]),bb([2 2 4 4 2]),'r','linewidth',2); %false positive
                end    

                    
              
              end 
    
       
             
      
    
    
            
            title(sprintf('image: "%s" det %d (green=true,red=false,yellow=difficult black/blue=gt)',gtids{i},total_detections),'interpreter','none');   
               
           %  savefig(f,fullfile(opts.resultDir,post_process_pref, sprintf('%s.fig',name(1:end-4))));
             saveas(f,fullfile(opts.resultDir,post_process_pref, name));
             close(f);
          
       end
           
     % save(opts.results_file,'images','offsets','class_probs','orig_offsets','orig_classes','filenames');
     
     % fclose(fid1);
     
   
    
        
end 
 
function[net]= get_test_net(opts)
        orig_net=load(opts.modelPath);
        net = dagnn.DagNN.loadobj(orig_net.net) ;
 
        if ~isempty(opts.gpus)
          gpuDevice(opts.gpus(1)) ;
          net.move('gpu') ;
        end 
 
        
       % for name = {'loss'}
       %       net.removeLayer(name) ;
       % end
        %    net.meta.normalization.averageImage = reshape(net.meta.normalization.rgbMean,1,1,3) ;
 
       
      %  net.addLayer('Softmax',dagnn.SoftMax(),{'prediction2'}, {'prob'});
        net.mode = 'test' ;
end 
 

