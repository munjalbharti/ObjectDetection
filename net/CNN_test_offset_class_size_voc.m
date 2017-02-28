function [] = CNN_test_offset_class_size_voc()
 
        close all;
        clear;
 
        % Do not forget to run Setup first!
        run ..\RGBObjectDetectionSetUp.m
 
        opts.baseDir = 'E:\Bharti\Code\Thesis\data\' ;
 
        opts.expDir=[opts.baseDir filesep 'rgb_object_detection-coco-voc-normalised-L2AdaptiveLoss'];
       
        opts.epoch=51; %124;
        opts.resultDir=[opts.expDir filesep 'ResultsNow' filesep 'VOCvalFinal' filesep sprintf('%d',opts.epoch)];
        opts.modelPath = [opts.expDir filesep sprintf('net-epoch-%d.mat',opts.epoch)] ;
       % 2008_000075
        opts.gpus=[1];
        method=2;
%method 1
switch(method)
    case 1 
           opts.threshold=20; 
           post_process_pref=sprintf('cc_th_%d',opts.threshold);
           
        
    case 2  
           opts.threshold=5;
           opts.bin_size=[5,5];
           opts.non_m_win_size=9;
           post_process_pref=sprintf('hough_th_%d_bin_%d_win_%d_with_nmx_conf_avg_last',opts.threshold,opts.bin_size(1),opts.non_m_win_size);
        
    case 3
           opts.threshold=5;
           opts.bandwidth=sqrt(36);
           opts.iteration=2;
         %  post_process_pref=sprintf('meanshift_th_%d_bw_%d_itr_%d_with_nmx',opts.threshold,opts.bandwidth,opts.iteration);
           post_process_pref=sprintf('meanshift_th_%d_bw_sqrt_36_itr_%d_with_nmx_conf_avg_time_test',opts.threshold,opts.iteration);
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
 
opts.nclasses=length(opts.classes);
fids={};
%VOCPATH=[opts.baseDir filesep 'VOC2012'];
VOCPATH='E:\Bharti\Code\Thesis\data\VOC2012';

VOCPATH='E:\Bharti\Code\Thesis\voc\VOCtest_06-Nov-2007\VOCdevkit\VOC2007';
opts.annopath=[VOCPATH  filesep 'Annotations'];
opts.imgpath=[VOCPATH filesep 'JPEGImages' ];
opts.imgsetpath= fullfile(VOCPATH, 'ImageSets', 'Main' , 'test.txt');
 

   
 net=get_test_net(opts);     
 predVar1 = net.getVarIndex('prediction1') ;
 predVar2 = net.getVarIndex('prob') ;
 predVar3 = net.getVarIndex('prediction3') ;
 
 %predVar = net.getVarIndex('prediction') ;
 inputVar = 'data' ;
       
 [ids,~]=textread(opts.imgsetpath,'%s %d');
    %2008_000064.png
 for i=1859:1859
             fprintf('Evaluating image %d\n',i);
          
             name=sprintf('%s.png',ids{i});
             rgb= imread(fullfile(opts.imgpath,sprintf('%s.jpg',ids{i})));
     
             im_ =single(rgb) ;
             
             im_height=size(im_,1);
             im_width=size(im_,2);
             
             if(im_height < im_width)
                  im_=imresize(im_,[256,NaN]);
             else 
                  im_=imresize(im_,[NaN,256]);
             end 
        
             if ~isempty(opts.gpus)
                im_ = gpuArray(im_) ;
             end
   
        
          
 
             %tStart=tic;                    
             net.eval({inputVar, im_}) ;
             %tElapsed = toc(tStart);
             %fprintf('\nEvaluation %f sec for image %d X %d\n#####################\n',tElapsed,size(im_,1),size(im_,2));
           
   
             %prediction= gather(net.vars(predVar).value)  ;   
             %offset=prediction(:,:,[22,23])*256;
             offset= gather(net.vars(predVar1).value)  ;
             offset=offset*256;
            
             bbox_size = gather(net.vars(predVar3).value) ;
             bbox_size= bbox_size*256;
             switch method
                 case 1
                     [centers_y,centers_x,contri]=find_centers_connected_components(offset,opts.threshold);      
                 case 2
                    % [centers_y,centers_x,votes]=find_centers_hough_voting_fast(offset,opts.bin_size,opts.threshold, opts.non_m_win_size); 
                      [centers_y,centers_x,contri]=find_centers_hough_voting1(offset,opts.bin_size,opts.threshold,opts.non_m_win_size);
                 
               %[centers_y,centers_x,widths,heights,contri]= find_centers_4dhough_voting(offset,bbox_size, opts.bin_size,opts.threshold,opts.non_m_win_size);
                 case 3 
                    % tic;
                     [centers_y,centers_x,contri]= find_centers_mean_shift(offset,opts.threshold,opts.bandwidth,opts.iteration);
                    % toc
             end
             
               
              prob = gather(net.vars(predVar2).value) ;
              
           
              bbox_size_w=bbox_size(:,:,1) ;
              bbox_size_h=bbox_size(:,:,2) ;
             
             %  prob= prediction(:,:,[1:21]);
              % bbox_size_w=prediction(:,:,[24]) * 256;
             %  bbox_size_h=prediction(:,:,[25]) * 256;
               
               [prob_pred,class_pred] = max(prob,[],3) ;
     

    %Result=struct('min_x',[],'min_y',[],'max_x',[],'max_y',[],'contri_x',{},'contri_y',{},'avg_prob1',[],'per_class_pixels1',[],'avg_prob2',[],'per_class_pixels2',[],'contri_by_total',[],'classes',[],'confidence',[]);
    
    bboxes=[];
    classes=[];
    confidences=[];
    total_detections=0;
    contri_x={};
    contri_y={};
    c_x=[];
    c_y=[];
    

           for k=1:length(centers_y)
               
                   %one center will correspond to bin_size centers at
                   %higher resolution
                   
          %centers_x = all_centers_x(:,(bin_size(2)/2)+1);              
                    
                  ind=sub2ind([size(offset,1),size(offset,2)],contri(k).y_pos,contri(k).x_pos);
               %  ind = find(votes(:,1)==centers_x(k) & votes(:,2)==centers_y(k));
              %   [y_pos,x_pos] = ind2sub([size(offset,1),size(offset,2)],ind);
                % ind=sub2ind([size(offset,1),size(offset,2)],contri(k).y_pos,contri(k).x_pos);
                 
                  width= round(median(bbox_size_w(ind)));
                  height= round(median(bbox_size_h(ind)));
                
                %  width= widths(k)*256;
                %  height= heights(k)*256;
                  
                  if(width <= 0)
                       continue;
                  end
                      
                  if(height <= 0)
                       continue;
                  end
                    
                  min_cx=centers_x(k)-floor(width/2)+1;
                  min_cy =centers_y(k)-floor(height/2)+1;
                    
                  if(min_cx < 1)
                        min_cx = 1;
                  end
                      
                  if(min_cy < 1)
                        min_cy = 1;
                  end
                  
                  max_cx =  min_cx + width - 1;
                  max_cy =  min_cy + height - 1;
               
                  if(max_cx > size(offset,2))
                       max_cx = size(offset,2);
                  end
                      
                 
                     
                   if(max_cy > size(offset,1))
                         max_cy = size(offset,1);
                   end
                
                    total_contri_pixels=length(ind);
        
                 %{
                   classes_box=class_pred([min_cy:max_cy], [min_cx:max_cx]);                
                   total_pixels= length(classes_box(:));                  
                   classes_count2 = accumarray(class_pred(ind),1);                 
                   per_class_pixels2= (classes_count2 /total_contri_pixels) ;
                   %for background
                   classes_count2(1)=-inf ;              
                   [count2,class2]= max(classes_count2);
                %}
                    %%{
                avgs=[];
                avgs(1)=0;
                for class_in=2:21
                    ind1= sub2ind([size(offset,1),size(offset,2),class_in],contri(k).y_pos,contri(k).x_pos,zeros(size(contri(k).x_pos,1),1)+class_in);
                    avgs(class_in)=sum(prob(ind1))/total_contri_pixels;
                end 
       
                [count2,class2]= max(avgs);
                
                if(count2 > 0.3)
                   %%}
               % if(class2 ~=1 ) 
                      %Add this detection for results
                      %avg_prob2= sum(prob_pred(ind(find(class_pred(ind) == class2))))/count2; 
                      total_detections=total_detections+1;              
                      classes(total_detections)=class2;
                      %confidences(total_detections)=per_class_pixels2(class2)* total_contri_pixels/total_pixels;
                      confidences(total_detections)= count2;                
                      c_x(total_detections)= centers_x(k);
                      c_y(total_detections)= centers_y(k);
                      
                      contri_x{total_detections} = contri(k).x_pos;
                      contri_y{total_detections} = contri(k).y_pos;  
                      bboxes(:,total_detections)= [min_cx;min_cy;max_cx;max_cy];
 
          
                end
               
              
           
           end
               
          
           % Display and save results
           %%{
             f=figure ;
            
             subplot(2,2,1);
             imshow(uint8(im_));
             impixelinfo;
             hold on ;
             title('Ground');
             
             
              orig_class=ones(size(im_,1),size(im_,2));
              recs=PASreadrecord(fullfile(opts.annopath,sprintf('%s.xml',ids{i})));
             
             for j=1:size(recs.objects(:),1)
                  bbgt=cat(1,recs.objects(j).bbox)';
                  cls=recs.objects(j).class ;
                  clsind = strmatch(cls,opts.classes,'exact');
                  bbgt(1)= round(bbgt(1)*size(im_,2)/im_width);
                  bbgt(3)= round(bbgt(3)*size(im_,2)/im_width);
                  bbgt(2)= round(bbgt(2)*size(im_,1)/im_height);
                  bbgt(4)= round(bbgt(4)*size(im_,1)/im_height);
 
                  rectangle('Position',[bbgt(1) bbgt(2) (bbgt(3)-bbgt(1)+1-2) (bbgt(4)-bbgt(2)+1-2)],'EdgeColor','b','LineWidth',2);
                  orig_class([bbgt(2):bbgt(4)],[bbgt(1):bbgt(3)])=clsind+1;
                 
             end 
             hold off ;
             
             
             
             subplot(2,2,2);
             image(uint8(orig_class-1)) ;
            
             
             subplot(2,2,3);
             imshow(uint8(im_));
                      
             hold on ;          
             quiver([1:size(offset,2)],[1:size(offset,1)],offset(:,:,1),offset(:,:,2));
            %%}
             for k=1:opts.nclasses
                    class_name= opts.classes{k};
                    det_result_file2=sprintf('%s_det_val_%s.txt','comp3',class_name);
                    fid=fopen(fullfile(opts.resultDir,post_process_pref,det_result_file2),'at');
                    fids{k}=fid;
        
             end 

           [bboxes_filter, classes_filter, confidences_filter,contri_x_filter,contri_y_filter,centers_x_filter,centers_y_filter]=   mergeCenters(bboxes, classes, confidences,contri_x,contri_y,c_x,c_y);
              %[bboxes_filter, classes_filter, confidences_filter,contri_x_filter,contri_y_filter, centers_x_filter ,centers_y_filter]=   mergeCenters(bboxes, classes, confidences,contri_x,contri_y,centers_x,centers_y);
          
            det_count=numel(classes_filter);
            if(total_detections ~= det_count)
                fprintf('Image %s has overlapping detections',name);
            end 
              for k=1:det_count
                    class_ass=classes_filter(k);
                    fid2=fids{class_ass-1};
                    
                    r_min_x = bboxes_filter(1,k) ;    
                    r_min_y = bboxes_filter(2,k) ;
                    r_max_x = bboxes_filter(3,k) ;
                    r_max_y = bboxes_filter(4,k) ;
                    confidence = confidences_filter(k);
                    
                    
                    %r_min_x= round(r_min_x*im_width/size(im_,2));
                    %r_max_x= round(r_max_x*im_width/size(im_,2));
                    %r_min_y= round(r_min_y*im_height/size(im_,1));
                    %r_max_y= round(r_max_y*im_height/size(im_,1));
                    
                    fprintf(fid2,'%s %f %f %f %f %f \n',name(1:end-4),r_min_x,r_min_y,r_max_x,r_max_y,confidence);
                    %%{
                    plot(contri_x_filter{k},contri_y_filter{k},'g.'); 
                    plot(centers_x_filter(k),centers_y_filter(k),'r+','MarkerSize',12);
                    rectangle('Position',[r_min_x r_min_y (r_max_x-r_min_x+1) (r_max_y-r_min_y+1)],'EdgeColor','b','LineWidth',4);
                    %%}
              
              end 
    
               for k=1:opts.nclasses
                      fid=fids{k};
                      fclose(fid);
        
               end 
             
      
              %%{
    
             title('Prediction');
             hold off ;
       
 
    
             subplot(2,2,4);
             image(uint8(class_pred-1)) ;
             
             colormap(labelColors(21)) ;
 
             savefig(f,fullfile(opts.resultDir,post_process_pref, sprintf('%s.fig',name(1:end-4))));
             saveas(f,fullfile(opts.resultDir,post_process_pref, name));
             close(f);
            % %}
          
       end
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
 
       
        net.addLayer('Softmax',dagnn.SoftMax(),{'prediction2'}, {'prob'});
        net.mode = 'test' ;
end 

