function [] = CNN_test_offset_class1()

        close all;
        clear;

        % Do not forget to run Setup first!
        run ..\RGBObjectDetectionSetUp.m

        opts.baseDir = 'E:\Bharti\Code\Thesis\data\' ;


        opts.expDir=[opts.baseDir filesep 'rgb_object_detection-offset-all-class-L-0.001-normalised-L2Adaptive'];
       
        opts.epoch=486; %124;
        opts.resultDir=[opts.expDir filesep 'ResultsNow' filesep 'val' filesep sprintf('%d',opts.epoch)];
        opts.modelPath = [opts.expDir filesep sprintf('net-epoch-%d.mat',opts.epoch)] ;
       % opts.imdbPath = fullfile([opts.baseDir filesep  'imdbVOC2012_segmentation_offsets_ssc256.mat']) ;
       
        opts.gpus=[];
        
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
           post_process_pref=sprintf('hough_th_%d_bin_%d_win_%d_conf_1',opts.threshold,opts.bin_size(1),opts.non_m_win_size);
        
       case 3
           opts.threshold=10;
           opts.bandwidth=sqrt(10);
           opts.iteration=2;
           post_process_pref=sprintf('meanshift_th_%d_bw_sqrt_10_itr_%d_conf_1',opts.threshold,opts.iteration);
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

       VOCopts.seg.imgsetpath='E:\Bharti\Code\Thesis\data\VOC2012\ImageSets\Segmentation\val.txt';
       VOCopts.annopath='E:\Bharti\Code\Thesis\data\VOC2012\Annotations\';
       VOCopts.imgpath='E:\Bharti\Code\Thesis\data\VOC2012\JPEGImages\';
       VOCopts.seg.clsimgpath='E:\Bharti\Code\Thesis\data\VOC2012\SegmentationClass\';


       opts.nclasses=length(opts.classes);
       fids={};

        %imdb = IMDB.load(opts.imdbPath);
        %val = find(imdb.set == 2) ;
  
        ids=textread(VOCopts.seg.imgsetpath,'%s');
        net=get_test_net(opts);
       
        predVar1 = net.getVarIndex('prediction1') ;
        predVar2 = net.getVarIndex('prob') ;
     
        inputVar = 'data' ;
       
        
       %for i=1:numel(val)
        for i=1:length(ids)
          %j= val(i);
          %name=sprintf('img_%d.png',i);
          %name = imdb.images.filenames{j} ;
          %rgb=imdb.images.data(:,:,:,j);
          
          %rgb=imdb.images{j};
         fprintf('Evaluating image %d\n',i);
         name=sprintf('%s.png',ids{i});
         I_orig = imread([VOCopts.imgpath,sprintf('%s.jpg',ids{i})]);
          
         im_height=size(I_orig,1);
         im_width=size(I_orig,2);
         
         if(size(I_orig,1) < size(I_orig,2))
            im_ =imresize(I_orig,[256,NaN]);
         else 
            im_ =imresize(I_orig,[NaN,256]);
         end
    
        

          im_ =single(im_) ; 
                           
          if ~isempty(opts.gpus)
            im_ = gpuArray(im_) ;
          end
          
          net.eval({inputVar, im_}) ;
          offset= gather(net.vars(predVar1).value)  ;
          
          offset=offset*256;
          
          prob = gather(net.vars(predVar2).value) ;
          [prob_pred,class_pred] = max(prob,[],3) ;
          
          %mask =imdb.images.mask(:,:,:,j) ;
          %orig_offset=mask(:,:,[1,2],:);
          %orig_class=mask(:,:,[3],:);
          
          %orig_offset=imdb.offsets{j};
          %[orig_x,orig_y,orig_widths,orig_heights]=find_gt_boxes(orig_offset);
          %orig_class=imdb.labels{j};
          
          
            
          
          switch method
             case 1 
                [centers_y,centers_x,contri]=find_centers_connected_components(offset,opts.threshold);      
             case 2
                 [centers_y,centers_x,contri]=find_centers_hough_voting(offset,opts.bin_size,opts.threshold,opts.non_m_win_size);           
             case 3 
                  [centers_y,centers_x,contri]= find_centers_mean_shift(offset,opts.threshold,opts.bandwidth,opts.iteration);
           end
             
             
             bboxes=[];
             classes=[];
             confidences=[];
             total_detections=0;
             contri_x={};
             contri_y={};
             
              %find min_x,min_y and use make_offset_fig
             for k=1:length(centers_y)
                     ind=sub2ind([size(offset,1),size(offset,2)],contri(k).y_pos,contri(k).x_pos);
                     max_cx =  max(contri(k).x_pos);
                     %max_cx= round(max_cx * im_width/ size(im_,2));
                         
                     min_cx =  min(contri(k).x_pos);
                     max_cy = max(contri(k).y_pos);
                     min_cy = min(contri(k).y_pos);

                     if(min_cx < 1)
                        min_cx = 1;
                     end
                      
                     if(min_cy < 1)
                        min_cy = 1;
                     end
                    
                     if(max_cx > size(offset,2))
                       max_cx = size(offset,2);
                     end                   
                     
                     if(max_cy > size(offset,1))
                         max_cy = size(offset,1);
                     end 

                     total_contri_pixels=length(ind);
                      %Confidence 1
                      %%{
                       classes_box=class_pred([min_cy:max_cy], [min_cx:max_cx]);                
                       total_pixels= length(classes_box(:));                  
                       classes_count = accumarray(class_pred(ind),1);                 
                       per_class_pixels2= (classes_count /total_contri_pixels) ;
                       %for background
                       classes_count(1)=-inf ;              
                       [count2,class2]= max(classes_count);
                      %%}
                      %Confidence 2
                     %{ 
                      avgs=[];
                      avgs(1)=0;
                      for class_in=2:21
                        ind1= sub2ind([size(offset,1),size(offset,2),class_in],contri(k).y_pos,contri(k).x_pos,zeros(size(contri(k).x_pos,1),1)+class_in);
                        avgs(class_in)=sum(prob(ind1))/total_contri_pixels;
                      end 
       
                     [count2,class2]= max(avgs);
                 
                     if(count2 > 0.3)
                     %}
                     if(class2 ~=1 ) 
                      %Add this detection for results
                      %avg_prob2= sum(prob_pred(ind(find(class_pred(ind) == class2))))/count2; 
                      total_detections=total_detections+1;              
                      classes(total_detections)=class2;
                      confidences(total_detections)=per_class_pixels2(class2)* total_contri_pixels/total_pixels;
                      %confidences(total_detections)= count2;                
                      
                      contri_x{total_detections} = contri(k).x_pos;
                      contri_y{total_detections} = contri(k).y_pos;  
                      bboxes(:,total_detections)= [min_cx;min_cy;max_cx;max_cy];
                     end               
           
               end


             % Display and save results
             f=figure ;
            
             subplot(2,2,1);
             imshow(uint8(im_));
             %impixelinfo;
             hold on ;
             title('Ground');
             
             [orig_class,map]=imread([VOCopts.seg.clsimgpath,sprintf('%s.png',ids{i})]);
             orig_class = imresize(orig_class,[size(im_,1),size(im_,2)],'nearest'); 
             orig_class(orig_class==255)=0;
          
             
              recs=PASreadrecord(fullfile(VOCopts.annopath,sprintf('%s.xml',ids{i})));
             
             for j=1:size(recs.objects(:),1)
                  bbgt=cat(1,recs.objects(j).bbox)';
                  %cls=recs.objects(j).class ;
                  %clsind = strmatch(cls,opts.classes,'exact');
                  bbgt(1)= round(bbgt(1)*size(im_,2)/im_width);
                  bbgt(3)= round(bbgt(3)*size(im_,2)/im_width);
                  bbgt(2)= round(bbgt(2)*size(im_,1)/im_height);
                  bbgt(4)= round(bbgt(4)*size(im_,1)/im_height);
 
                  rectangle('Position',[bbgt(1) bbgt(2) (bbgt(3)-bbgt(1)+1) (bbgt(4)-bbgt(2)+1)],'EdgeColor','b','LineWidth',2);
                 % orig_class([bbgt(2):bbgt(4)],[bbgt(1):bbgt(3)])=clsind+1;
                  %plot(bbgt([1 3 3 1 1]),bbgt([2 2 4 4 2]),'y-','linewidth',2);
             end 
             hold off ;
             
             
             
             subplot(2,2,2);
             imshow(orig_class,map) ;
            
             
             subplot(2,2,3);
             imshow(uint8(im_));
                      
             hold on ;          
             quiver([1:size(offset,2)],[1:size(offset,1)],offset(:,:,1),offset(:,:,2));
             
             for k=1:opts.nclasses
                    class_name= opts.classes{k};
                    det_result_file2=sprintf('%s_det_val_%s.txt','comp3',class_name);
                    fid=fopen(fullfile(opts.resultDir,post_process_pref,det_result_file2),'at');
                    fids{k}=fid;
        
             end 

            [bboxes_filter, classes_filter, confidences_filter,contri_x_filter,contri_y_filter]=   mergeCenters(bboxes, classes, confidences,contri_x,contri_y);
            
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
                    
                    
                    fprintf(fid2,'%s %f %f %f %f %f \n',name(1:end-4),r_min_x,r_min_y,r_max_x,r_max_y,confidence);
                   %fflush(fid2);
                    plot(contri_x_filter{k},contri_y_filter{k},'g.'); 
                    rectangle('Position',[r_min_x r_min_y (r_max_x-r_min_x+1) (r_max_y-r_min_y+1)],'EdgeColor','b','LineWidth',2);
                    
              
              end 
    
               for k=1:opts.nclasses
                      fid=fids{k};
                      fclose(fid);
        
               end 
             
    
             title('Prediction');
             hold off ;
        
             subplot(2,2,4);
             image(uint8(class_pred-1)) ;
             
             colormap(labelColors(21)) ;
 
             savefig(f,fullfile(opts.resultDir,post_process_pref, sprintf('%s.fig',name(1:end-4))));
             saveas(f,fullfile(opts.resultDir,post_process_pref, name));
             close(f);
          
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