function [] = CNN_test_offset_class_size1()

        close all;
        clear;

        % Do not forget to run Setup first!
        run ..\RGBObjectDetectionSetUp.m

        opts.baseDir = 'E:\Bharti\Code\Thesis\data\' ;

        opts.expDir=[opts.baseDir filesep 'rgb_object_detection-offset-class-size-L-0.0001-normalised'];
       
        opts.epoch=205; %124;
        opts.resultDir=[opts.expDir filesep 'Results1' filesep 'val' filesep sprintf('%d',opts.epoch)];
        opts.modelPath = [opts.expDir filesep 'net-epoch-133.mat'] ;
        opts.imdbPath = fullfile([opts.baseDir filesep  'imdbVOC2012_segmentation_offsets_ssc256.mat']) ;
       
        opts.gpus=[1];
        opts.results_file=fullfile(opts.resultDir,'results.mat');
        opts.segment_class_name='all';
        
        method=2;

%method 1
switch(method)
    case 1 
           opts.threshold=20; 
           post_process_pref=sprintf('cc_th_%d',opts.threshold);
           
    case 2  
           opts.threshold=250;
           opts.bin_size=[16,16];
           opts.non_m_win_size=7;
           post_process_pref=sprintf('hough_th_%d_bin_%d_win_%d',opts.threshold,opts.bin_size(1),opts.non_m_win_size);
        
          % post_process_pref=sprintf('hough_th_%d_bin_%d',opts.threshold,opts.bin_size(1));
        
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

opts.nclasses=length(opts.classes);
fids={};


for k=1:opts.nclasses
        class_name= opts.classes{k};
        det_result_file2=sprintf('%s_det_val_%s.txt','comp3',class_name);
        fid=fopen(fullfile(opts.resultDir,post_process_pref,det_result_file2),'w');
        fids{k}=fid;
        
end 


        imdb = IMDB.load(opts.imdbPath);
        val = find(imdb.set == 2) ;
  
        net=get_test_net(opts);
       
        %predVar1 = net.getVarIndex('prediction') ;
        predVar1 = net.getVarIndex('prediction1') ;
        predVar2 = net.getVarIndex('prob') ;
        predVar3 = net.getVarIndex('prediction3') ;
        
 
        inputVar = 'data' ;
       
        
       for i=1:numel(val)
          j= val(i);
        %  j=1353;
          name=sprintf('img_%d.png',i);
   
          rgb=imdb.images{j};
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
             
        
    


          tStart=tic;
       
          net.eval({inputVar, im_}) ;
          offset= gather(net.vars(predVar1).value)  ;
          
          offset=offset*256;
          
        
   
         % prediction= gather(net.vars(predVar1).value)  ;   
         % offset=prediction(:,:,[22,23])*256;
  

            
              switch method
              case 1
       
                 [centers_y,centers_x,contri]=find_centers_connected_components(offset,opts.threshold);      
              case 2
       
                 [centers_y,centers_x,contri]=find_centers_hough_voting1(offset,opts.bin_size,opts.threshold,opts.non_m_win_size);           
              case 3 

                  [centers_y,centers_x,contri]= find_centers_mean_shift(offset,opts.threshold,opts.bandwidth,opts.iteration);
              end
             
               
          tElapsed = toc(tStart);
          fprintf('Time elapsed %f sec for image %d\n',tElapsed,i);
               
          %prob= prediction(:,:,[1:21]);
            prob = gather(net.vars(predVar2).value) ;
            bbox_size = gather(net.vars(predVar3).value) ;
           
          bbox_size_w=bbox_size(:,:,1) * 256;
          bbox_size_h=bbox_size(:,:,2) * 256;

          [prob_pred,class_pred] = max(prob,[],3) ;
           

          orig_offset=imdb.offsets{j};
          orig_class=imdb.labels{j};
               
             f=figure ;
            
             subplot(2,2,1);
             imshow(uint8(im_));
             impixelinfo;
             hold on ;
             title('Ground');
             
             quiver([1:size(orig_offset,2)],[1:size(orig_offset,1)],orig_offset(:,:,1),orig_offset(:,:,2));
             
           
            % for k=1:length(orig_y)
            %        rectangle('Position',[orig_x(k) orig_y(k) orig_widths(k) orig_heights(k)],'EdgeColor','b','LineWidth',1);
            % end
            
                
              obj=imdb.objects{1,j};

              no_of_gt_detections= size(obj.bbox,2);
               for k=1:no_of_gt_detections
                  bbgt=cat(1,obj.bbox(k))';
                  rectangle('Position',[bbgt.xmin bbgt.ymin (bbgt.xmax-bbgt.xmin+1) (bbgt.ymax-bbgt.ymin+1)],'EdgeColor','b','LineWidth',1);
               end 
             hold off ;
       
             subplot(2,2,2);
             image(uint8(orig_class-1)) ;
             
             subplot(2,2,3);
             imshow(uint8(im_));
                      
             hold on ;          
             quiver([1:size(offset,2)],[1:size(offset,1)],offset(:,:,1),offset(:,:,2));
            
          
               min_x=[];
               min_y=[];
               widths=[];
               heights=[];
             
               %find min_x,min_y and use make_offset_fig
               
          
           
               for k=1:length(centers_y)
                   
                    ind=sub2ind([size(offset,1),size(offset,2)],contri(k).y_pos,contri(k).x_pos);
                
                    bb_size_w_c= bbox_size_w(ind);
                    width=(median(bb_size_w_c(:)));
                    
                    bb_size_h_c= bbox_size_h(ind);
                    height=(median(bb_size_h_c(:)));
                    
                    min_cx=centers_x(k)-floor(width/2)+1;
                    min_cy =centers_y(k)-floor(height/2)+1;
                    
                    
                     max_cx =  min_cx + width - 1;
                     max_cy =  min_cy + height - 1;
               
                      if(max_cx > size(offset,2))
                           max_cx=size(offset,2);
                      end
                      
                      if(min_cx < 1)
                          min_cx=1;
                      end
                      
                       if(min_cy < 1)
                          min_cy = 1;
                      end
                     
                      if(max_cy > size(offset,1))
                           max_cy=size(offset,1);
                      end
                
                     
                      min_x = [min_x; min_cx];
                      min_y = [min_y;min_cy];
                
              
                     
                
                     probs_box=prob_pred([min_cy:max_cy], [min_cx:max_cx]);
                     classes_box=class_pred([min_cy:max_cy], [min_cx:max_cx]);
                
                
                
                  
                    widths=[widths;width];
                    heights=[heights;height];
                 
                     classes_count1 = accumarray(classes_box(:),1);
                     total_pixels= length(classes_box(:));
                     per_class_pixels1= (classes_count1 /total_pixels) ;
                    %for background
                     classes_count1(1)=-inf ;
               
                     [count1,class1]= max(classes_count1);
                      %class will be 1 if there is only background
                
                      classes_count2 = accumarray(class_pred(ind),1);
                      total_contri_pixels=length(ind);
                      per_class_pixels2= (classes_count2 /total_contri_pixels) ;
                     %for background
                     classes_count2(1)=-inf ;
               
                    [count2,class2]= max(classes_count2);
                
                
                %    if(class1 ~=1 )                    
                %      avg_prob1=sum(probs_box(classes_box == class1))/count1;
                %      avg_prob2= sum(prob_pred(ind(find(class_pred(ind) == class2))))/count2;
                    
                %      fprintf(fid1,'%s %f %f %f %f %f %f %f %f %f\n',name(1:end-4),min_cx,min_cy,max_cx,max_cy,avg_prob1,per_class_pixels1(class1),avg_prob2,per_class_pixels2(class2),total_contri_pixels/total_pixels);
              
               % end
                
                if(class2 ~=1 ) 
                      avg_prob1=sum(probs_box(classes_box == class1))/count1;                    
                      avg_prob2= sum(prob_pred(ind(find(class_pred(ind) == class2))))/count2;
                      fid2=fids{class2-1};
                      fprintf(fid2,'%s %f %f %f %f %f %f %f %f %f\n',name(1:end-4),min_cx,min_cy,max_cx,max_cy,avg_prob1,per_class_pixels1(class1),avg_prob2,per_class_pixels2(class2),total_contri_pixels/total_pixels);
              
                end
               
                plot(contri(k).x_pos,contri(k).y_pos,'g.'); 
              
                hold on ;
           
    end
    for k=1:length(centers_y)
         plot(centers_x(k),centers_y(k),'r+','MarkerSize', 12); 
         rectangle('Position',[min_x(k) min_y(k) widths(k) heights(k)],'EdgeColor','b','LineWidth',1);
    end 
       title('Prediction');
       hold off ;
       
  
             
             
             
           
            %[prob_pred,class_pred] = max(prob,[],3) ;
    
             subplot(2,2,4);
             image(uint8(class_pred-1)) ;
             
             colormap(labelColors(21)) ;

             savefig(f,fullfile(opts.resultDir,post_process_pref, sprintf('%s.fig',name(1:end-4))));
             saveas(f,fullfile(opts.resultDir,post_process_pref, name));
             close(f);
          
       end
           
     % save(opts.results_file,'images','offsets','class_probs','orig_offsets','orig_classes','filenames');
     
     % fclose(fid1);
     
    for k=1:opts.nclasses
        fid=fids{k};
        fclose(fid);
        
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