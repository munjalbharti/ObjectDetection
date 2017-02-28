function [] = CNN_test_offset()

        close all;
        clear;

        % Do not forget to run Setup first!
        run ..\RGBObjectDetectionSetUp.m
        
        method=2;

%method 1
switch(method)
    case 1 
           opts.threshold=20; 
           post_process_pref=sprintf('cc_th_%d',opts.threshold);
           
    case 2  
           opts.threshold=500;
           opts.bin_size=[16,16];
           post_process_pref=sprintf('hough_th_%d_bin_%d',opts.threshold,opts.bin_size(1));
        
    case 3
           opts.threshold=20;
           opts.bandwidth=30;
           opts.iteration=5;
           post_process_pref=sprintf('meanshift_th_%d_bw_%d_itr_%d_result_seg',opts.threshold,opts.bandwidth,opts.iteration);
end 


        opts.baseDir = 'F:\Bharti\Thesis\data\' ;

        
        opts.expDir=[opts.baseDir filesep 'rgb_object_detection-offset-all-class-L-0.001-unnormalised-4'];
        opts.epoch=84;
        
        opts.resultDir=[opts.expDir filesep 'Results'  filesep sprintf('%d',opts.epoch)];
        opts.modelPath = [opts.expDir filesep sprintf('net-epoch-%d.mat',opts.epoch)] ;
        opts.imdbPath = fullfile([opts.baseDir filesep  'rgb_object_detection-15-person-offset-class-4794-bkg-NaN-imdb.mat']) ;
       
        opts.data_dir=[opts.baseDir filesep 'VOC2012'];
        opts.seg_obj_dir=fullfile(opts.data_dir,'SegmentationObject');
        
        opts.segment_class=15;
        opts.segment_class_name='person';
        opts.masks_dir=fullfile(opts.data_dir,'Class_Masks',sprintf('%d_%s',opts.segment_class,opts.segment_class_name));
        
        opts.image_fixed_size=[256,256];
        opts.gpus=[];

      
        %if not (exist(opts.resultDir,'dir')==7)
                mkdir(fullfile(opts.resultDir,post_process_pref));
        %end


        imdb = IMDB.load(opts.imdbPath);
        val = find(imdb.images.set == 1) ;
        opts.offset_dir=fullfile(opts.masks_dir,'offset','train');
    
        
        net=get_test_net(opts);
       
        predVar = net.getVarIndex('prediction') ;
        inputVar = 'data' ;

  
       %
        %valuate each image
        for i=1:numel(val)
          j= val(i);
         % j=4823;
         %j=4839;
         % imId = imdb.images.id(j) ;
          name = imdb.images.filenames{j} ;
       
          rgb=imdb.images.data(:,:,:,j);
          im_ =single(rgb) ;
          mask =imdb.images.mask(:,:,:,j) ;

          
          mask=single(mask);
          orig_offset=mask(:,:,[1,2],:);
          %orig_class=mask(:,:,[3],:);

          %to display zero offsets in background
          orig_offset(orig_offset == single(0.1))= 0 ;
          
          if ~isempty(opts.gpus)
            im_ = gpuArray(im_) ;
          end

          net.eval({inputVar, im_}) ;
          scores_ = gather(net.vars(predVar).value) ;
      
            % Print vector;
            
            f=figure ;
            
            subplot(1,2,1);
            imshow(uint8(im_));
            impixelinfo;
            
            hold on ;
   
            gt=load(fullfile(opts.offset_dir,sprintf('%s.mat',name(1:end-4))));
            obj_rects=gt.obj_rects ;
            %orig_offset1=gt.offset ;
            
            for k=1:length(obj_rects.x_mins)
                x_min=obj_rects.x_mins(k);
                y_min=obj_rects.y_mins(k);
                width=obj_rects.widths(k);
                height=obj_rects.heights(k);
       
                rectangle('Position',[x_min-1 y_min-1 width+1 height+1],'LineWidth',1,'EdgeColor','b'); 
    
                center_x  = x_min+floor((width-1)/2);
                center_y =  y_min+floor((height-1)/2);
                plot(center_x,center_y,'r+','MarkerSize', 12);
  
            end 
      
            title('Ground');
             
             quiver([1:size(orig_offset,1)],[1:size(orig_offset,2)],orig_offset(:,:,1),orig_offset(:,:,2));
             hold off ;
            
             subplot(1,2,2);
             imshow(uint8(im_));
             
           
             hold on ;
             offset=scores_;             
             quiver([1:size(offset,1)],[1:size(offset,2)],offset(:,:,1),offset(:,:,2));
             hold on ;
             %quiver([1:size(orig_offset,1)],[1:size(orig_offset,2)],orig_offset(:,:,1),orig_offset(:,:,2),'color',[1 1 0]);
            
            
             %[centers_y,centers_x,contri]=find_centers_connected_components(offset,[1,1],20);
            % [centers_y,centers_x,contri]=find_centers_hough_voting(uint8(im_),offset,opts.bin_size,opts.threshold);
            %  [centers_y,centers_x,contri]= find_centers_mean_shift( offset,opts.threshold,30,15 )   ;
            
               switch method
                 case 1
                   [centers_y,centers_x,contri]=find_centers_connected_components(offset,opts.threshold);      
                 case 2
                   [centers_y,centers_x,contri]=find_centers_hough_voting(offset,opts.bin_size,opts.threshold);           
                 case 3
                 [centers_y,centers_x,contri]= find_centers_mean_shift(offset,opts.threshold,opts.bandwidth,opts.iteration);
               end
               
             min_x=[];
             min_y=[];
             widths=[];
             heights=[];
             
             %find min_x,min_y and use make_offset_fig
             for k=1:length(centers_y)
                max_cx =  max(contri(k).x_pos);
                min_cx =  min(contri(k).x_pos);
               
                min_x = [min_x; min_cx];
                
                max_cy = max(contri(k).y_pos);
                min_cy = min(contri(k).y_pos);
                
                min_y = [min_y;min_cy];
                
              
                widths=[widths;(max_cx-min_cx)];
                heights=[heights;(max_cy-min_cy)];
                
                
                
                plot(contri(k).x_pos,contri(k).y_pos,'g.'); 
              
                hold on ;
           
             end
             for k=1:length(centers_y)
                plot(centers_x(k),centers_y(k),'r+','MarkerSize', 12); 
                rectangle('Position',[min_x(k) min_y(k) widths(k) heights(k)],'EdgeColor','b','LineWidth',1);
             end 
             
             
             title('Prediction');
             hold off ;
       
             
             savefig(f,fullfile(opts.resultDir,post_process_pref, sprintf('%s.fig',name(1:end-4))))
             saveas(f,fullfile(opts.resultDir,post_process_pref, name));
             save(fullfile(opts.resultDir,post_process_pref, sprintf('%s.mat',name(1:end-4))),'offset','centers_y','centers_x');
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

        for name = {'loss'}
              net.removeLayer(name) ;
        end
        %    net.meta.normalization.averageImage = reshape(net.meta.normalization.rgbMean,1,1,3) ;

        net.mode = 'test' ;
end 