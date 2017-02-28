function [] = CNN_test_offset_mask()

        close all;
        clear;

        % Do not forget to run Setup first!
        run ..\RGBObjectDetectionSetUp.m

        opts.baseDir = 'F:\Bharti\Thesis\data\' ;

        
        opts.expDir=[opts.baseDir filesep 'rgb_object_detection-15-person-test-4794images-offset-new-imdb-L-0.0001-sc-0.01'];
        opts.epoch=81;
        opts.bin_size=[1,1];
        opts.threshold= 20;
        opts.resultDir=[opts.expDir filesep 'Results' filesep 'val_pred_mask' filesep sprintf('%d',opts.epoch) filesep sprintf('bin_%d_thr_%f',opts.bin_size(1),opts.threshold)];
        opts.modelPath = [opts.expDir filesep sprintf('net-epoch-%d.mat',opts.epoch)] ;
        opts.imdbPath = fullfile([opts.baseDir filesep  'rgb_object_detection-15-person-offset-4794.imdb-again-offset-saved.mat']) ;
       
        opts.data_dir=[opts.baseDir filesep 'VOC2012'];
        opts.seg_obj_dir=fullfile(opts.data_dir,'SegmentationObject');
        
        opts.segment_class=15;
        opts.segment_class_name='person';
        opts.masks_dir=fullfile(opts.data_dir,'Class_Masks',sprintf('%d_%s',opts.segment_class,opts.segment_class_name));
        
        opts.image_fixed_size=[256,256];
        opts.gpus=[];

        pred_masks_dir='F:\Bharti\Thesis\data\rgb_object_detection-15-person-class-799-online-aug-L-0.001\Results\val\62\' ;
      
        if not (exist(opts.resultDir,'dir')==7)
                mkdir(opts.resultDir);
        end


        imdb = IMDB.load(opts.imdbPath);
        val = find(imdb.images.set == 2) ;
        opts.offset_dir=fullfile(opts.masks_dir,'offset','val');
    
        
        net=get_test_net(opts);
       
        predVar = net.getVarIndex('prediction') ;
        inputVar = 'data' ;

   
       %
        %valuate each image
        for i=1:numel(val)
        %for i=7:7
          j= val(i);
         % j=4823;
         %j=4839;
         % imId = imdb.images.id(j) ;
          name = imdb.images.filenames{j} ;

          rgb=imdb.images.data(:,:,:,j);
          im_ =single(rgb) ;
          mask =imdb.images.mask(:,:,:,j) ;
          
          

          orig_offset=single(mask);

          %orig_offset(orig_offset == single(0.1))= 0 ;
          
          if ~isempty(opts.gpus)
            im_ = gpuArray(im_) ;
          end

          net.eval({inputVar, im_}) ;
          scores_ = gather(net.vars(predVar).value) ;
      
            % Print vector;
            
            f=figure ;
            
            subplot(1,3,1);
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
             
             pred1=load(fullfile(pred_masks_dir,sprintf('%s.mat',name(1:end-4))));
             pred_mask=pred1.mask;
             
              cmap = labelColors() ;
              subplot(1,3,2) ;
              image(uint8(pred_mask-1)) ;
              axis image ;
              title('predicted') ;

               colormap(cmap) ;
            
            
             subplot(1,3,3);
             imshow(uint8(im_));
             
           
             hold on ;
             offset=scores_;             
             quiver([1:size(offset,1)],[1:size(offset,2)],offset(:,:,1),offset(:,:,2));
             hold on ;
             %quiver([1:size(orig_offset,1)],[1:size(orig_offset,2)],orig_offset(:,:,1),orig_offset(:,:,2),'color',[1 1 0]);
            
            
             %[centers_y,centers_x,contri]=find_object_centers(offset,opts.bin_size,opts.threshold);
             [centers_y,centers_x,contri]=find_object_center_for_masks( offset,opts.bin_size,opts.threshold, pred_mask );
             %find min_x,min_y and use make_offset_fig
             
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
       
             
             savefig(f,fullfile(opts.resultDir, sprintf('%s.fig',name(1:end-4))))
             saveas(f,fullfile(opts.resultDir, name));
           %  save(fullfile(opts.resultDir, sprintf('%s.mat',name(1:end-4))),'offset','centers_y','centers_x');
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