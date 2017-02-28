 
close all;
clear;

 % Do not forget to run Setup first!
run ..\RGBObjectDetectionSetUp.m

%1.cc 2. hough 3. meanshift
method=3;

%method 1
switch(method)
    case 1 
           opts.threshold=20; 
           post_process_pref=sprintf('cc_th_%d',opts.threshold);
           
    case 2  
           opts.threshold=500;
           opts.bin_size=[32,32];
           post_process_pref=sprintf('hough_th_%d_bin_%d',opts.threshold,opts.bin_size(1));
        
    case 3
           opts.threshold=20;
           opts.bandwidth=30;
           opts.iteration=5;
           post_process_pref=sprintf('meanshift_th_%d_bw_%d_itr_%d_result_seg',opts.threshold,opts.bandwidth,opts.iteration);
end 

opts.resultDir='E:\Bharti\Code\Thesis\data\rgb_object_detection-offset-all-class-L-0.001-normalised-1\Results1\val\155\';
 
opts.baseDir = 'E:\Bharti\Code\Thesis\data\' ;
opts.data_dir=[opts.baseDir filesep 'VOC2012'];
opts.segment_class=15;
opts.segment_class_name='person';
opts.masks_dir=fullfile(opts.data_dir,'Class_Masks',sprintf('%d_%s',opts.segment_class,opts.segment_class_name));        
opts.image_fixed_size=[256,256];
opts.offset_dir=fullfile(opts.masks_dir,'offset','val');
        
results=load(fullfile(opts.resultDir,'results.mat'));   
images=results.images;
offsets=results.offsets;
class_probs=results.class_probs;
orig_offsets=results.orig_offsets;
orig_classes=results.orig_classes;
filenames= results.filenames;

mkdir(fullfile(opts.resultDir,post_process_pref));
det_result_file1=sprintf('%s_det_val_%s_ag1.txt','comp3',opts.segment_class_name);
fid1=fopen(fullfile(opts.resultDir,post_process_pref,det_result_file1),'w');

det_result_file2=sprintf('%s_det_val_%s_ag2.txt','comp3',opts.segment_class_name);
fid2=fopen(fullfile(opts.resultDir,post_process_pref,det_result_file2),'w');
      
no_of_images=size(offsets,4);

%result_seg=load('F:\Bharti\Thesis\data\rgb_object_detection-15-person-offset-class-1-L-0.0001-adaptiveLoss\Results1\val\4064\results.mat');
%class_probs=result_seg.class_probs;
%filenames1= result_seg.filenames;
 %ind=strmatch(sprintf('%s.png','2011_001632'),cellstr(filenames) , 'exact');

%for k=ind:ind
for k=1:no_of_images
    rgb=images{k};
    name=filenames{k};
   
    im_ =single(rgb) ;
    scores=class_probs{k};
    [prob_pred,class_pred] = max(scores,[],3) ;
    %prob_class=scores_(:,:,2);
    
    offset=offsets(:,:,:,k);
    
   orig_offset= orig_offsets{k} ;
   orig_class= orig_classes{k};
   
   f=figure ;
            
   subplot(2,2,1);
   imshow(uint8(im_));
   impixelinfo;
   hold on ;
   
  % gt=load(fullfile(opts.offset_dir,sprintf('%s.mat',name(1:end-4))));
   obj_rects=find_rects_gt(orig_offset);
  % obj_rects=gt.obj_rects ;
   %orig_offset=gt.offset ;
            
  for k=1:length(obj_rects.x_mins)
     x_min=obj_rects.x_mins(k);
     y_min=obj_rects.y_mins(k);
     width=obj_rects.widths(k);
     height=obj_rects.heights(k);
       
     rectangle('Position',[x_min y_min width height],'LineWidth',1,'EdgeColor','b'); 
    
     center_x  = x_min+floor((width-1)/2);
     center_y =  y_min+floor((height-1)/2);
     plot(center_x,center_y,'r+','MarkerSize', 12);
  
  end 
      
    title('Ground');
             
    quiver([1:size(orig_offset,1)],[1:size(orig_offset,2)],orig_offset(:,:,1),orig_offset(:,:,2));
    hold off ;
       
    subplot(2,2,2);
    image(uint8(orig_class-1)) ;
             
             
    subplot(2,2,3);
    imshow(uint8(im_));
             
           
   hold on ;          
   quiver([1:size(offset,1)],[1:size(offset,2)],offset(:,:,1),offset(:,:,2));
   %hold on ;
   %quiver([1:size(orig_offset,1)],[1:size(orig_offset,2)],orig_offset(:,:,1),orig_offset(:,:,2),'color',[1 1 0]);
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
                
                probs_box=prob_pred([min_cy:max_cy], [min_cx:max_cx]);
                classes_box=class_pred([min_cy:max_cy], [min_cx:max_cx]);
                
                
                ind=sub2ind([size(offset,1),size(offset,2)],contri(k).y_pos,contri(k).x_pos);
                
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
                
                
                if(class1 ~=1 )                    
                      avg_prob1=sum(probs_box(classes_box == class1))/count1;
                      avg_prob2= sum(prob_pred(ind(find(class_pred(ind) == class2))))/count2;
                    
                      fprintf(fid1,'%s %f %f %f %f %f %f %f %f %f\n',name(1:end-4),min_cx,min_cy,max_cx,max_cy,avg_prob1,per_class_pixels1(class1),avg_prob2,per_class_pixels2(class2),total_contri_pixels/total_pixels);
              
                end
                
                if(class2 ~=1 ) 
                      avg_prob1=sum(probs_box(classes_box == class1))/count1;                    
                      avg_prob2= sum(prob_pred(ind(find(class_pred(ind) == class2))))/count2;
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
       
       subplot(2,2,4);
       image(uint8(class_pred-1)) ;
             
       colormap(labelColors()) ;

       savefig(f,fullfile(opts.resultDir,post_process_pref, sprintf('%s.fig',name(1:end-4))))
       saveas(f,fullfile(opts.resultDir,post_process_pref, name));
       %  save(fullfile(opts.resultDir,post_process_pref, sprintf('%s.mat',name(1:end-4))),'offset','centers_y','centers_x');
            
       close(f);    
      
    
end 

 fclose(fid1);
 fclose(fid2);

   
            
             
             
            