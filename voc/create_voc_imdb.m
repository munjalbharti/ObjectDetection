function create_coco_voc_offset_imdb(  )
 
% This script will create IMDB with train:validation ratio as already done
% in directory mask/train mask/val 
%offsets are generated online
clear;
close all ;


%VOCopts.trainset='train';
VOCopts.annopath='E:\Bharti\Code\Thesis\data\VOC2012\Annotations\';
VOCopts.imgpath='E:\Bharti\Code\Thesis\data\VOC2012\JPEGImages\';

seg_class_dir='E:\Bharti\Code\Thesis\data\VOC2012\SegmentationClass\';
seg_obj_dir='E:\Bharti\Code\Thesis\data\VOC2012\SegmentationObject\';

base_dir='C:\Users\Bharti\Thesis\data';
opts.target_file = [base_dir filesep 'rgb_object_detection-voc-7-12-imdb.mat'];
count=1;


for set_no=1:2     
        if(set_no==1)
            VOCopts.seg.imgsetpath='E:\Bharti\Code\Thesis\data\VOC2012\ImageSets\Segmentation\train.txt'; 
        else
           VOCopts.seg.imgsetpath='E:\Bharti\Code\Thesis\data\VOC2012\ImageSets\Segmentation\val.txt';
        end  
      
    gtids=textread(VOCopts.seg.imgsetpath,'%s');
    
    
    for i=1:length(gtids)
         count=count+1;
   
         I_orig = imread([VOCopts.imgpath,sprintf('%s.jpg',gtids{i})]);
         if(size(I_orig,1) < size(I_orig,2))
            I=imresize(I_orig,[256,NaN]);
         else 
            I=imresize(I_orig,[NaN,256]);
         end
    
        img_height=size(I,1);
        img_width=size(I,2);

      
        offset_mask=zeros(img_height,img_width,2);
        size_mask=zeros(img_height,img_width,2);
      

        [seg_mask,~]=imread(fullfile(seg_class_dir,sprintf('%s.png',gtids{i})));  
        [seg_obj,~]=imread(fullfile(seg_obj_dir,sprintf('%s.png',gtids{i})));  


         seg_mask=imresize(seg_mask,[img_height,img_width],'nearest');
         seg_obj=imresize(seg_obj,[img_height,img_width],'nearest');

         seg_mask(seg_mask == 255)=0;
         seg_obj(seg_obj==255)=0;

       
         %in voc, bounding box starts from (1,1) and are given as [xmin,ymin,xmax,ymax]       
         uniq_vals=setdiff(unique(seg_obj),[0]);
      
         for m=1:length(uniq_vals)
               [seg_y,seg_x]=find(seg_obj == uniq_vals(m));
            
                x_min=min(seg_x); x_max=max(seg_x);               
                y_min=min(seg_y); y_max=max(seg_y);                
              
                bbx_width =  x_max-x_min+1;
                bbx_height = y_max-y_min+1;
                
                
                center_x  = x_min+floor(bbx_width/2);
                center_y =  y_min+floor(bbx_height/2);
           
       
                for n=1:size(seg_y)
                  offset_mask(seg_y(n),seg_x(n),1)  =  center_x-seg_x(n);
                 offset_mask(seg_y(n),seg_x(n),2)  =  center_y-seg_y(n);                    
                end 
                
       
                ind1=sub2ind([size(seg_obj),2],seg_y,seg_x,ones(size(seg_y,1),1));
                ind2=sub2ind([size(seg_obj),2],seg_y,seg_x,ones(size(seg_y,1),1)+1);
                size_mask(ind1)=bbx_width;
                size_mask(ind2)=bbx_height;       
                
          end 

 
            imdb.images{count}= I ;
            imdb.labels{count}= seg_mask+1;
            imdb.offsets{count}= cat(3,offset_mask,size_mask); 
           
         
            imdb.set(count)= uint8(set_no) ;
            imdb.id(count)= count; 
            count=count+1;
         
        
    end 
 
end 
    

save(opts.target_file, 'imdb', '-v7.3');

end 
 
    
 
 

