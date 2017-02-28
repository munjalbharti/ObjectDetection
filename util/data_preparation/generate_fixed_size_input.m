clear ;
close all;
run ../../RGBObjectDetectionSetUp.m;

segment_class=15;
segment_class_name='person';
class_dir=fullfile('../../data/VOC2012','Class_Masks',sprintf('%d_%s',segment_class,segment_class_name));
out_dir=fullfile('../../data/VOC2012/Fixed_Size_Input/','',sprintf('%d_%s',segment_class,segment_class_name));

out_img_dir=fullfile(out_dir,'Images');
out_masks_dir=fullfile(out_dir,'Masks');

 if not (exist(out_img_dir,'dir')==7)
        mkdir(out_img_dir);
 end
 
  if not (exist(out_masks_dir,'dir')==7)
        mkdir(out_masks_dir);
 end
    
  
fixed_width=228;
fixed_height=304;

files_info=dir([ class_dir filesep '*.png']);
total_files=size(files_info,1);
half_patch_size_x=114;
half_patch_size_y=152;

for file_no=1:total_files
      filename=files_info(file_no).name ;
      mask=imread(fullfile(class_dir,filename));
      im=imread(fullfile('../../data/VOC2012','JPEGImages',sprintf('%s.jpg',filename(1:end-4))));
      
      h=size(im,1);
      w=size(im,2);      
   
      if(h >= fixed_height && w >= fixed_width)  
         [c_img,c_mask] = crop_randomly(im,mask,fixed_height,fixed_width);
         
         imwrite(c_img,fullfile(out_img_dir,filename));
         imwrite(c_mask,fullfile(out_masks_dir,filename));
       
         %I2 = imcrop(im,[1,1,227,303]);
         %M2 = imcrop(mask,[1,1,227,303]);
      
      end
      
     
      
      
end 
