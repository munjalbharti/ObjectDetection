function create_coco_imdb(  )
 
% This script will create IMDB with train:validation ratio as already done
% in directory mask/train mask/val 
%offsets are generated online
clear;
close all ;
run RGBObjectDetectionSetUp;
 
% Variables (that you might want to change)
 
opts = struct;  
base_dir='E:\Bharti\Code\Thesis\';
opts.data_dir=[base_dir filesep 'voc'];   
opts.target_file = [base_dir filesep 'data' filesep 'rgb_object_detection-coco-voc-imdb.mat'];


 
for set_no=1:2
    if(set_no==1)
        fold='train_images1'; 
    else
        fold='val_images1'; 
    end
    
   dir_info= dir([opts.data_dir,fold],'*.jpg');
   
      
    for k=1:length(dir_info)
         file_name= dir_info(k).name ;
         imdb.images.filenames{count}=fullfile([opts.data_dir,fold,file_name]);
         imdb.images.set(count)=set_no ;
         imdb.images.id(count)=count;
         count=count+1;
    end 
 
end 
 
    
save(opts.target_file, 'imdb', '-v7.3');
 
end
 

