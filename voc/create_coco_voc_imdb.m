function create_coco_voc_imdb(  )
 
% This script will create IMDB with train:validation ratio as already done
% in directory mask/train mask/val 
%offsets are generated online
clear;
close all ;
%run RGBObjectDetectionSetUp;
 
% Variables (that you might want to change)
 
opts = struct;  
base_dir='E:\Bharti\Code\Thesis\';
opts.data_dir=[base_dir filesep 'voc'];   
opts.target_file = [base_dir filesep 'data' filesep 'rgb_object_detection-coco-voc-imdb.mat'];
opts.image_fixed_size=[256,256];

%load voc imdb
load([base_dir filesep 'data' filesep 'rgb_object_detection-coco-imdb.mat']);

count=length(imdb.images.set);
 
for set_no=1:2
    if(set_no==1)
        fold='train_images'; 
    else
        fold='val_images'; 
    end
 
   files_info = dir([fullfile(opts.data_dir,fold) filesep '*.jpg']);
   
      
    for k=1:length(files_info)
         count=count+1;
         imdb.images.filenames{count}=files_info(k).name;
         imdb.images.set(count)=set_no ;
         imdb.images.id(count)=2;
       
    end 
 
end 
 
    
save(opts.target_file, 'imdb', '-v7.3');
 
end
 



