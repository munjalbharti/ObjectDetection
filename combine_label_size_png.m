function combine_label_size_png(  )
 
% This script will create IMDB with train:validation ratio as already done
% in directory mask/train mask/val 
%offsets are generated online
clear;
close all ;
%run RGBObjectDetectionSetUp;
 
% Variables (that you might want to change)
 
opts = struct;  
base_dir='C:\Users\Bharti\Thesis\data\';
 
for set_no=1:2
    for dataset=1:2
        if(dataset==1)
             dataset_folder='coco';
        else 
             dataset_folder='voc';
        end 
        
        if(set_no==1)
            img_fold='train_images'; 
            labl_fold='train_labels';
            offset_fold='train_offsets1_uint8'; %sizes are stored as uint8
            target_folder='train_labels_sizes_uint8';
 
        else
            img_fold='val_images';
            labl_fold='val_labels';
            offset_fold='val_offsets1_uint8'; %sizes are stored as uint8
            target_folder='val_labels_sizes_uint8';
        end
 
         files_info = dir([fullfile(base_dir,dataset_folder,img_fold) filesep '*.jpg']);
   
      
    for k=1:length(files_info)
         filename=files_info(k).name;
         
         
         size_x= imread(fullfile(base_dir,dataset_folder,offset_fold,sprintf('%s_sx.png',filename(1:end-4))));
         size_y= imread(fullfile(base_dir,dataset_folder,offset_fold,sprintf('%s_sy.png',filename(1:end-4))));
       %  ind=find(size_x >= 255 | size_y >= 255);
       %  if(~isempty(ind))
       %     disp('here');
       % end 
         
         label= imread(fullfile(base_dir,dataset_folder,labl_fold,sprintf('%s.png',filename(1:end-4))));
 
        label_size_comb= cat(3,label,size_x,size_y);         
 
        imwrite(label_size_comb,fullfile(base_dir,dataset_folder,target_folder,sprintf('%s.png',filename(1:end-4))));
 
       
    end 
 
    end 
end 
 
 
 
end
 
 
 
 

