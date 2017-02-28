function create_coco_voc_offset_imdb(  )
 

clear;
close all ;
%run RGBObjectDetectionSetUp;
 
% Variables (that you might want to change)
 
opts = struct;  
base_dir='C:\Users\Bharti\Thesis\data';
opts.target_file = [base_dir filesep 'rgb_object_detection-coco-voc-offset-ssd.mat'];
count=0;
 
for set_no=1:2
    for dataset=1:2
        if(dataset==1)
             dataset_folder='coco';
        else 
             dataset_folder='voc';
        end 
        
        if(set_no==1)
            img_fold='train_images'; 
            labl_size_fold='train_labels_sizes_uint8';
            offset_fold='train_offsets_mat_int';
 
        else
            img_fold='val_images';
            labl_size_fold='val_labels_sizes_uint8';
            offset_fold='val_offsets_mat_int';
        end
 
         files_info = dir([fullfile(base_dir,dataset_folder,img_fold) filesep '*.jpg']);
   
      
    for k=1:length(files_info)
         count=count+1;
         filename=files_info(k).name;
         
         imdb.images.filenames{count}=fullfile(base_dir,dataset_folder,img_fold,filename);
         imdb.images.label_size_filenames{count}= fullfile(base_dir,dataset_folder,labl_size_fold,sprintf('%s.png',filename(1:end-4)));
         mat_file= load(fullfile(base_dir,dataset_folder,offset_fold,sprintf('%s.mat',filename(1:end-4))));
         imdb.images.offsets{count}= mat_file.offset_gt ;
         
         
         %imdb.images.offset_x_filenames{count}= fullfile(base_dir,dataset_folder,offset_fold,sprintf('%s_ox.png',filename(1:end-4)));
         %imdb.images.offset_y_filenames{count}= fullfile(base_dir,dataset_folder,offset_fold,sprintf('%s_oy.png',filename(1:end-4)));
         
         %imdb.images.size_x_filenames{count}= fullfile(base_dir,dataset_folder,offset_fold,sprintf('%s_sx.png',filename(1:end-4)));
         %imdb.images.size_y_filenames{count}= fullfile(base_dir,dataset_folder,offset_fold,sprintf('%s_sy.png',filename(1:end-4)));
         imdb.images.set(count)= uint8(set_no) ;
         imdb.images.id(count)= uint8(dataset); % 1 for coco
        
    end 
 
    end 
end 
 
    
save(opts.target_file, 'imdb', '-v7.3');
 
end
 
 

