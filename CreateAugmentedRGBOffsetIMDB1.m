function CreateAugmentedRGBOffsetIMDB1(  )

% This script will create IMDB with train:validation ratio as already done
% in directory mask/train mask/val 
%offsets are generated online
clear;
close all ;
run RGBObjectDetectionSetUp;

% Variables (that you might want to change)

opts = struct;  
base_dir='F:\Bharti\Thesis\';
opts.data_dir=[base_dir filesep 'data' filesep 'VOC2012'];   
opts.target_file = [base_dir filesep 'data' filesep 'rgb_object_detection-15-person-offset-1-imdb.mat'];
opts.segment_class=15;
opts.segment_class_name='person';
opts.image_fixed_size=[256,256];

imdb = IMDB.init();

%Storing some meta information
[pathstr,name,~] = fileparts(opts.target_file);
imdb.meta.pathstr = pathstr;
imdb.meta.name = name;
imdb.sets.id = uint8([1 2]) ;
imdb.sets.name = {'train', 'val'} ;
imdb.classes.id = [opts.segment_class];  %wat all classes are included
imdb.classes.name = [opts.segment_class_name]; %wat are the names of the classes


%maximum_nos=1;

%image directory..images are in jpg format
imdb.paths.image = fullfile(opts.data_dir,'JPEGImages');
%masks directory for specific class "person" => 'Class_Masks/15_person/'
%masks are in png format
imdb.paths.mask  = fullfile(opts.data_dir,'Class_Masks',sprintf('%d_%s',opts.segment_class,opts.segment_class_name));
seg_obj_dir=fullfile(opts.data_dir,'SegmentationObject');

imdb.images.data = zeros(0,0,0,0,'uint8'); 
imdb.images.mask = zeros(0,0,0,0,'single');  

count=1;
no_of_aug_per_img=5;

for set_no=1:2
    if(set_no==1)
        masks_dir=fullfile(imdb.paths.mask,'train');
        offset_dir=fullfile(imdb.paths.mask ,'offset','train1');
        files_info = dir([masks_dir filesep '*.png']);
    else
        masks_dir=fullfile(imdb.paths.mask,'val'); 
        offset_dir=fullfile(imdb.paths.mask ,'offset','val1');
        files_info = dir([masks_dir filesep '*.png']);

    end
    
    %length(files_info)
for file_no=1:1
    filename= '2008_007219.png';
    
    %filename=files_info(file_no).name;
    
    orig_img= imread(fullfile(imdb.paths.image,sprintf('%s.jpg',filename(1:end-4))));
    orig_img=imresize(orig_img,opts.image_fixed_size,'nearest');
    
    [In_cls,in_map]=imread(fullfile(masks_dir,filename));
    [seg_obj,seg_map]=imread(fullfile(seg_obj_dir,filename));
       
     In_cls=imresize(In_cls,opts.image_fixed_size,'nearest');
     seg_obj=imresize(seg_obj,opts.image_fixed_size,'nearest');
     
     
     %Add image without augmentation
     [offset,obj_rects]= generate_offset(In_cls,seg_obj);
      %save offset only for non-augmented data 
     save_offset(In_cls,offset,obj_rects,offset_dir,filename(1:end-4) )
     imdb.images.data(:,:,:,count)=uint8(orig_img);
     imdb.images.id(count)=count;
     imdb.images.labels(count)=opts.segment_class;
     imdb.images.filenames{count} = filename;
     imdb.images.mask(:,:,:,count) =offset;
     imdb.images.set(count) =set_no;
     count=count+1;
        
        
   %add augmentations for training only
%{  
if(set_no ==1)
     for k=1:no_of_aug_per_img
        [img_aug,mask_aug,obj_aug]= random_transform2(orig_img,In_cls,seg_obj);
        [offset_aug,obj_rects_aug]= generate_offset(mask_aug,obj_aug);
        %save_offset(mask_aug,offset_aug,obj_rects_aug,offset_dir,sprintf('%s_aug_%d',filename(1:end-4),k) );
     
     
        imdb.images.data(:,:,:,count)=uint8(img_aug);
    
        imdb.images.id(count)=count;
        imdb.images.labels(count)=opts.segment_class;
        imdb.images.filenames{count} = filename;
        imdb.images.mask(:,:,:,count) =offset_aug;
        imdb.images.set(count) =set_no;
        
        count=count+1;
     end

   end 
 %}
     
end

end 

imdb.images.dataMean = mean(imdb.images.data(:,:,:,find(imdb.images.set == 1)), 4);
imdb.images.dataStd  =  std(single(imdb.images.data(:,:,:,find(imdb.images.set == 1))),0, 4);

%imdb = IMDB.shuffle(imdb);


% Save the actual imdb file 

save(opts.target_file, 'imdb', '-v7.3');

end

