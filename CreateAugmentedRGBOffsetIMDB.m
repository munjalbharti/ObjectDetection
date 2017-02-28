function CreateAugmentedRGBOffsetIMDB(  )

% This script will create IMDB with train:validation ratio as configured in
% the script using opts.partitions

clear;
close all ;
%run RGBObjectDetectionSetUp;

% Variables (that you might want to change)

opts = struct;   
opts.data_dir=['data' filesep 'VOC2012'];   
opts.target_file = ['data' filesep 'rgb_object_detection-15-person-offset-4440.imdb.mat'];
opts.segment_class=15;
opts.segment_class_name='person';
opts.image_fixed_size=[256,256];
opts.partitions = [0.8 0.2 0]; % How to partition the imdb struct into training, validation and testing data

imdb = IMDB.init();

%Storing some meta information
[pathstr,name,~] = fileparts(opts.target_file);
imdb.meta.pathstr = pathstr;
imdb.meta.name = name;
imdb.sets.id = uint8([1 2]) ;
imdb.sets.name = {'train', 'val'} ;
imdb.classes.id = [opts.segment_class];  %wat all classes are included
imdb.classes.name = [opts.segment_class_name]; %wat are the names of the classes


maximum_nos=1;
count=1;
%image directory..images are in jpg format
imdb.paths.image = fullfile(opts.data_dir,'JPEGImages');
%masks directory for specific class "person" => 'Class_Masks/15_person/'
%masks are in png format
imdb.paths.mask  = fullfile(opts.data_dir,'Class_Masks',sprintf('%d_%s',opts.segment_class,opts.segment_class_name));
seg_obj_dir=fullfile(opts.data_dir,'SegmentationObject');

imdb.images.data = zeros(0,0,0,0,'uint8'); 
imdb.images.mask = zeros(0,0,0,0,'single');  


%Read masks of class 'person'
files_info = dir([imdb.paths.mask filesep '*.png']);
no_of_aug_per_img=5;
for file_no=1:length(files_info)
    %filename= '2008_007219.png';
    
    filename=files_info(file_no).name;
    
    orig_img= imread(fullfile(imdb.paths.image,sprintf('%s.jpg',filename(1:end-4))));
    orig_img=imresize(orig_img,opts.image_fixed_size,'nearest');
    
    [In_cls,in_map]=imread(fullfile(imdb.paths.mask,filename));
    [seg_obj,seg_map]=imread(fullfile(seg_obj_dir,filename));
       
     In_cls=imresize(In_cls,opts.image_fixed_size,'nearest');
     seg_obj=imresize(seg_obj,opts.image_fixed_size,'nearest');
   
     for k=1:no_of_aug_per_img
        [img_aug,mask_aug,obj_aug]= random_transform2(orig_img,In_cls,seg_obj);
        [offset,~]= generate_offset(mask_aug,obj_aug);
    
       
        imdb.images.data(:,:,:,count)=uint8(img_aug);
    
        imdb.images.id(count)=count;
        imdb.images.labels(count)=opts.segment_class;
        imdb.images.filenames{count} = filename;
        imdb.images.mask(:,:,:,count) =offset;
        count=count+1;
     end 
   % if(file_no == maximum_nos)  
    %    break ;
    %end
end

imdb = IMDB.repartition1(imdb, opts.partitions);

imdb.images.dataMean = mean(imdb.images.data(:,:,:,find(imdb.images.set == 1)), 4);
imdb.images.dataStd  =  std(single(imdb.images.data(:,:,:,find(imdb.images.set == 1))),0, 4);

imdb = IMDB.shuffle(imdb);


% Save the actual imdb file 
save(opts.target_file, 'imdb', '-v7.3');

end
