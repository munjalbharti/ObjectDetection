function CreateRGBOffsetIMDB1(  )

% This script will create IMDB with train:validation ratio as configured in
% the directories mask/train mask/val
%The offset will be generated online
clear;
close all ;
run RGBObjectDetectionSetUp;

% Variables (that you might want to change)

opts = struct;   
opts.data_dir=['data' filesep 'VOC2012'];   
opts.target_file = ['data' filesep 'rgb_object_detection-15-person-offset-888-new.imdb.mat'];
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


maximum_nos=1;
%image directory..images are in jpg format
imdb.paths.image = fullfile(opts.data_dir,'JPEGImages');
%masks directory for specific class "person" => 'Class_Masks/15_person/'
%masks are in png format
imdb.paths.mask  = fullfile(opts.data_dir,'Class_Masks',sprintf('%d_%s',opts.segment_class,opts.segment_class_name));
seg_obj_dir=fullfile(opts.data_dir,'SegmentationObject');

imdb.images.data = zeros(0,0,0,0,'uint8'); 
imdb.images.mask = zeros(0,0,0,0,'single');  


%Read masks of class 'person'
count=1;
for set_no=1:2
    if(set_no==1)
        masks_dir=fullfile(imdb.paths.mask,'train');
        files_info = dir([masks_dir filesep '*.png']);

    else
        masks_dir=fullfile(imdb.paths.mask,'val'); 
        files_info = dir([masks_dir filesep '*.png']);

    end
    
    for file_no=1:length(files_info)
        %filename= '2008_007219.mat';

        filename=files_info(file_no).name;

        orig_img= imread(fullfile(imdb.paths.image,sprintf('%s.jpg',filename(1:end-4))));

        imdb.images.data(:,:,:,count)=uint8(imresize(orig_img,opts.image_fixed_size,'nearest'));

        imdb.images.id(count)=count;
        imdb.images.labels(count)=opts.segment_class;
        imdb.images.filenames{count} = filename;

         [In_cls,in_map]=imread(fullfile(masks_dir,filename));
         [seg_obj,seg_map]=imread(fullfile(seg_obj_dir,filename));
         In_cls=imresize(In_cls,opts.image_fixed_size,'nearest');
         seg_obj=imresize(seg_obj,opts.image_fixed_size,'nearest');
     
         [offset,~]= generate_offset(In_cls,seg_obj);
        %load(fullfile(masks_dir,filename),'offset_img'); 


        imdb.images.mask(:,:,:,count) =offset;
        
        
        imdb.images.set(count)=set_no;
        
        count=count+1;
       % if(file_no == maximum_nos)  
        %    break ;
        %end
    end
end 
%imdb = IMDB.repartition1(imdb, opts.partitions);

imdb.images.dataMean = mean(imdb.images.data(:,:,:,find(imdb.images.set == 1)), 4);
imdb.images.dataStd  =  std(single(imdb.images.data(:,:,:,find(imdb.images.set == 1))),0, 4);

imdb = IMDB.shuffle(imdb);


% Save the actual imdb file 
save(opts.target_file, 'imdb', '-v7.3');

end
