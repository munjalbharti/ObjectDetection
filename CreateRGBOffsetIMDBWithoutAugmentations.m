function CreateRGBOffsetIMDB(  )

% This script will create IMDB with train:validation ratio as configured in
% the script using opts.partitions
%The offsets will be loaded from directory

clear;
close all ;
run RGBObjectDetectionSetUp;

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
%image directory..images are in jpg format
imdb.paths.image = fullfile(opts.data_dir,'JPEGImages');
%masks directory for specific class "person" => 'Class_Masks/15_person/'
%masks are in png format
imdb.paths.mask  = fullfile(opts.data_dir,'Class_Masks',sprintf('%d_%s',opts.segment_class,opts.segment_class_name),'offset');

imdb.images.data = zeros(0,0,0,0,'uint8'); 
imdb.images.mask = zeros(0,0,0,0,'single');  


%Read masks of class 'person'
files_info = dir([imdb.paths.mask filesep '*.mat']);

for file_no=1:length(files_info)
    %filename= '2008_007219.mat';
    
    filename=files_info(file_no).name;
    
    orig_img= imread(fullfile(imdb.paths.image,sprintf('%s.jpg',filename(1:end-4))));
    
    imdb.images.data(:,:,:,file_no)=uint8(imresize(orig_img,opts.image_fixed_size,'nearest'));
    
    imdb.images.id(file_no)=file_no;
    imdb.images.labels(file_no)=opts.segment_class;
    imdb.images.filenames{file_no} = filename;
    
    load(fullfile(imdb.paths.mask,filename),'offset_img'); 
    
    
    imdb.images.mask(:,:,:,file_no) =offset_img;
    
    if(file_no == maximum_nos)  
        break ;
    end
end

imdb = IMDB.repartition1(imdb, opts.partitions);

imdb.images.dataMean = mean(imdb.images.data(:,:,:,find(imdb.images.set == 1)), 4);
imdb.images.dataStd  =  std(single(imdb.images.data(:,:,:,find(imdb.images.set == 1))),0, 4);

imdb = IMDB.shuffle(imdb);


% Save the actual imdb file 
save(opts.target_file, 'imdb', '-v7.3');

end
