function CreateRGBObjectDetectionIMDB1()

% This script to prepare IMDB for object detection
% The imdb will have train:validation ratio same as in VOC 2012 dataset


clear;
close all ;

run RGBObjectDetectionSetUp;

opts = struct;   
opts.dataDir=['data' filesep 'VOC2012']; 
opts.target_file = ['data' filesep 'rgb_object_detection-15-person-train-val.imdb.mat'];
opts.segment_class=15;
opts.segment_class_name='person';
opts.image_fixed_size=[256,256];    
%options.includeTest =false; % To enable this download VOC2012 test set


imdb = IMDB.init();

%Storing some meta information
[pathstr,name,~] = fileparts(opts.target_file);
imdb.meta.pathstr = pathstr;
imdb.meta.name = name;
imdb.sets.id = uint8([1 2]) ;
imdb.sets.name = {'train', 'val'} ;


%image directory..images are in jpg format
imdb.paths.image = fullfile(opts.dataDir, 'JPEGImages') ;
%masks directory for specific class "person" => 'Class_Masks/15_person/'
%masks are in png format
imdb.paths.mask = fullfile(opts.dataDir, 'Class_Masks',sprintf('%d_%s',opts.segment_class,opts.segment_class_name)) ;
imdb.paths.orig_segmentation=fullfile(opts.data_dir,'SegmentationClass');

imdb.images.data = zeros(0,0,0,0,'uint8'); 
imdb.images.mask = zeros(0,0,0,0,'uint8');  

%add train and val data to imdb
index = containers.Map() ;
[imdb, index]=  addSegmentationSet(opts, imdb, index, 'train', 1);
[imdb, index] = addSegmentationSet(opts, imdb, index, 'val', 2) ;

%if opts.includeTest, [imdb, index] = addSegmentationSet(opts, imdb, index, 'test', 3) ; end

% Compress data types
imdb.images.id = uint32(imdb.images.id) ;
imdb.images.set = uint8(imdb.images.set) ;

%Storing image sizes as meta data..can ignore!!
imdb = getImageSizes(imdb) ;

imdb.images.dataMean = mean(imdb.images.data(:,:,:,find(imdb.images.set == 1)), 4);
imdb.images.dataStd  =  std(single(imdb.images.data(:,:,:,find(imdb.images.set == 1))),0, 4);

imdb = IMDB.shuffle(imdb);
save(opts.target_file, 'imdb', '-v7.3');

end 


function [imdb, index] = addSegmentationSet(opts, imdb, index, setName, setCode)
% -------------------------------------------------------------------------
segAnnoPath = fullfile(opts.dataDir, 'ImageSets', 'Segmentation_Per_Class',sprintf('%s_%s.txt',opts.segment_class_name,setName)) ;
fprintf('%s: reading %s\n', mfilename, segAnnoPath) ;
segFileNames = textread(segAnnoPath, '%s') ;

j = numel(imdb.images.id) ;
for i=1:length(segFileNames)
   segName=segFileNames{i};
   if index.isKey(segName)
        k = index(segName) ;
        setCode=imdb.images.set(k) ;
        fprintf('image %s: is already present in %s Set\n', segName, setCode) ;
   else      
        j = j + 1 ;
        index(segName) = j ;
        imdb.images.id(j) = j ;
        imdb.images.set(j) = setCode ;
        imdb.images.filenames{j} = segName ;
        imdb.images.labels(j)=opts.segment_class;
        
        orig_img=imread(sprintf('%s.jpg',fullfile(imdb.paths.image,segName)));
        imdb.images.data(:,:,:,j)=uint8(imresize(orig_img,opts.image_fixed_size,'nearest'));
        
        [orig_mask,~]=imread(sprintf('%s.png',fullfile(imdb.paths.mask,segName)));
        
        resizd_mask=imresize(orig_mask,opts.image_fixed_size,'nearest');          
        imdb.images.mask(:,:,:,j) =uint8(resizd_mask);
    
  end
end

end 

function imdb = getImageSizes(imdb)
    for j=1:numel(imdb.images.id)
        info = imfinfo(sprintf('%s.jpg',fullfile(imdb.paths.image, imdb.images.filenames{j}))) ;
        imdb.images.size(:,j) = uint16([info.Width ; info.Height]) ;
       % fprintf('%s: checked image %s [%d x %d]\n', mfilename, imdb.images.filenames{j}, info.Height, info.Width) ;
    end
end 


