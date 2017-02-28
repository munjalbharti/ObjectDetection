opts.offset_dir='F:\Bharti\RGBObjectDetection\data\rgb_object_detection-15-person-test-4794images-offset-rmse-new-loss-new-imdb\Results\143\';
opts.img_dir='F:\Bharti\RGBObjectDetection\data\VOC2012\JPEGImages\';


filename='2011_002075';
threshold=500;
bin_size=[1,1];

%load offset
load(fullfile(opts.offset_dir,sprintf('%s.mat',filename)));
[centers_y,centers_x]=find_object_centers(offset,bin_size,threshold);


figure;
imshow(imread(fullfile(opts.img_dir,sprintf('%s.jpg',filename))));
hold on ;
plot(center_x,center_y,'r+','MarkerSize', 12);
quiver([1:size(offset,1)],[1:size(offset,2)],offset(:,:,1),offset(:,:,2));
impixelinfo ;