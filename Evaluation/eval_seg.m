 
run ..\RGBObjectDetectionSetUp.m
 
opts.baseDir = 'E:\Bharti\Code\Thesis\data\' ; 
opts.data_dir='E:\Bharti\Code\Thesis\data\VOC2012';
%opts.minoverlap=0.5;
%opts.annopath=[opts.data_dir filesep 'Annotations'];
opts.imgpath=[opts.data_dir filesep 'JPEGImages' ];
 
opts.seg.imgsetpath= fullfile(opts.data_dir, 'ImageSets', 'Segmentation' , 'val.txt');
opts.testset='val';
opts.seg.clsimgpath=[opts.data_dir '\SegmentationClass'];
 
opts.seg.clsresdir='E:\Bharti\Code\Thesis\data\rgb_object_detection-coco-voc-normalised-L2AdaptiveLoss\Results\VOCval\51\Segmentation\comp5_val_cls\';
 opts.classes={...
    'aeroplane'
    'bicycle'
    'bird'
    'boat'
    'bottle'
    'bus'
    'car'
    'cat'
    'chair'
    'cow'
    'diningtable'
    'dog'
    'horse'
    'motorbike'
    'person'
    'pottedplant'
    'sheep'
    'sofa'
    'train'
    'tvmonitor'};
 
%opts.annocachepath=[opts.data_dir filesep 'AnnotationsCache' filesep 'val_anno.mat' ];
 
opts.nclasses=20;
%VOCPATH='F:\Bharti\Thesis\data\VOC2012\';
%opts.annopath=[VOCPATH  filesep 'Annotations'];
 
VOCevalseg(opts,'comp5');
 

