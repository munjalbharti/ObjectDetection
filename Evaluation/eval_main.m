
clear;
run ..\RGBObjectDetectionSetUp.m

opts.baseDir = 'E:\Bharti\Code\Thesis\data\' ; 
opts.data_dir=[opts.baseDir filesep 'VOC2012'];

 opts.minoverlap=0.5;
 opts.annopath=[opts.data_dir filesep 'Annotations'];
 opts.imgpath=[opts.data_dir filesep 'JPEGImages' ];
 opts.imgsetpath= fullfile(opts.data_dir, 'ImageSets', 'Main' , 'val.txt');
%opts.imgsetpath= fullfile(opts.data_dir, 'ImageSets', 'Segmentation' , 'val.txt');
 
opts.resultDir=fullfile('E:\Bharti\Code\Thesis\data\rgb_object_detection-coco-voc-normalised-L2AdaptiveLoss\Results\VOCval\51\hough_th_5_bin_5_win_9_with_nmx_conf_avg_time\');
opts.annocachepath=[opts.data_dir filesep 'AnnotationsCache' filesep 'val_anno.mat' ];
%opts.annocachepath=[opts.data_dir filesep 'AnnotationsCache' filesep 'val_seg_anno.mat' ];

opts.imdbPath = fullfile([opts.baseDir filesep  'imdbVOC2012_segmentation_offsets_ssc256.mat']) ;

  

%opts.masks_dir=fullfile(opts.data_dir,'Class_Masks',sprintf('%d_%s',opts.segment_class,opts.segment_class_name));
       
%opts.offset_dir=fullfile(opts.masks_dir,'offset','val');
    
%imdb = IMDB.load(opts.imdbPath);

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
 
opts.nclasses=length(opts.classes);
VOCPATH=[opts.baseDir filesep 'VOC2012'];
opts.annopath=[VOCPATH  filesep 'Annotations'];

pre='ALL_0_perct_orig';
eval_dir=fullfile( opts.resultDir,pre);
mkdir(eval_dir);
results = cell(20,3);
sum=0;
for k=1:20
  class_name=opts.classes{k};
  %[ap,total_gt,total_det,td,fd]=  VOCevaldet1(opts,imdb,class_name,true);
  %[ap,total_gt,total_det,td,fd]= VOCevaldet_orig_disp(opts,'id',class_name,true,pre);
  [ap,total_gt,total_det,td,fd]= VOCevaldet_orig(opts,'id',class_name,true);
   
 results{k,1}=class_name;
  results{k,2}=num2str(ap);
  sum=sum+ap;
  results{k,3}= strcat(num2str(total_gt),'/',num2str(total_det),'/',num2str(td),'/',num2str(fd));
   %if k<opts.nclasses
   %     fprintf('press any key to continue with next class...\n');
   %     drawnow;
        
   %     pause;
   % end
    
end 

avg=sum/20;


fid = fopen(fullfile(eval_dir,'results.csv'),'wt');
 if fid>0
     fprintf(fid,'%s,%s,%s\n','Class','AP','Total_gt/Total_det/True Det/False Det');
     for k=1:size(results,1)
         fprintf(fid,'%s,%s,%s\n',results{k,:});
     end
     fprintf(fid,'%s,%s\n','Avg',num2str(avg));
     fclose(fid);
 end


