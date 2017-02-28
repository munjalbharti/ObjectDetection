function [] = results_resize_val()
 
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
opts.baseDir = 'E:\Bharti\Code\Thesis\data\' ; 
opts.data_dir=[opts.baseDir filesep 'VOC2012'];
opts.imgpath=[opts.data_dir filesep 'JPEGImages' ];

opts.resultDir='E:\Bharti\Code\Thesis\data\rgb_object_detection-coco-voc-normalised-L2Loss\Results\VOCval\145\hough_th_5_bin_5_win_9_with_nmx_conf_avg\';
    
for k=1:opts.nclasses

  det_result_file1=sprintf('%s_det_val_%s.txt','comp3',opts.classes{k});
  filename=fullfile(opts.resultDir,det_result_file1);
  [ids,b1,b2,b3,b4,confidence]=textread(filename,'%s %f %f %f %f %f');
  
  BB=[b1 b2 b3 b4]';

  nd=length(confidence);
  det_result_file2=sprintf('%s_det_val_%s.txt','comp4',opts.classes{k});
  fid=fopen(fullfile(opts.resultDir,det_result_file2),'w');
       
   for d=1:nd
        I=imread(fullfile(opts.imgpath,sprintf('%s.jpg',ids{d})));
        im_width=size(I,2);
        im_height=size(I,1);

        if(im_height < im_width)
             im_=imresize(I,[256,NaN]);
         else 
             im_=imresize(I,[NaN,256]);
        end 
        bb=BB(:,d);

        r_min_x= round(bb(1) * im_width/size(im_,2));
        r_max_x= round(bb(3) * im_width/ size(im_,2));
        r_min_y= round(bb(2) * im_height/size(im_,1));
        r_max_y= round(bb(4) * im_height/size(im_,1));
                                    
       fprintf(fid,'%s %f %f %f %f %f \n',ids{d},r_min_x,r_min_y,r_max_x,r_max_y,confidence(d));      
   end 
   
    fclose(fid);
    
end 
end 