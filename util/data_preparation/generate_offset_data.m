clear ;
close all;
run ../../RGBObjectDetectionSetUp.m;

segment_class=15;
segment_class_name='person';

base_dir=fullfile('..',  '..' , 'data', 'VOC2012','Class_Masks',sprintf('%d_%s',segment_class,segment_class_name));
seg_obj_dir=fullfile('..' ,'..' ,'data', 'VOC2012','SegmentationObject');

for k=2:2
    if(k==1)       
        seg_out_dir=fullfile(base_dir,'offset','train');
        class_dir=fullfile(base_dir,  'train');
       
        files_info=dir([class_dir filesep '*.png']);
        total_files=size(files_info,1);
        
        if not (exist(seg_out_dir,'dir')==7)
            mkdir(seg_out_dir);
         end

    else
        
       
        seg_out_dir=fullfile(base_dir,'offset','val');
        class_dir=fullfile(base_dir,'val');
        
        files_info=dir([class_dir filesep '*.png']);
        total_files=size(files_info,1);

        if not (exist(seg_out_dir,'dir')==7)
          mkdir(seg_out_dir);
        end

    
    
    end
    
    for file_no=1:total_files
       file_name=files_info(file_no).name;
  
       [In_cls,in_map]=imread(fullfile(class_dir,file_name));
       [seg_obj,seg_map]=imread(fullfile(seg_obj_dir,file_name));
       
       In_cls=imresize(In_cls,[256,256],'nearest');
       seg_obj=imresize(seg_obj,[256,256],'nearest');
       
       [offset_img] = generate_offset(In_cls,seg_obj);
     
       save(fullfile(seg_out_dir,sprintf('%s.mat',file_name(1:end-4))),'offset_img')
    end  
end 
