in_dir=['VOC2012' filesep 'SegmentationClass'];
segment_class_names={'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','plotted_plant','sheep','sofa','train','tv'};
for segment_class=1:20
    min_width=10000;
    min_height=10000;
    count=0;

    segment_class_name=segment_class_names{segment_class};

    out_dir=fullfile(['VOC2012' filesep 'Class' filesep sprintf('%d_%s',segment_class,segment_class_name)]);

    if not (exist(out_dir,'dir')==7)
        mkdir(out_dir);
    end

    files_info=dir([in_dir filesep '*.png']);
    total_files=size(files_info,1);
    for file_no=1:total_files
         file_name=files_info(file_no).name;
         [I_in,map]=imread(fullfile(in_dir,file_name));
         class_pixels=find(I_in==segment_class);
         h=size(I_in,1);
         w=size(I_in,2);
         I_out=zeros(h,w);
         
         if(class_pixels > 0)
            %fprintf('class %s found in image  %d',segment_class_name,file_name);
            I_out(class_pixels)=1;
            imwrite(I_out,fullfile(out_dir,file_name));
            if(min_height > h)
                min_height=h;
            end 
         
            if(min_width > w)
                min_width=w;
            end 
            
            if(h >=304 && w >= 228)
                count=count+1;
            end 
         end 
         
    end 
     
    fprintf('Segment class %s Min-Height %d Min-Width %d\n',segment_class_name,min_height,min_width);
     fprintf('Segment class %s Height_Width_greater %d\n',segment_class_name,count);


end 
