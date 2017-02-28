
segment_class_ids=[1:20];
segment_class_names={'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','plotted_plant','sheep','sofa','train','tv'};

opts.dataDir='../../data/VOC2012/';
opts.out_dir= fullfile(opts.dataDir,'ImageSets/Segmentation_Per_Class/');
opts.out_maks_dir = fullfile(opts.dataDir, 'Class_Masks1') ;
opts.classSegmentationDir = fullfile(opts.dataDir, 'SegmentationClass') ; %png


train_images_per_class=cell(1,20);
val_images_per_class=cell(1,20);

if not (exist(opts.out_dir,'dir')==7)
        mkdir(opts.out_dir);
end
 
if not (exist(opts.out_maks_dir,'dir')==7)
        mkdir(opts.out_maks_dir);
end
 
 setsNames={'train','val'};
 
 for i=1:length(setsNames)
    setName=setsNames{i};
    segAnnoPath = fullfile(opts.dataDir, 'ImageSets', 'Segmentation', [setName '.txt']) ;
    fprintf('%s: reading %s\n', mfilename, segAnnoPath) ;
 
    segFileNames = textread(segAnnoPath, '%s') ;
    for j=1:length(segFileNames)
        file_name=sprintf('%s.png',segFileNames{j});
        segmented_image=imread(fullfile(opts.classSegmentationDir,file_name));
        
        for class_id = segment_class_ids
             class_name=segment_class_names{class_id};
             class_pixels=find(segmented_image == class_id);
            
             if(class_pixels > 0)
                 mask=zeros(size(segmented_image));
                 mask(class_pixels)=1;
                 
                 drc=fullfile(opts.out_maks_dir,sprintf('%d_%s',class_id,class_name));
                 if not (exist(drc,'dir')==7)
                     mkdir(drc);
                 end
                 imwrite(mask,fullfile(drc,file_name));
                 
                if(i==1)
                    train_images_per_class{class_id}{end+1}=segFileNames{j};
                else 
                    val_images_per_class{class_id}{end+1}=segFileNames{j};
                end
             end    
             
        end 
    end
 end 
for class_id = segment_class_ids
    for i=1:length(setsNames)   
        fileID = fopen(fullfile(opts.out_dir,sprintf('%s_%s.txt',segment_class_names{class_id},setsNames{i})),'w');
        if (i==1)
            for row_no=1:size(train_images_per_class{class_id},2)
                fprintf(fileID,'%s\n',train_images_per_class{class_id}{row_no});
            end
        else 
            for row_no=1:size(val_images_per_class{class_id},2)
                fprintf(fileID,'%s\n',val_images_per_class{class_id}{row_no});
            end            
        end 
          
         fclose(fileID);
       
     end 
end 