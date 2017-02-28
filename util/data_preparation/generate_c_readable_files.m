in_dir='F:\Bharti\Thesis\data\VOC2012\Class_Masks\15_person\offset\train\';
out_dir='F:\Bharti\Thesis\data\VOC2012\Class_Masks\15_person\offset\train\c';
mkdir(out_dir);

files_info = dir([in_dir filesep '*.mat']);

for file_no=2:2

    filename=files_info(file_no).name;
    f=load(fullfile(in_dir,filename));
    off=f.offset;
   
    fileID = fopen(fullfile(out_dir,sprintf('%s_offset.txt',filename(1:end-4))),'w');
   
   
   for m=1:size(off,1)
     for n=1:size(off,2)            
         fprintf(fileID,'%.2f %.2f \n',off(m,n,1),off(m,n,2));
     end 
   end 
    fclose(fileID);
    rects=f.obj_rects;
    x=rects.x_mins;
    y=rects.y_mins;
    w=rects.widths;
    h=rects.heights;
    
    fileID = fopen(fullfile(out_dir,sprintf('%s_rects.txt',filename(1:end-4))),'w');
    
    if(~isempty(x)) 
        fprintf(fileID,'%d ',x);    
        fprintf(fileID,'\n');
        fprintf(fileID,'%d ',y);
        fprintf(fileID,'\n');
        fprintf(fileID,'%d ',w);
        fprintf(fileID,'\n');
        fprintf(fileID,'%d ',h);
    end
    
     fclose(fileID);
end
        