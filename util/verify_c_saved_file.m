out_dir='F:\Bharti\Thesis\data\VOC2012\Class_Masks\15_person\offset\train\c';
filename='2007_000129.mat';


M = dlmread(fullfile(out_dir,sprintf('%s_offset.txt',filename(1:end-4))));

offset1=zeros(256,256,2,'single');

for k=1:size(M,1)
    
    row=ceil(k/256);
    col= k-256*(row-1) ; 
    if(k == 65536)
        disp('test');
    end 
    
    offset1(row,col,1)=M(k,1);
    offset1(row,col,2)=M(k,2);

end 