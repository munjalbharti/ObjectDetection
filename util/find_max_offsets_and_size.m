base_dir='C:\Users\Bharti\Thesis\data\';
opts.constant=2000;
 
max_offset=-inf;
max_offset_ind='';

min_offset=inf;
min_offset_ind='';

max_size=-inf;
max_size_ind='';

min_size= inf;
min_size_ind='';

max_height=-inf;
max_height_ind='';

min_height=inf;
min_height_ind='';

max_width=-inf;
max_width_ind='';

min_width=inf;
min_width_ind='';


for set=1:2
    
  for k=1:2
    if(k==1)
        dataset_folder='coco';
    else
        dataset_folder='pascal';
    end 
    
    if(set==1)
        img_fold = 'train_offsets';
        label_fold='train_labels';
    else 
        img_fold = 'val_offsets';
        label_fold='val_labels';
    end 
    files_info = dir([fullfile(base_dir,dataset_folder,img_fold) filesep '*.mat']);
   
      
    for m=1:length(files_info)
        filename=files_info(m).name;
        mat_file=load(fullfile(base_dir,dataset_folder,img_fold, filename));
        [mask,map]=imread(fullfile(base_dir,dataset_folder,label_fold,sprintf('%s.png',filename(1:end-4))));
        m_c= cat(3, mask, mask);

       height= size(mask,1);
       width= size(mask,2);
       
       if(height > max_height )
           max_height=height;
           max_height_ind=filename;
       end 
       
        if(height < min_height )
           min_height=height;
           min_height_ind=filename;
       end 
       
       if(width > max_width )
           max_width=width;
           max_width_ind= filename;
       end 
       
        if(width < min_width )
           min_width=width;
           min_width_ind= filename;
       end 
       
       out=single(mat_file.offset_gt); 
       ss = single(mat_file.size_gt);
       
       %background is 1  
       out(m_c==1)=NaN;     
       out =out-opts.constant;
       
        max_o_value=max(out(:));
        min_o_value=min(out(:)); 
        
        if(max_o_value > max_offset )
            max_offset=max_o_value; 
            max_offset_ind=filename;
        end 
        
          if(min_o_value < min_offset )
            min_offset=min_o_value; 
            min_offset_ind=filename;
          end 
         
        
        ss(m_c==1)=NaN;
        max_s_value=max(ss(:));
        min_s_value=min(ss(:));
        
        if(max_s_value > max_size )
            max_size=max_s_value; 
            max_size_ind=filename;
        end 
        
          if(min_s_value < min_size )
            min_size=min_s_value; 
            min_size_ind=filename;
         end 
        
        
    end 
 end 

end 

max_offset
max_offset_ind

min_offset
min_offset_ind

max_size
max_size_ind

min_size
min_size_ind

max_height
max_height_ind

min_height
min_height_ind

max_width
max_width_ind

min_width
min_width_ind
