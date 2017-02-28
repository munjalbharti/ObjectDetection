function[offset_img,rects_struct] = generate_offset_and_class(In_cls_aug,seg_obj_aug)
     %  file_name=files_info(file_no).name;
  
         rects_struct = struct('x_mins',[],'y_mins',[],'widths',[],'heights',[]);
          
         uniq_vals=setdiff(unique(seg_obj_aug),[0,255]);
       
         offset_img=zeros(size(In_cls_aug,1),size(In_cls_aug,2),3,'single');
         offset_img(:,:,1)=NaN;
         offset_img(:,:,2)=NaN;
         offset_img(:,:,3)=1;
       
    
       
         for k=1:length(uniq_vals)
            [seg_y,seg_x]=find(seg_obj_aug == uniq_vals(k));
            
            if(In_cls_aug(seg_y(1),seg_x(1)) == 255)
                x_min=min(seg_x);
                x_max=max(seg_x);
                
                y_min=min(seg_y);
                y_max=max(seg_y);
                
                width = x_max-x_min+1;
                height = y_max-y_min+1;
                
                
                center_x  = x_min+floor((x_max-x_min)/2);
                center_y =  y_min+floor((y_max-y_min)/2);
                
                for k=1:size(seg_y)
                    offset_img(seg_y(k),seg_x(k),1)  =  center_x-seg_x(k);
                    offset_img(seg_y(k),seg_x(k),2)  =  center_y-seg_y(k); 
                    offset_img(seg_y(k),seg_x(k),3)  =  2; 
                    
                end 
                
                rects_struct.x_mins=[rects_struct.x_mins;x_min];
                rects_struct.y_mins=[rects_struct.y_mins;y_min];
                rects_struct.widths=[rects_struct.widths;width];
                rects_struct.heights=[rects_struct.heights;height];
                
                
            
                
            end 
            
           
       
         end
       
      
     
end 

       
      

