function[offset_img,rects_struct,boundary] = generate_offset_and_class_and_boundary_mask(In_cls_aug,seg_obj_aug)
     %  file_name=files_info(file_no).name;
  
         rects_struct = struct('x_mins',[],'y_mins',[],'widths',[],'heights',[]);
          
         uniq_vals=setdiff(unique(seg_obj_aug),[0,255]);
       
         offset_img=zeros(size(In_cls_aug,1),size(In_cls_aug,2),3,'single');
         offset_img(:,:,1)=NaN;
         offset_img(:,:,2)=NaN;
         offset_img(:,:,3)=1;
         
         boundary=zeros(size(In_cls_aug,1),size(In_cls_aug,2),'single');
         
       
    
       
         for k=1:length(uniq_vals)
            [seg_y,seg_x]=find(seg_obj_aug == uniq_vals(k));
            
            if(In_cls_aug(seg_y(1),seg_x(1)) == 255)
                obj=zeros(size(In_cls_aug,1),size(In_cls_aug,2),'single');
                obj(sub2ind(size(seg_obj_aug),seg_y,seg_x))=1;
                no_of_pixels_in_obj=size(seg_y,1);
                if(no_of_pixels_in_obj <= 3000)
                    se = ones(5,5);
                else 
                    if(3000 < no_of_pixels_in_obj && no_of_pixels_in_obj <= 7000)
                        se = ones(11,11);
                    else
                        if( 7000 < no_of_pixels_in_obj && no_of_pixels_in_obj <= 20000) 
                            se = ones(25,25);
                        else 
                            if(  20000 < no_of_pixels_in_obj && no_of_pixels_in_obj <= 35000)
                                 se = ones(35,35);
                            else 
                                 se = ones(45,45);
                            end 
                        end 
                    end
                end 
                
                eroded_obj=imerode(obj,se);
                obj_boundary=obj-eroded_obj;
                
               % f1=figure;
               % imshow(obj);
                
               % f2=figure;
               % imshow(obj_boundary);
                
               % close(f1);
               % close(f2);
                
                
                
                
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
                
                boundary=boundary | obj_boundary;
                
                rects_struct.x_mins=[rects_struct.x_mins;x_min];
                rects_struct.y_mins=[rects_struct.y_mins;y_min];
                rects_struct.widths=[rects_struct.widths;width];
                rects_struct.heights=[rects_struct.heights;height];
                
                
            
                
            end 
            
           
       
         end
       
      
     
end 

       
      

