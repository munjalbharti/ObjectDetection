function [ cropped_img ] = crop_at_point( imgs,center_point,height,width )
%CROP_AT_CENTER  Given an image, a point and a patchsize, extract a patch
%around that point
% imgs: the input image
% masks: corresponding masks
% center_point: a 2-element vector in the format of [y x]
% patchsize(height,width): the size of the patch to crop around
% center_point

    patchsize_y_half = floor(height /2);
    patchsize_x_half = floor(width /2);


    if(mod(height,2)==0)
       y_max= center_point(1)+patchsize_y_half-1 ;
    else 
       y_max= center_point(1)+patchsize_y_half ;
    end 
    
    if(mod(width,2)==0)
       x_max= center_point(2)+patchsize_x_half-1 ;
    else 
       x_max= center_point(2)+patchsize_x_half ;
    end 
    
      
    cropped_img = imgs(center_point(1)-patchsize_y_half:y_max, ...
              center_point(2)-patchsize_x_half:x_max,:,:);


end

