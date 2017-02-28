function [ min_x,min_y, widths,heights] = find_gt_boxes( offset )
%FIND_GT_BOXES Summary of this function goes here
%   Detailed explanation goes here

[ r_center_mask,r_center_mask_pos ] = centers_heat_map( offset, [1,1] );
 f1=figure;
 imagesc(r_center_mask);
 colorbar;
  
  
[centers_y,centers_x]=find(r_center_mask);
 contri = struct('x_pos',[],'y_pos',[]);  
 
 widths =[];
 heights=[];
 min_x=[];
 min_y=[];
  
  for k=1:length(centers_y)

        contri(k).x_pos= r_center_mask_pos(centers_y(k),centers_x(k)).x_pos ;
        contri(k).y_pos= r_center_mask_pos(centers_y(k),centers_x(k)).y_pos ;
       
        max_cx =  max(contri(k).x_pos);
        min_cx =  min(contri(k).x_pos);
               
                
        max_cy = max(contri(k).y_pos);
        min_cy = min(contri(k).y_pos);
                
        min_x = [min_x; min_cx];
        min_y = [min_y;min_cy];
         
        widths=[widths;(max_cx-min_cx)];
        heights=[heights;(max_cy-min_cy)];
          
  end

  close(f1);
end
