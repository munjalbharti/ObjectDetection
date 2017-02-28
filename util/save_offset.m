function [  ] = save_offset(mask,offset,obj_rects,path,filename )
%SAVE_OFFSET Summary of this function goes here
%   Detailed explanation goes here
   
   f= figure;
   imshow(mask);
   hold on ;
   
   
   for k=1:length(obj_rects.x_mins)
       x_min=obj_rects.x_mins(k);
       y_min=obj_rects.y_mins(k);
       width=obj_rects.widths(k);
       height=obj_rects.heights(k);
       
     rectangle('Position',[x_min-1 y_min-1 width+1 height+1],'LineWidth',3,'EdgeColor','r'); 
    
     center_x  = x_min+floor((width-1)/2);
     center_y =  y_min+floor((height-1)/2);
     plot(center_x,center_y,'r.');
  
   end 
   
   quiver([1:size(offset,1)],[1:size(offset,2)],offset(:,:,1),offset(:,:,2));
         
    savefig(f,fullfile(path, sprintf('%s.fig',filename)))
    saveas(f,fullfile(path, sprintf('%s.png',filename)));
    save(fullfile(path, sprintf('%s.mat',filename)),'offset','obj_rects');
    close(f);

end

