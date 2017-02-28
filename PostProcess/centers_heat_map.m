function [ r_center_mask,r_center_mask_pos ] = centers_heat_map( offset, bin_size )
%CENTERS_HEAT_MAP Summary of this function goes here
%   Detailed explanation goes here

r_center_mask=zeros(size(offset,1)/bin_size(1),size(offset,2)/bin_size(2));
r_center_mask_pos(1:size(offset,1)/bin_size(1),1:size(offset,2)/bin_size(2)) = struct('x_pos',[],'y_pos',[]);

for y=1:size(offset,1)
    for x=1:size(offset,2)
          
       if(isnan(offset(y,x,1)) || isnan(offset(y,x,2)))
            continue ;
       end 
        
       x1= round(x + offset(y,x,1));
       y1= round(y + offset(y,x,2));
       
       if(x1==x && y1==y)
        continue ;
       end 
        
       
       %or can add to the last position
       if( x1 < 1 || y1 < 1 || x1 > size(offset,2) || y1 > size(offset,1))
           continue;
       end
       
       
            c_m_y1=ceil(y1/bin_size(1));
            c_m_x1=ceil(x1/bin_size(2));
            
        
           
           % if( r_center_mask(c_m_y1,c_m_x1)==0)
            %        r_center_mask(c_m_y1,c_m_x1)=1;
             %       r_center_mask_pos(c_m_y1,c_m_x1).x_pos=[x];
              %      r_center_mask_pos(c_m_y1,c_m_x1).y_pos=[y];
            %else
                    r_center_mask(c_m_y1,c_m_x1)= r_center_mask(c_m_y1,c_m_x1)+1;
                    r_center_mask_pos(c_m_y1,c_m_x1).x_pos=[r_center_mask_pos(c_m_y1,c_m_x1).x_pos;x];
                    r_center_mask_pos(c_m_y1,c_m_x1).y_pos=[r_center_mask_pos(c_m_y1,c_m_x1).y_pos;y];
           % end
           
          
        %{   
           c_m_y=y1;
           c_m_x=x1;
           
           if(center_mask(c_m_y,c_m_x) == 0)
                center_mask(c_m_y,c_m_x)=1;            
                center_mask_pos(c_m_y,c_m_x).x_pos=[x];
                center_mask_pos(c_m_y,c_m_x).y_pos=[y];          
           else
               center_mask(c_m_y,c_m_x)=center_mask(c_m_y,c_m_x)+1 ;           
               center_mask_pos(c_m_y,c_m_x).x_pos=[center_mask_pos(c_m_y,c_m_x).x_pos;x];
               center_mask_pos(c_m_y,c_m_x).y_pos=[center_mask_pos(c_m_y,c_m_x).y_pos;y];             
           end         
           %}
        
       
        
    end 
end 

end



