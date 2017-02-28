function [ mask ] = rotate_vectors( mask,rot )
%ROTATE_VECTORS Summary of this function goes here
%   Detailed explanation goes here
 R = [cosd(rot) -sind(rot); sind(rot) cosd(rot)];
          
            for m=1:size(mask,1)
              for n=1:size(mask,2)
               if(mask(m,n,1) ~= double(0.1) && mask(m,n,2) ~= double(0.1) )
                        vec=[mask(m,n,1); mask(m,n,2)];
                        %r_vec=imrotate(vec,90,'nearest');
                        r_vec = R*vec;
                        mask(m,n,1)=r_vec(1);
                        mask(m,n,2)=r_vec(2);
               else 
                 %  disp('test');
                end 
                
               end 
            end 

end



