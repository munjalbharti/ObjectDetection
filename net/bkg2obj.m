function [pos_y,pos_x,pos_z]= bkg2obj(predictions,mask)

         u_mask=squeeze(mask(:,:,[1],:));
         pos_x= repmat([1:size(predictions,2)],size(predictions,1),1,size(predictions,4));
         pos_y= repmat([1:size(predictions,1)]',1,size(predictions,2),size(predictions,4));
      
         offsets_x = squeeze(predictions(:,:,1,:)) .* 256;
         offsets_y = squeeze(predictions(:,:,2,:)) .* 256;
         
       
         end_pos_x=  round(offsets_x + pos_x);
         end_pos_y=  round(offsets_y  + pos_y);
       
         %end_pos_x=  round(offsets_x + pos_x);
         %end_pos_y=  round(offsets_y  + pos_y);
       
         
         end_pos_x(end_pos_x < 1)=1;
         end_pos_x(end_pos_x > size(predictions,2))=size(predictions,2);
         end_pos_y(end_pos_y < 1)=1;
         end_pos_y(end_pos_y > size(predictions,1))=size(predictions,1);
         
       
         
         end_pos_z= cumsum(ones(size(predictions,1),size(predictions,2),size(predictions,4)),3);

         
         s=[size(predictions,1),size(predictions,2),size(predictions,4)];
         ind=sub2ind(s,end_pos_y(:),end_pos_x(:),end_pos_z(:));
         
         pos= find(u_mask(ind)==1);
         %Only background
         [pos_y,pos_x,pos_z]=ind2sub(s,pos(u_mask(pos)==0));
         
         
         %[pos_y,pos_x,pos_z]=ind2sub(s,pos(u_mask(pos)==0));
         %[y_sel,x_sel,z_sel] = ind2sub(s,ind(find(u_mask(ind)==1)));
         
         %[pos_y,pos_x,pos_z]=find(ismember(end_pos_x,x_sel) & ismember(end_pos_y,y_sel) & ismember(end_pos_z,z_sel));
         
         
         
         %
         %
         %f_pos_y=[];
         %f_pos_x=[];
         %f_pos_z=[];
         
        % for k=1:size(pos_y,1)
         %   f_pos_y=[f_pos_y;pos_y(k);pos_y(k)];
         %   f_pos_x=[f_pos_x;pos_x(k);pos_x(k)];
         %   f_pos_z=[f_pos_z;(2*pos_z(k)-1);(2*pos_z(k)-1)+1];
            
         
        % end 
         
        
        
         
        % [pos_y,pos_x,pos_z]=find(ismember(end_pos_x,x_sel) & ismember(end_pos_y,y_sel) & ismember(end_pos_z,z_sel));
         %pos=find(end_pos_x== x_sel & end_pos_y==y_sel & end_pos_z==z_sel);
    end 