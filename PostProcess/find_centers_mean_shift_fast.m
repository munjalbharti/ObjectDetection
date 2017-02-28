function [ centers_y,centers_x,votes ] = find_centers_mean_shift_fast( offset, threshold, bandwidth, iterations )
%FIND_CENTERS_MEAN_SHIFT Summary of this function goes here
%   Detailed explanation goes here

         pos_x= repmat([1:size(offset,2)],size(offset,1),1,1);
         pos_y= repmat([1:size(offset,1)]',1,size(offset,2),1);
       
         end_pos_x=  round(offset(:,:,1) + pos_x);
         end_pos_y=  round(offset(:,:,2)  + pos_y);
         
         end_pos_x(end_pos_x < 1)=1;
         end_pos_x(end_pos_x > size(offset,2))=size(offset,2);
         end_pos_y(end_pos_y < 1)=1;
         end_pos_y(end_pos_y > size(offset,1))=size(offset,1);
         
         
         votes= [end_pos_x(:),end_pos_y(:)];
     
         center_mask=accumarray([end_pos_y(:),end_pos_x(:)],1,[size(offset,1),size(offset,2)]);
      
       
     %   f1=figure;
     %   imagesc(center_mask);

        %thresholding
        th_center_mask=center_mask;
        th_center_mask(find(th_center_mask <= threshold)) = 0;

        [y,x,v]=find(th_center_mask);
        points=[];
        
       
        if(size(y,1)==0)
            centers_y=[];
            centers_x=[];
           % close(f1);
            return ;
        end 
        
        for k=1:size(y,1)
             points=[points; repmat([y(k),x(k)],[v(k),1] )  ];
        end

       %centers and points have same size
         
        centers=mean_shift(points,bandwidth,iterations);
        centers_count=accumarray(centers,1);
        th=50;
        [centers_y, centers_x]=find(centers_count > th);
   

        %update votes
        updated_votes=votes;
        for i=1:length(centers_y)
             y=centers_y(i);
             x=centers_x(i);
             
             contri_points=unique(points(find(centers(:,1)==y & centers(:,2)==x),:),'rows');
             all_centers_y = contri_points(:,1);
             all_centers_x = contri_points(:,2);
            
             c = ismember(votes(:,1), all_centers_x) & ismember(votes(:,2), all_centers_y);
            
             ind = find(c);
             updated_votes(ind,1)=x;
             updated_votes(ind,2)=y;
             
        end
          
   


       % close(f1);
end

