
function[centers_y,centers_x,votes]= find_centers_hough_voting_fast(offset,bin_size,threshold,non_m_win_siz)
 
         pos_x= repmat([1:size(offset,2)],size(offset,1),1,1);
         pos_y= repmat([1:size(offset,1)]',1,size(offset,2),1);
       
         end_pos_x=  round(offset(:,:,1) + pos_x);
         end_pos_y=  round(offset(:,:,2)  + pos_y);
         
         end_pos_x(end_pos_x < 1)=1;
         end_pos_x(end_pos_x > size(offset,2))=size(offset,2);
         end_pos_y(end_pos_y < 1)=1;
         end_pos_y(end_pos_y > size(offset,1))=size(offset,1);
         
         
        votes= [end_pos_x(:),end_pos_y(:)];
         %r_center_mask_full=accumarray(end_pos_full,1);
                
         end_pos=[ceil(end_pos_y(:)/bin_size(2)), ceil(end_pos_x(:)/bin_size(1))];
         
         r_center_mask=accumarray(end_pos,1,[size(offset,1)/bin_size(1),size(offset,2)/bin_size(2)]);
         

          %f2=figure;
          %imagesc(r_center_mask);
          %colorbar;

          %thresholding
          mask_threshold = bin_size(1)*bin_size(2)+threshold;
          th_center_mask=r_center_mask;
          th_center_mask(find(th_center_mask <= mask_threshold)) = 0;

          %do non maximal supression
          sup_size=[non_m_win_siz,non_m_win_siz];
          win=ones(sup_size(1),sup_size(2));
          mid=ceil(non_m_win_siz/2);
          win(mid,mid)=0;
          tmp = th_center_mask > imdilate(th_center_mask, win);

         % f3=figure;
         % imagesc(tmp);
         % impixelinfo;
 
 
          [th_centers_y, th_centers_x] = find(tmp);    
         
           %Retrned centers %Returning centers as if bin_size is [1,1]
         
          centers_y=(th_centers_y-1) .* bin_size(1)+1 + bin_size(1)/2;
          centers_x=(th_centers_x-1) .* bin_size(2)+1 + bin_size(2)/2;
          
          %Update votes
          for m=1:length(th_centers_y)
                all_centers_y=(th_centers_y(m)-1) .* bin_size(1) + [1:bin_size(1)];
                all_centers_x=(th_centers_x(m)-1) .* bin_size(2) + [1:bin_size(2)];
                all_centers_y=all_centers_y(:);
                all_centers_x=all_centers_x(:);
                c = ismember(votes(:,1), all_centers_x) & ismember(votes(:,2), all_centers_y);
                % Extract the elements of a at those indexes.
                ind = find(c);
                votes(ind,1)=centers_x(m);
                votes(ind,2)=centers_y(m);                  
         end
              
           
          
        
        
        % all_centers_y= (repmat(th_centers_y,1,bin_size(1)) + repmat([1:bin_size(1)],length(th_centers_y),1))';
        % all_centers_x= (repmat(th_centers_x,1,bin_size(1)) + repmat([1:bin_size(1)],length(th_centers_x),1))';
 
         
         
          %centers_y = all_centers_y(:,(bin_size(1)/2)+1);
          %centers_x = all_centers_x(:,(bin_size(2)/2)+1);
                    
        
                    
         % close(f2);
         % close(f3);
 

end 
 
 
 
 

