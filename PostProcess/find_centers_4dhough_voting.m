function[centers_y,centers_x,widths,heights,contri]= find_centers_4dhough_voting(offset,bbox_size,bin_size,size_bin_size,threshold,sup_size,size_non_m_win_size)

 tStart=tic ;
%generate heat map
  h=ceil(size(offset,1)/bin_size(1));
  w=ceil(size(offset,2)/bin_size(2));
  w1=ceil(size(offset,2)/size_bin_size(2));
  h1=ceil(size(offset,1)/size_bin_size(1));
  
r_center_mask=zeros(h,w,w1,h1);
r_center_mask_pos(1:h,1:w,1:w1,1:h1) = struct('x_pos',[],'y_pos',[]);

for y=1:size(offset,1)
    for x=1:size(offset,2)
                
       x1= round(x + offset(y,x,1));
       y1= round(y + offset(y,x,2));
       
       width = round(bbox_size(y,x,1));
       height= round(bbox_size(y,x,2));
       
        
       
      % if(width <=0 || height <= 0 || width > size(offset,2) || height > size(offset,1))
       %     continue; 
      % end 
  
      
       if(width <=0 || height <= 0)
            continue; 
       end 
       
       if( width > size(offset,2) )
           width=  size(offset,2);
       end 
       
       if(height > size(offset,1))
           height=  size(offset,1);
       end 
       
      if(x1==x && y1==y)
        continue ;
       end 
        
       
       %or can add to the last position
       if( x1 < 1 || y1 < 1 || x1 > size(offset,2) || y1 > size(offset,1))
           continue;
       end
       
         
         c_m_y1=ceil(y1/bin_size(1));
         c_m_x1=ceil(x1/bin_size(2));
         c_m_z1=ceil(width/size_bin_size(2));
         c_m_z2=ceil(height/size_bin_size(1));
            
           % if(c_m_y1==42 && c_m_x1==30)
           %     disp('ee');
           % end 
          
    
 
            r_center_mask(c_m_y1,c_m_x1,c_m_z1,c_m_z2)= r_center_mask(c_m_y1,c_m_x1,c_m_z1,c_m_z2)+1;
            r_center_mask_pos(c_m_y1,c_m_x1,c_m_z1,c_m_z2).x_pos=[r_center_mask_pos(c_m_y1,c_m_x1,c_m_z1,c_m_z2).x_pos;x];
            r_center_mask_pos(c_m_y1,c_m_x1,c_m_z1,c_m_z2).y_pos=[r_center_mask_pos(c_m_y1,c_m_x1,c_m_z1,c_m_z2).y_pos;y];
              
    end 
end



  %thresholding
  mask_threshold = threshold;
  th_center_mask=r_center_mask;
  th_center_mask(find(th_center_mask <= mask_threshold)) = 0;
  
  %do non maximal supression
  win_size_half=floor(sup_size/2);
  swin_size_half=floor(size_non_m_win_size/2);
  tmp=zeros(size(th_center_mask));
  heat_map=padarray(th_center_mask,[win_size_half,win_size_half,swin_size_half,swin_size_half]);

  tStart1=tic ;
  for k=1+win_size_half:size(heat_map,1)-win_size_half
   for l=1+win_size_half:size(heat_map,2)-win_size_half
      for m=1+swin_size_half:size(heat_map,3)-swin_size_half
        for n=1+swin_size_half:size(heat_map,4)-swin_size_half
            vals=heat_map(k-win_size_half:k+win_size_half,l-win_size_half:l+win_size_half,m-swin_size_half:m+swin_size_half,n-swin_size_half:n+swin_size_half);
            if(heat_map(k,l,m,n) ==  max(vals(:)) && heat_map(k,l,m,n) ~= 0);
                  tmp(k-win_size_half,l-win_size_half,m-swin_size_half,n-swin_size_half)=1;
             end 
        end 
      end 
   end
  end 
 
 tElapsed = toc(tStart1);
 fprintf('\nNon maximal supp %f sec for image %d X %d\n',tElapsed,size(offset,1),size(offset,2));
   
 
  %Retrned centers
  ind = find(tmp);
  [th_centers_y,th_centers_x,th_widths,th_heights]=ind2sub(size(tmp),ind);
 
  %Find Contributions for these centers
  contri = struct('x_pos',[],'y_pos',[]);  
  

  
  centers_y=(th_centers_y-1) .* bin_size(1)+1 + floor(bin_size(1)/2);
  centers_x=(th_centers_x-1) .* bin_size(2)+1 + floor(bin_size(2)/2);
  centers_y(centers_y > size(offset,1))=size(offset,1); 
  centers_x(centers_x > size(offset,2))=size(offset,2);
   
   
  heights=(th_heights-1) .* size_bin_size(1)+1 + floor(size_bin_size(1)/2);
  widths=(th_widths-1) .* size_bin_size(2)+1 + floor(size_bin_size(2)/2);
  heights(heights > size(offset,1))=size(offset,1); 
  widths(widths > size(offset,2))=size(offset,2);
          
  
  for k=1:length(th_centers_y)
        c_y=th_centers_y(k);
        c_x=th_centers_x(k);
        c_w=th_widths(k);
        c_h=th_heights(k);
 
        contri(k).x_pos= r_center_mask_pos(c_y,c_x,c_w,c_h).x_pos ;
        contri(k).y_pos= r_center_mask_pos(c_y,c_x,c_w,c_h).y_pos ;
          
  end 
   
   tElapsed = toc(tStart);
   fprintf('\n4d hough Voting %f sec for image %d X %d\n',tElapsed,size(offset,1),size(offset,2));
           
 % th_centers_z1=th_centers_z1/100;
 % th_centers_z2=th_centers_z2/100;

end 
 

