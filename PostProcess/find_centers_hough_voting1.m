function[centers_y,centers_x,contri]= find_centers_hough_voting1(offset,bin_size,threshold,non_m_win_siz)

 % tStart=tic ;
  %generate heat map
  h=ceil(size(offset,1)/bin_size(1));
  w=ceil(size(offset,2)/bin_size(2));
  
r_center_mask=zeros(h,w);
r_center_mask_pos(1:h,1:w) = struct('x_pos',[],'y_pos',[]);

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
    

            r_center_mask(c_m_y1,c_m_x1)= r_center_mask(c_m_y1,c_m_x1)+1;
            r_center_mask_pos(c_m_y1,c_m_x1).x_pos=[r_center_mask_pos(c_m_y1,c_m_x1).x_pos;x];
            r_center_mask_pos(c_m_y1,c_m_x1).y_pos=[r_center_mask_pos(c_m_y1,c_m_x1).y_pos;y];
              
    end 
end 

  %f2=figure;
  %imagesc(r_center_mask);
  %colorbar;

  %thresholding
  %mask_threshold = bin_size(1)*bin_size(2)+threshold;
  
  mask_threshold = threshold;
  th_center_mask=r_center_mask;
  th_center_mask(find(th_center_mask <= mask_threshold)) = 0;
  
  %do non maximal supression
  sup_size=[non_m_win_siz,non_m_win_siz];
  win=ones(sup_size(1),sup_size(2));
  mid=ceil(non_m_win_siz/2);
  win(mid,mid)=0;
  tmp = th_center_mask > imdilate(th_center_mask, win);

  %f3=figure;
  %imagesc(tmp);
  %impixelinfo;


  %Retrned centers
  [th_centers_y, th_centers_x] = find(tmp);

  %Find Contributions for these centers
  contri = struct('x_pos',[],'y_pos',[]);  
  
  %centers_y=[];
  %centers_x=[];
  
   %Retrned centers %Returning centers as if bin_size is [1,1]
   centers_y=(th_centers_y-1) .* bin_size(1)+1 + floor(bin_size(1)/2);
   centers_x=(th_centers_x-1) .* bin_size(2)+1 + floor(bin_size(2)/2);
          
  centers_y(centers_y > size(offset,1))=size(offset,1); 
  centers_x(centers_x > size(offset,2))=size(offset,2);
   
   for k=1:length(th_centers_y)
        c_y=th_centers_y(k);
        c_x=th_centers_x(k);
    
       
   
       %same as sup_size
        contri(k).x_pos= r_center_mask_pos(c_y,c_x).x_pos ;
        contri(k).y_pos= r_center_mask_pos(c_y,c_x).y_pos ;
          
        
    
       % min_x= min(contri(k).x_pos);
       % max_x= max(contri(k).x_pos);
       % min_y= min(contri(k).y_pos);
       % max_y= max(contri(k).y_pos);
    
       % centers_y(k)=min_y+ floor((max_y-min_y+1)/2);
       % centers_x(k)=min_x+ floor((max_x-min_x+1)/2);
        
        
         
         
   end 

  % tElapsed = toc(tStart);
  % fprintf('\nHough Voting %f sec for image %d X %d\n',tElapsed,size(offset,1),size(offset,2));
         
  %  close(f2);
  %  close(f3);

end 


%{
    center_mask=zeros(size(offset,1),size(offset,2));
    center_mask_pos = struct('x_pos',[],'y_pos',[]);
%}

%r_center_mask1=zeros(size(offset,1)/bin_size(1),size(offset,2)/bin_size(2));
%for m=1:size(offset,1)/bin_size(1)
 %   for n=1:size(offset,2)/bin_size(2)
  %     val= sum(sum(center_mask((m-1)*bin_size(1)+1:m*bin_size(1),(n-1)*bin_size(2)+1:n*bin_size(2))));
   %    r_center_mask1(m,n)= val ;
  %  end 
%end 








%Returning centers as if bin_size is [1,1]
%th_centers_y=(th_centers_y-1) .* bin_size(1)+1 + bin_size(1)/2;
%th_centers_x=(th_centers_x-1) .* bin_size(2)+1 + bin_size(2)/2;

