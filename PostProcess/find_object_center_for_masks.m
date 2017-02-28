function [ th_centers_y,th_centers_x,th_center_mask_pos ] = find_object_center_for_masks( offset,bin_size,threshold, object_mask )
%FIND_OBJECT_CENTER_FOR_MASKS Summary of this function goes here
%   Detailed explanation goes here

    [r_center_mask,r_center_mask_pos]=centers_heat_map( offset, bin_size );

    f1=figure;
    imagesc(r_center_mask);
    colorbar;
    
    mask_threshold = bin_size(1)*bin_size(2)+threshold;
    th_center_mask=r_center_mask;
    th_center_mask(find(th_center_mask <= mask_threshold)) = 0;
    
    
    cc = bwconncomp(th_center_mask);
    stats = regionprops(cc,th_center_mask, 'MaxIntensity'); 

    th_centers_y=[];
    th_centers_x=[];

    th_center_mask_pos=struct('x_pos',[],'y_pos',[]);
    for i=1:cc.NumObjects
        th_center_mask_pos(i) = struct('x_pos',[],'y_pos',[]);

        m_value=stats(i).MaxIntensity ;
        pixels=cc.PixelIdxList{i};
        [max_pos]=find(th_center_mask(pixels)== m_value);
        pos= pixels(max_pos(1,1));
        [I,J] = ind2sub([256,256],pos);
        th_centers_y=[th_centers_y;I];
        th_centers_x=[th_centers_x;J];
   
   
        %pos_struct=r_center_mask_pos(pixels) ;
         for k=1:length(pixels)
                 [y,x] = ind2sub([256,256],pixels(k,1));
        
                 x1=r_center_mask_pos(y,x).x_pos ;
                 y1=r_center_mask_pos(y,x).y_pos ;        
                 th_center_mask_pos(i).x_pos=[th_center_mask_pos(i).x_pos;x1];
                 th_center_mask_pos(i).y_pos=[th_center_mask_pos(i).y_pos;y1];   
         end 
       
    end 


close(f1);
end 



%r_center_mask1=zeros(size(offset,1)/bin_size(1),size(offset,2)/bin_size(2));
%for m=1:size(offset,1)/bin_size(1)
 %   for n=1:size(offset,2)/bin_size(2)
  %     val= sum(sum(center_mask((m-1)*bin_size(1)+1:m*bin_size(1),(n-1)*bin_size(2)+1:n*bin_size(2))));
   %    r_center_mask1(m,n)= val ;
  %  end 
%end 

%thresholding



%th_center_mask=zeros(size(r_center_mask));

%adaptive thresholding

%th_center_mask(find(th_center_mask > mask_threshold)) = 1;



%do non maximal supression
%win=ones(3,3);
%win(2,2)=0;
%tmp = th_center_mask > imdilate(th_center_mask, win);

%f2=figure;
%imagesc(tmp);
%impixelinfo;

%{
[th_centers_y, th_centers_x] = find(tmp);


th_center_mask_pos = struct('x_pos',[],'y_pos',[]);
for k=1:length(th_centers_y)
    c_y=th_centers_y(k);
    c_x=th_centers_x(k);
    
    th_center_mask_pos(k).x_pos= r_center_mask_pos(c_y,c_x).x_pos ;
    th_center_mask_pos(k).y_pos= r_center_mask_pos(c_y,c_x).y_pos ;
    
   min_x= min(th_center_mask_pos(k).x_pos);
   max_x= max(th_center_mask_pos(k).x_pos);
   min_y= min(th_center_mask_pos(k).y_pos);
   max_y= max(th_center_mask_pos(k).y_pos);
    
   th_centers_y(k)=min_y+ ceil((max_y-min_y)/2);
   th_centers_x(k)=min_x+ ceil((max_x-min_x)/2);
end 

%}
%Returning centers as if bin_size is [1,1]
%th_centers_y=(th_centers_y-1) .* bin_size(1)+1 + bin_size(1)/2;
%th_centers_x=(th_centers_x-1) .* bin_size(2)+1 + bin_size(2)/2;

%close(f2);


