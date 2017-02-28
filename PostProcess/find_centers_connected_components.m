function[centers_y,centers_x,contri]= find_centers_connected_components(offset,threshold)

       

        %generate heat map
        [center_heat_map,center_contri]=centers_heat_map( offset, [1,1] );

        f1=figure;
        imagesc(center_heat_map);
        colorbar;
        
        %thresholding
        th_center_mask=center_heat_map;
        th_center_mask(find(th_center_mask <= threshold)) = 0;

        f2=figure;
        imagesc(th_center_mask);
        colorbar;
        
        %Find maximas in connected components 
        cc = bwconncomp(th_center_mask);
        stats = regionprops(cc,th_center_mask, 'MaxIntensity'); 

        centers_y=[];
        centers_x=[];

        %contributing pixels to this maxima.
        contri=struct('x_pos',[],'y_pos',[]);
        one_max_per_cc=true ;
         
        for i=1:cc.NumObjects
             contri(i) = struct('x_pos',[],'y_pos',[]);
             if(~one_max_per_cc)
                 %Can give multiple maximas in same conencted component
                 [centers_y,centers_x] = find(imregionalmax(th_center_mask));
             else
                  %TO get one maxima in one caonnected comonent 
     
                  pixels=cc.PixelIdxList{i};           
                  [max_pos]=find(th_center_mask(pixels)== stats(i).MaxIntensity);
            
                  %There can be multiple maximas in same connected component..taking only one
                  pos= pixels(max_pos(1,1));
                  [y,x] = ind2sub([256,256],pos);
            
                  %These maximas are the returned centers
                  centers_y=[centers_y;y];
                  centers_x=[centers_x;x];    
                 
                 
             end
                         
            for k=1:length(pixels)
                     [y,x] = ind2sub([256,256],pixels(k,1));
                      contri(i).x_pos=[contri(i).x_pos;center_contri(y,x).x_pos];
                      contri(i).y_pos=[contri(i).y_pos;center_contri(y,x).y_pos];   
            end 
            
       
         
        end 


        close(f1);
        close(f2);
end 





%r_center_mask1=zeros(size(offset,1)/bin_size(1),size(offset,2)/bin_size(2));
%for m=1:size(offset,1)/bin_size(1)
 %   for n=1:size(offset,2)/bin_size(2)
  %     val= sum(sum(center_mask((m-1)*bin_size(1)+1:m*bin_size(1),(n-1)*bin_size(2)+1:n*bin_size(2))));
   %    r_center_mask1(m,n)= val ;
  %  end 
%end 


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

