
in_dir='F:\Bharti\Thesis\data';

o=load(fullfile(in_dir,'rgb_object_detection-15-person-offset-1-imdb.mat'));
opts.target_file=fullfile(in_dir,'rgb_object_detection-15-person-offset-1-imdb-bkg-random.mat');
imdb=o.imdb ;

mask=imdb.images.mask;
offsets=mask(:,:,[1,2],:);

for j=1:size(offsets,4)
    fprintf('Setting background for offset %d\n',j);
    offset= offsets(:,:,:,j);
 
    offset_org=offset;
   
    [ys,xs] =find(offset(:,:,1) == single(0.1));
 
    for k=1:length(ys)
        y=ys(k);
        x=xs(k);
        
        offset_x= randi([-25,25],1,1)-1; 
        offset_y= randi([-25,25],1,1)-1; 
        
        pointed_y=y+offset_y;
        pointed_x=x+offset_x;
            
        while( pointed_y > size(offset,1) || pointed_y < 1 || pointed_x < 1 || pointed_x > size(offset,2) || offset_org(pointed_y,pointed_x) ~= single(0.1) )
              % if( pointed_y <= size(offset,1) && pointed_y >= 1 && pointed_x >= 1 && pointed_x <= size(offset,2) && offset_org(pointed_y,pointed_x) ~= single(0.1) )
              %  disp('here');
              % end 
                offset_x= randi([-25,25],1,1)-1; 
                offset_y= randi([-25,25],1,1)-1; 
                
                pointed_y=y+offset_y;
                pointed_x=x+offset_x;
            
        end
             
        offset(y,x,1)=offset_x;
        offset(y,x,2)=offset_y;
             
      
    end 
    
    
    
    
    
    
 %{
vectors_set=false;

 
 while(~vectors_set)
            val1= randi([-25,25],length(pixels_unset_y),1)-1; 
            val2= randi([-25,25],length(pixels_unset_y),1)-1; 
            
            offset_x_pos= find(pixels_unset_z==1);
            offset_y_pos= find(pixels_unset_z==2);
            
            [IND]=sub2ind([256,256,2],pixels_unset_y(offset_x_pos),pixels_unset_x(offset_x_pos), pixels_unset_z(offset_x_pos));           
            offset(IND)= val1;
            
            [IND]=sub2ind([256,256,2],pixels_unset_y(offset_y_pos),pixels_unset_x(offset_y_pos),pixels_unset_z(offset_y_pos));           
            offset(IND)= val2;
           
           
            pointed_pixel_y=pixels_unset_y(offset_y_pos)+val2;
            pointed_pixel_x=pixels_unset_x(offset_x_pos)+val1;
                       
            IND = sub2ind([256,256],pointed_pixel_y,pointed_pixel_x); 
            
            pos=find(offset_org(IND) ~= single(0.1));
           
            pixels_unset_y=pixels_unset_y(pos);
            pixels_unset_x=pixels_unset_x(pos);
            pixels_unset_z=pixels_unset_z(pos);
           
            if(length(pixels_unset) == 0)
                vectors_set=true;
            end 
 end 
 
%}
   offsets(:,:,:,j)=offset;
   %figure;
   %quiver([1:size(offset,1)],[1:size(offset,2)],offset(:,:,1),offset(:,:,2));
   %set(gca,'Ydir','reverse')
end     

mask(:,:,[1,2],:)= offsets;
imdb.images.mask = mask ;
save(opts.target_file, 'imdb', '-v7.3');


