in_dir='F:\Bharti\Thesis\data';
o=load(fullfile(in_dir,'rgb_object_detection-15-person-offset-4794.imdb-again-offset-saved.mat'));
opts.target_file=fullfile(in_dir,'rgb_object_detection-15-person-offset-4794.imdb-again-offset-saved-bkg.mat');
imdb=o.imdb ;
uniq_val=0.5;
offsets=imdb.images.mask;

for j=1:size(offsets,4)
  offset= offsets(:,:,:,j);
 [y,x] =find(offset(:,:,1) == single(0.1));
 for m=1:length(y)
        rem_y=mod(y(m),2);
        rem_x=mod(x(m),2);
        if(rem_y == 0 && rem_x==0)
            k_x= -uniq_val;
            k_y= 0;

        elseif (rem_y == 0 && rem_x ~= 0)
            k_x= 0;
            k_y= -uniq_val;
        elseif (rem_y ~=0 && rem_x==0)
            k_x= 0;
            k_y= uniq_val;
        else
            k_x= uniq_val;
            k_y= 0;
        end 

        offset(y(m),x(m),1)= k_x;
        offset(y(m),x(m),2)= k_y; 
   
       
 end 

   offsets(:,:,:,j)=offset;
   %figure;
   %quiver([1:size(offset,1)],[1:size(offset,2)],offset(:,:,1),offset(:,:,2));
   %set(gca,'Ydir','reverse')
end     
imdb.images.mask=offsets;
save(opts.target_file, 'imdb', '-v7.3');