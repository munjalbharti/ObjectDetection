function [ augmented_imgs, augmented_masks ] = random_transform_offset( orig_images,orig_masks )
%RANDOM_TRANSFORM_OFFSET Summary of this function goes here
%Applies random transformations to set of images.
%Currently applies 3 transformations one after another <Rot,Scale, flip with chance>%

   padding_size=[ceil(size(orig_images,1)/2),ceil(size(orig_images,2)/2)];
                       
   augmented_imgs=zeros(0,0,0,0);
   augmented_masks=zeros(0,0,0,0);
   I_padded = padarray(orig_images,padding_size ,0);
   mask_padded = padarray(orig_masks, padding_size,0.5);   %padding with 0.5
     
   for k=1:size(orig_images,4)
     
          rot=randi([-5,5],1,1);  %random value between -5 to 5    
          %rot=45;
          I_rot=imrotate(I_padded(:,:,:,k),rot,'nearest');        
          mask_rot=imrotate(mask_padded(:,:,:,k),rot,'nearest');
          mask_rot=rotate_vectors(mask_rot,rot);
       
          %SCALING
          scale = 1+0.5*rand; %possible scaling factors [1 to 1.5] 
          scaled_image=imresize(I_rot,scale,'nearest');
          scaled_mask=imresize(mask_rot ,scale,'nearest');
      
      
           if(rand > 0.5)
              scaled_image=fliplr(scaled_image);
              scaled_mask=fliplr(scaled_mask);
              scaled_mask(:,:,1)=-scaled_mask(:,:,1); %Change x to -x
           end 
         
          
        center_point_img=[floor(size(scaled_image,1)/2)+1, floor(size(scaled_image,2)/2)+1];
        [image]=crop_at_point(scaled_image,center_point_img,size(orig_images,1),size(orig_images,2));
        
        
        center_point_mask=[floor(size(scaled_mask,1)/2)+1, floor(size(scaled_mask,2)/2)+1];       
        [mask]=crop_at_point(scaled_mask,center_point_mask,size(orig_masks,1),size(orig_masks,2));
     
       
        augmented_imgs(:,:,:,k) = image;
        augmented_masks(:,:,:,k) = mask;
  end 
      


end

