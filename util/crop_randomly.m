function [ cropped_image,cropped_mask] = crop_randomly(imgs,masks,h,w )

img_h=size(imgs,1);
img_w=size(imgs,2);

%x_min_range=[1:img_w - w+1]:
%y_min_range=[1:img_h-h+1];

x_min=randperm(img_w-w+1,1); % generate 1 random number between 1:img_w - w+1
y_min=randperm(img_h-h+1,1);

cropped_image=imgs(y_min:y_min+h-1,x_min:x_min+w-1,:,:);
cropped_mask=masks(y_min:y_min+h-1,x_min:x_min+w-1,:,:);


%cropped_image=imcrop(img,[1,1,w-1,h-1]);


end

