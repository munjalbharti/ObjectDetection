function [  ] = display_image_and_offset( image, offset)
%DISPLAY_MASK_AND_OFFSET Summary of this function goes here
%   Detailed explanation goes here

figure;
subplot(1,2,1);

%y is downwards
imshow(uint8(image));
%hold on ;
subplot(1,2,2);
%y is  upwards 

%quiver([1:size(offset,1)],sort([1:size(offset,2)],'descend'),offset(:,:,1),offset(:,:,2));
%
quiver([1:size(offset,2)],[1:size(offset,1)],offset(:,:,1),offset(:,:,2));
set (gca,'Ydir','reverse')

%hold off ;

end



