function [  ] = display_image_over_offset( image, offset )
%DISPLAY_IMAGE_OVER_OFFSET Summary of this function goes here
%   Detailed explanation goes here

figure;
imshow(uint8(image));
hold on ;
quiver([1:size(offset,2)],[1:size(offset,1)],offset(:,:,1),offset(:,:,2));

end

