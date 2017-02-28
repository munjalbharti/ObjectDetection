function [  ] = display_image_and_mask( imdb, count_array )
%DISPLAY_IMAGE_AND_MASK Summary of this function goes here

for count=count_array;
%   Detailed explanation goes here
        image=imdb.images.data(:,:,:,count);
        mask=imdb.images.mask(:,:,:,count);
        filename= imdb.images.filenames{count};
        [ind_seg,map]=imread(fullfile(imdb.paths.orig_segmentation,filename));
        orig_segm=ind2rgb(ind_seg,map);


        figure ;
        subplot(1,3,1)
        imshow(image);
        title('Image');

        subplot(1,3,2)
        imshow(orig_segm);
        title('Original segmentation');

        subplot(1,3,3)
        imshow(get_mask(mask,false));
        title('Mask');
end 

end

