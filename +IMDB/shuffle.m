function [ imdb ] = shuffle( imdb )
%SHUFFLE Shuffle the images in the imdb matrix
%   

 
ridx = randperm(length( imdb.images.id));
imdb.images.id =  imdb.images.id(1,ridx);
imdb.images.set = imdb.images.set(1,ridx);
imdb.images.data = imdb.images.data(:,:,:,ridx);
imdb.images.mask = imdb.images.mask(:,:,:,ridx);
imdb.images.filenames = imdb.images.filenames(ridx);
imdb.images.boundary_mask=imdb.images.boundary_mask(:,:,ridx);

end

