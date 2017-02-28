function [ y ] = getOffsetBatchAugmented( imdb,opts, batch )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


    images = imdb.images.data(:,:,:,batch);
    labels=imdb.images.mask(:,:,:,batch);
    
    if(imdb.images.set(batch)==1)  %only for training ..not for validation
        [images,labels]=random_transform_offset(images,labels);
    else 
      %  fprintf('batch is from vali set, not adding augmentations!');
    end
    
    
    images = single(images);
    labels=single(labels);
   
   
   max_x= max(abs(labels(:,:,1)));
   max_y= max(abs(labels(:,:,2)));
   
   labels(labels(:,:,1) == single(0.5))= NaN ;
   labels(labels(:,:,2) == single(0.5))= NaN ;
   
   labels(:,:,1)=labels(:,:,1) ./ max_x ;
   labels(:,:,2)=labels(:,:,2) ./ max_y ;
   
    numGpus = numel(opts.gpus) ;
    if(numGpus >= 1)
      images= gpuArray(images);
    end 
 
    y = {'data', images, 'label', labels} ;
end




