function [ y ] = getBatchAugmented( imdb,opts, batch )
%GETBATCHAUGMENTED
%Pick a batch of images from the imdb struct
%Applying random transformations to the images and masks


    images = imdb.images.data(:,:,:,batch);
    mask=imdb.images.mask(:,:,:,batch);
    
    labels=mask(:,:,[3],:);
     
    
    if(imdb.images.set(batch)==1)  %only for training ..not for validation
        [images,labels]=random_transform(images,labels);
    else 
      %  fprintf('batch is from vali set, not adding augmentations!');
    end 
    mask(:,:,[3],:)= labels;
   
    images = single(images);
   
   
  % labels = get_labeled_mask(labels);
   mask=single(mask);
   
    numGpus = numel(opts.gpus) ;
    if(numGpus >= 1)
      images= gpuArray(images);
    %  labels= gpuArray(labels);      
    end 
 
    y = {'data', images, 'label', mask} ;
end




