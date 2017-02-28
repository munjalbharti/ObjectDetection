function [y] = getBatch(imdb,opts, batch)
%GETBATCHLOADED Pick a batch of images from the imdb struct

    images = imdb.images.data(:,:,:,batch);
    labels=imdb.images.mask(:,:,:,batch);
    
    images = single(images);
   
   % mean=imdb.images.dataMean;
   % S=imdb.images.dataStd ;
   % images = bsxfun(@rdivide,bsxfun(@minus,images,mean),S);
    
    
   %masks contains 0 for background and 255 for foreground..however
   %matconvnet loss function accepts ground truths labels starting from 1
   %[1,2,3,...]
  
   labels = get_labeled_mask(labels);
   labels=single(labels);
   
    numGpus = numel(opts.gpus) ;
    if(numGpus >= 1)
      images= gpuArray(images);      
    end 
    
    
    y = {'data', images, 'label', labels} ;
end



