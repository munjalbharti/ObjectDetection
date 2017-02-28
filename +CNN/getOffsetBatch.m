function [ y ] = getOffsetBatch( imdb,opts, batch )
%GETOFFSETBATCH Summary of this function goes here
%   Detailed explanation goes here

    images = imdb.images.data(:,:,:,batch);
    labels=imdb.images.mask(:,:,:,batch);
    
    images = single(images);
   
   % mean=imdb.images.dataMean;
   % S=imdb.images.dataStd ;
   % images = bsxfun(@rdivide,bsxfun(@minus,images,mean),S);
    
    
   labels=single(labels);
      
   %%%%u can call here with imagesa and labels ...it will give u augmented images nd labels%%%
  
   labels(labels == single(0.1))= NaN ;
   %labels(:,:,[1,2],:)= labels(:,:,[1,2],:)/256;
   
    numGpus = numel(opts.gpus) ;
    if(numGpus >= 1)
      images= gpuArray(images);      
    end 
    
    
    %and then u pass here augmented images and labels 
    y = {'data', images, 'label', labels} ;

end


