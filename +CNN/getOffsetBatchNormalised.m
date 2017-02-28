function [ y ] = getOffsetBatchNormalised( imdb,opts, batch )
%GETOFFSETBATCH Summary of this function goes here
%   Detailed explanation goes here

    images = imdb.images.data(:,:,:,batch);
    labels=imdb.images.mask(:,:,:,batch);
    
    images = single(images);
   
   % mean=imdb.images.dataMean;
   % S=imdb.images.dataStd ;
   % images = bsxfun(@rdivide,bsxfun(@minus,images,mean),S);
    
    
   %labels=imresize(labels,[128,128],'nearest');
   labels=single(labels);
  
   labels(labels == single(0.1))= NaN ;
  
    %normalise
   max_v= max(max(abs(labels))); %is a vector
   labels=bsxfun(@rdivide, labels,  max_v);
   
   
  
    numGpus = numel(opts.gpus) ;
    if(numGpus >= 1)
      images= gpuArray(images);      
    end 
    
    
    y = {'data', images, 'label', labels} ;

end


