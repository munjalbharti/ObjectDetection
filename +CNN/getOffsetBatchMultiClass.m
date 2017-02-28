function [ y ] = getOffsetBatchMultiClass( imdb,opts, batch  )
%GETOFFSETBATCHMULTICLASS Summary of this function goes here
%   Detailed explanation goes here
images=zeros(256,256,3,length(batch),'single');
labels=zeros(256,256,3,length(batch),'double');

for k=1:length(batch)
    ind=batch(k);
    o=single(imdb.offsets{ind});
    img=single(imdb.images{ind});
    mask=cat(3,o(:,:,[1,2]),single(imdb.labels{ind}));
    %crop
    [img,mk]=crop_randomly(img,mask,256,256);
    
    %flip
    if(rand > 0.5)
              img=fliplr(img);
              mk=fliplr(mk);
    end
     
   
    
    images(:,:,:,k)=img;
    labels(:,:,:,k)=mk;
end 
   


   % images = single(images);
   % labels=single(labels);
   
   % mean=imdb.images.dataMean;
   % S=imdb.images.dataStd ;
   % images = bsxfun(@rdivide,bsxfun(@minus,images,mean),S);
    
    

      
  
   %labels(labels == single(0.1))= NaN ;
   %labels(:,:,[1,2],:)= labels(:,:,[1,2],:)/256;
   
    numGpus = numel(opts.gpus) ;
    if(numGpus >= 1)
      images= gpuArray(images);      
    end 
    
    
    y = {'data', images, 'label', labels} ;
end

