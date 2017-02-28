function [ y ] = getOffsetBatchMultiClass2( imdb,opts, batch  )
%GETOFFSETBATCHMULTICLASS Summary of this function goes here
%   Detailed explanation goes here
images=zeros(256,256,3,length(batch),'single');
labels=zeros(256,256,5,length(batch),'double');

for k=1:length(batch)
    ind=batch(k);
    o=single(imdb.offsets{ind});
    
    img=single(imdb.images{ind});
    mk=cat(3,single(imdb.labels{ind}),o);
    
    if(imdb.set(batch)==1)
         %crop
         [img,mk]=crop_randomly(img,mk,256,256);
    
         %flip
         if(rand > 0.5)
             img=fliplr(img);
               
             %flip x component of offset vector
             mk(:,:,2)=-mk(:,:,2);
             mk=fliplr(mk);
        end
         
         
    else 
        center_point=[floor(size(img,1)/2)+1, floor(size(img,2)/2)+1];
        img=crop_at_point( img,center_point,256,256 );
        mk= crop_at_point( mk,center_point,256,256 );
        
     
        
       
     %   disp('here');
    end 
    % color jittering
    img=single(img);
   % img(:,:,1)=img(:,:,1)*((1.2 - 0.8)*rand + 0.8);
   % img(:,:,2)=img(:,:,2)*((1.2 - 0.8)*rand + 0.8);
   % img(:,:,3)=img(:,:,3)*((1.2 - 0.8)*rand + 0.8);
   % img=uint8(255*img/max(img(:)));
   
   % mk(:,:,[1,2])=mk(:,:,[1,2])/256;
    
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

