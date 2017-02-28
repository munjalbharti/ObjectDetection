 
%% ------------------------------------------------------------------------
function inputs = getCOCOBatch3(imdb,opts, batch)
% -------------------------------------------------------------------------
 %Loads offsets from memory as int16 so no need to add constant
 % size and labels together as uint8, must add +1 to siye values
 % (0-255)=>(1-256)
 
opts.imageSize = [256,256];
 
filenames=imdb.images.filenames(batch);
lb_size_filenames=imdb.images.label_size_filenames(batch);
 
%cmap = labelColors(21) ;
 
file_n_labels=cell(1,2*length(batch));
 
 if(opts.prefetch)
      t0=tic;
      vl_imreadjpeg( {filenames{:},lb_size_filenames{:}},'Prefetch');
      timePrefetch=toc(t0);
      disp(['Prefetch In Dur:', num2str(timePrefetch)]);
 else
     t1=tic;
     file_n_labels= vl_imreadjpeg({filenames{:},lb_size_filenames{:}});
     timeElapsed = toc(t1);
     disp(['Fetch In Dur:', num2str(timeElapsed)]);

     t2=tic;
     images_orig= file_n_labels(1:length(batch)); %will return uint8 values
     labels_size_orig= file_n_labels(length(batch)+1:2*length(batch)); 
     [offsets_orig,labels_orig]= cellfun(@(x1,x2) preprocess_offset(x1,x2) , imdb.images.offsets(batch),labels_size_orig, 'UniformOutput', false);
     timeElapsedPost1 = toc(t2);
     disp(['Post1 In Dur:', num2str(timeElapsedPost1)]);
     
     t3=tic;
     sizes = cellfun(@size, images_orig, 'UniformOutput', false);
     cropSize = cellfun(@(x) [randi(x(1)-opts.imageSize(1)+1), randi(x(2)-opts.imageSize(2)+1)], sizes, 'UniformOutput', false);
 
     %apply the same crops to all frames
     images = cellfun(@(x,y) x(y(1):y(1)+opts.imageSize(1)-1, y(2):y(2)+opts.imageSize(2)-1, :), images_orig, cropSize, 'UniformOutput', false);
     images = single(cat(4, images{:}));
     labels = cellfun(@(x,y) x(y(1):y(1)+opts.imageSize(1)-1, y(2):y(2)+opts.imageSize(2)-1, :), labels_orig, cropSize, 'UniformOutput', false);
     labels = double(cat(4, labels{:}));
     offsets = cellfun(@(x,y) x(y(1):y(1)+opts.imageSize(1)-1, y(2):y(2)+opts.imageSize(2)-1, :), offsets_orig, cropSize, 'UniformOutput', false);
     offsets = double(cat(4, offsets{:}));
 
     %normalize
     offsets(:,:,[1,3],:) = offsets(:,:,[1,3],:) / opts.imageSize(1);
     offsets(:,:,[2,4],:) = offsets(:,:,[2,4],:) / opts.imageSize(2);
 
 
     numGpus = numel(opts.gpus) ;
     if(numGpus >= 1)
      images= gpuArray(images);      
     end
 
     inputs = {'data', images, 'label', cat(3,labels,offsets)}; 
     timeElapsedPost2=toc(t3); 
     disp(['Post2 In Dur:', num2str(timeElapsedPost2)]);
  end
 end
 
 
function [out,mask]=preprocess_offset(in_o,in_mask_size)
      %offsets are int16, no need to add constant
      % size are  uint8 add 1
      %does not has NaN value
       out=cat(3,single(in_o),single(in_mask_size(:,:,2))+1,single(in_mask_size(:,:,3))+1);
       mask = in_mask_size(:,:,1);
       m_c= cat(3, mask, mask, mask, mask);
       %background is 1
       out(m_c==1)=NaN;
       %offsets are assumed to containing negative offsets already
       
end 
 
 
  %{
    I= imread(fullfile(imdb.paths.image,pre_imgs,filenames{k}));
    if(size(I,3)==1)
        images_orig{k}=repmat(I,[1,1,3]);
    else 
        images_orig{k}=I;
    end 
 
    lb= imread(fullfile(imdb.paths.image,pre_labels,filenames{k}))  ;
    labels_orig{k}= rgb2ind(lb,cmap)+1; 
    oo=load(fullfile(imdb.paths.image,pre_offsets,sprintf('%s.mat',filenames{k}(1:end-4))));
     %}
