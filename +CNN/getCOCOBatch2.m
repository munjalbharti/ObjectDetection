
%% ------------------------------------------------------------------------
function inputs = getCOCOBatch2(imdb,opts, batch)
% -------------------------------------------------------------------------
%offsets are read as uint8 but with double the matrix size (2 bytes),  need
%to restore 16 bit value from two 8 bit values
% size and labels together as uint8, must add +1 to size values
% (0-255)=>(1-256)
%uses prefecth and vl_imreadjpg

opts.imageSize = [256,256];
filenames=imdb.images.filenames(batch);

offset_x_filenames=imdb.images.offset_x_filenames(batch);
offset_y_filenames=imdb.images.offset_y_filenames(batch);
lb_size_filenames=imdb.images.label_size_filenames(batch);


 cmap = labelColors(21) ;
 %full_filenames = cellfun(@(x) fullfile(imdb.paths.image,pre_imgs,x), filenames, 'UniformOutput', false);
 %full_labels = cellfun(@(x) fullfile(imdb.paths.image,pre_imgs,x), filenames, 'UniformOutput', false);
 %full_offsets=cellfunc(@x fullfile(imdb.paths.image,pre_offsets,sprintf('%s.mat',filenames{k}(1:end-4))),);
 
file_n_labels=cell(1,4*length(batch));

 if(opts.prefetch)
     % t0=tic;
      vl_imreadjpeg( {filenames{:},lb_size_filenames{:}, offset_x_filenames{:},offset_y_filenames{:}},'Prefetch');
     % timePrefetch = toc(t0);
     % disp(['Prefetch In Dur:', num2str(timePrefetch)]);

 else
  % t1=tic;
   file_n_labels= vl_imreadjpeg( {filenames{:},lb_size_filenames{:}, offset_x_filenames{:},offset_y_filenames{:}});
  % timeElapsed = toc(t1);
  % disp(['Fetch In Dur:', num2str(timeElapsed)]);
  

   % t2=tic;
    images_orig= file_n_labels(1:length(batch));
    labels_size_orig= file_n_labels(length(batch)+1:2*length(batch));
    offset_x_orig=file_n_labels(2*length(batch)+1:3*length(batch));
    offset_y_orig=file_n_labels(3*length(batch)+1:4*length(batch));

    [offsets_orig,labels_orig]= cellfun(@(x1,x2,x3) preprocess_offset(x1,x2,x3) , offset_x_orig,offset_y_orig,labels_size_orig, 'UniformOutput', false);

    %timeElapsedPost1 = toc(t2);
   % disp(['Post1 In Dur:', num2str(timeElapsedPost1)]);

    %t3=tic;
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
     %timeElapsedPost2=toc(t3); 
     %disp(['Post2 In Dur:', num2str(timeElapsedPost2)]);
   end
 end

function [I]=add_channels(I)
           if(size(I,3)==1)
               I=repmat(I,[1,1,3]);
           end 
end 

function [out,mask]=preprocess_offset(in_ox,in_oy,in_mask_size)

      %offsets are read as two bytes of uint8..convert them to
      %uint16..subtract the constant
      %sizes are read with labels, add +1
       ox= single(in_ox(:,1:size(in_ox,2)/2)) .* 256 +  single(in_ox(:,(size(in_ox,2)/2)+1:end )) ;
       oy= single(in_oy(:,1:size(in_oy,2)/2)) .* 256 +  single(in_oy(:,(size(in_oy,2)/2)+1:end )) ;
        
       opts.constant=2000;
       out=cat(3,ox-opts.constant,oy-opts.constant,single(in_mask_size(:,:,2))+1,single(in_mask_size(:,:,3))+1);
       
       mask=in_mask_size(:,:,1);
       m_c= cat(3, mask, mask, mask, mask);
       %background is 1
       out(m_c==1)=NaN;
       
      
       %out(:,:,1)=out(:,:,1)-opts.constant;
       %out(:,:,2)=out(:,:,2)-opts.constant;
       %out(:,:,1)=out(:,:,1)+1;
       %out(:,:,2)=out(:,:,2)+1;
       
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