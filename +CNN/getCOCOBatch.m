
%% ------------------------------------------------------------------------
function inputs = getCOCOBatch(imdb,opts, batch)
% -------------------------------------------------------------------------

opts.imageSize = [256,256];


filenames=imdb.images.filenames(batch);
lb_filenames=imdb.images.label_filenames(batch);
offset_x_filenames=imdb.images.offset_x_filenames(batch);

% cmap = labelColors(21) ;
 
%s1=tic;

file_n_labels= vl_imreadjpeg( {filenames{:},lb_filenames{:}});
images_orig= file_n_labels(1:length(batch));
labels_orig= file_n_labels(length(batch)+1:2*length(batch));

%toc(s1);

% images_orig= cellfun(@(x) add_channels(x) , images_orig, 'UniformOutput', false);
%background will have label 1
 %labels_orig=cellfun(@(x) rgb2ind(uint8(x),cmap)+1,labels_orig,'UniformOutput', false);
 
 

%{
images_orig = vl_imreadjpeg(filenames,'NumThreads', 6);
 labels_orig = vl_imreadjpeg(lb_filenames,'NumThreads', 6);
 offset_x_orig = vl_imreadjpeg(offset_x_filenames,'NumThreads', 6);
 offset_y_orig = vl_imreadjpeg(offset_y_filenames,'NumThreads', 6);
 size_x_orig = vl_imreadjpeg(size_x_filenames,'NumThreads', 6);
 size_y_orig = vl_imreadjpeg(size_y_filenames,'NumThreads', 6);
 
%}

 
%s2=tic;
 parfor k=1:length(filenames) 
    offset_filename= sprintf('%s.mat',offset_x_filenames{k}(1:end-7));
    oo=load(offset_filename);
    offsets_orig{k}= cat(3,single(oo.offset_gt),single(oo.size_gt));
   % images_orig{k}=imread(filenames{k});
   % labels_orig{k} = imread(lb_filenames{k});
  
 end

offsets_orig= cellfun(@(offset,mask) preprocess_offset(offset,mask) , offsets_orig,labels_orig, 'UniformOutput', false);


%toc(s2);


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
 end

function [I]=add_channels(I)
           if(size(I,3)==1)
               I=repmat(I,[1,1,3]);
           end 
end 

function [out]=preprocess_offset(o,mask)
       m_c= cat(3, mask, mask, mask, mask);
       %background is 1  
       out= single(o);
       out(m_c==1)=NaN;
       opts.constant=2000;
       out(:,:,1)=out(:,:,1)-opts.constant;
       out(:,:,2)=out(:,:,2)-opts.constant;
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