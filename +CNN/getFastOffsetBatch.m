
%% ------------------------------------------------------------------------
function inputs = getFastOffsetBatch(imdb,opts, batch)
% -------------------------------------------------------------------------

opts.imageSize = [256,256];


sizes = cellfun(@size, imdb.images(batch), 'UniformOutput', false);
cropSize = cellfun(@(x) [randi(x(1)-opts.imageSize(1)+1), randi(x(2)-opts.imageSize(2)+1)], sizes, 'UniformOutput', false);

%apply the same crops to all frames
images = cellfun(@(x,y) x(y(1):y(1)+opts.imageSize(1)-1, y(2):y(2)+opts.imageSize(2)-1, :), imdb.images(batch), cropSize, 'UniformOutput', false);
images = single(cat(4, images{:}));
labels = cellfun(@(x,y) x(y(1):y(1)+opts.imageSize(1)-1, y(2):y(2)+opts.imageSize(2)-1, :), imdb.labels(batch), cropSize, 'UniformOutput', false);
labels = double(cat(4, labels{:}));
offsets = cellfun(@(x,y) x(y(1):y(1)+opts.imageSize(1)-1, y(2):y(2)+opts.imageSize(2)-1, :), imdb.offsets(batch), cropSize, 'UniformOutput', false);
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
