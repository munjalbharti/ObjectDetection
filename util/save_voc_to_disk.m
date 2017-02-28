imdb=load('E:\Bharti\Code\Thesis\data\imdbVOC2012_segmentation_offsets_ssc256.mat');
%opts.voc_imagespath='E:\Bharti\Code\Thesis\voc\val_images\';
%opts.voc_labelspath='E:\Bharti\Code\Thesis\voc\val_labels\';
opts.voc_offsetspath='E:\Bharti\Code\Thesis\voc\train_offsets_uint8\';
opts.voc_offsetspath1='E:\Bharti\Code\Thesis\voc\train_offsets1_uint8\';

imdb=imdb.imdb;
ind=find(imdb.set==2);
for m=1:length(ind)
  k=ind(m);
  %img=imdb.images{k};
  %label_gt=imdb.labels{k}-1;
  off_size=imdb.offsets{k};
  
  file_name=sprintf('voc_%d.jpg',k);

  offset_gt = off_size(:,:,[1,2]);
  size_gt = off_size(:,:,[3,4]);
 % anns = imdb.objects{k};
  
 
 % imwrite(label_gt,cmap,fullfile(opts.voc_labelspath,sprintf('%s.png',file_name(1:end-4))));
     
  %imwrite(img,fullfile(opts.voc_imagespath,file_name));
  save(fullfile(opts.voc_offsetspath,sprintf('%s.mat',file_name(1:end-4))), 'offset_gt','size_gt', 'anns', '-v7.3');
     
  % imwrite(offset_gt(:,:,1),fullfile(opts.voc_offsetspath,sprintf('%s_ox.png',img.file_name(1:end-4))));
  % imwrite(offset_gt(:,:,1),fullfile(opts.voc_offsetspath,sprintf('%s_oy.png',img.file_name(1:end-4))));
  % imwrite(size_gt(:,:,1),fullfile(opts.voc_offsetspath,sprintf('%s_sx.png',img.file_name(1:end-4))));
  % imwrite(size_gt(:,:,2),fullfile(opts.voc_offsetspath,sprintf('%s_sy.png',img.file_name(1:end-4))));
       
     
    
         
       


end 
