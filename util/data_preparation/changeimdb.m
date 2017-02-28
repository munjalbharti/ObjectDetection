%imdb_old=load('F:\Bharti\Thesis\data\imdbVOC2012_segmentation_offsets_ssc256.mat');
imdb = IMDB.init();


for k=1:2913
    img=crop_randomly(imdb_old.imdb.images{k},);
    imdb.images.data(:,:,:,k)=;
    o=imdb_old.imdb.offsets{k};
    imdb.images.mask(:,:,:,k)=cat(3,o(:,:,[1,2]),imdb_old.imdb.labels{k});
    imdb.images.set(k)=imdb_old.imdb.set(1,k);
    imdb.images.id(k)=k;
end 



save('F:\Bharti\Thesis\data\all_classes_imdb.mat', 'imdb', '-v7.3');