function imdb = load_shuf_val(path)
%INIT load and prepare an IMDB
%
% @Author: Christoph Baur

    load(path);
    
    val_set = find(imdb.images.set==2);
    ridx = val_set(randperm(numel(val_set)));
    imdb.images.data(:,:,:,val_set) = imdb.images.data(:,:,:,ridx);
    imdb.imagas.labels(1,val_set) = imdb.images.labels(1,ridx);
    % Add filename to meta if not available
    if ~isfield(imdb.meta, 'name') || ~isfield(imdb.meta, 'pathstr')
        [pathstr,name,ext] = fileparts(path);
        imdb.meta.name = name;
        imdb.meta.pathstr = pathstr;
    end
end

