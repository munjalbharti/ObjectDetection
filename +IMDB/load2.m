function imdb = load2(path)
%INIT load and prepare an IMDB
%
% @Author: Christoph Baur

    load(path);
    if round(max(imdb.images.data(:))) > 1
        imdb.images.data = normalize2(imdb.images.data, 'single', 8);
    else
        imdb.images.data = single(imdb.images.data);
    end
    imdb.images.labels = single(imdb.images.labels);
    
    % Add filename to meta if not available
    if ~isfield(imdb.meta, 'name') || ~isfield(imdb.meta, 'pathstr')
        [pathstr,name,ext] = fileparts(path);
        imdb.meta.name = name;
        imdb.meta.pathstr = pathstr;
    end
    filenames = imdb.images.filenames;
    im_cell = vl_imreadjpeg(filenames,'NumThreads',6); 
    images = cat(4,im_cell{:});
end

