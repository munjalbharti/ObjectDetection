function imdb = load3(path)
%INIT load and prepare an IMDB
%
% @Author: Christoph Baur

    load(path);
    
    % Add filename to meta if not available
    if ~isfield(imdb.meta, 'name') || ~isfield(imdb.meta, 'pathstr')
        [pathstr,name,ext] = fileparts(path);
        imdb.meta.name = name;
        imdb.meta.pathstr = pathstr;
    end
    
    filenames = imdb.images.filenames;
    num_files = length(filenames);
    data = zeros(131,131,3,num_files,'uint8');
   
    for f=1:num_files
      data(:,:,:,f) = imread(filenames{f});
       
    end
    imdb.images.data = data;
    imdb.images.filenames = [];
    imdb_target_file = fullfile('F:','CamelyonTrainingData','imdb','camlyon_500k_500km-L4-131-loaded.imdb.mat');

    save(imdb_target_file, 'imdb', '-v7.3');
end

