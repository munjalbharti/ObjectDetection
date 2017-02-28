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
    
    batch_size = 1000;
    imdb_target_file = fullfile('F:','CamelyonTrainingData','imdb','camlyon_500k_500km-L4-131-loaded.imdb.mat');
    imdb.meta.data_mat_file = imdb_target_file;
    m=matfile(imdb_target_file,'Writable',true);
    
    for  batch_start=1:batch_size:num_files
        
        batch_end = min(num_files,batch_start+batch_size -1);
        cur_batch_len = batch_end - batch_start + 1;
        data = zeros(131,131,3,cur_batch_len,'uint8');

        for f=batch_start:batch_end
          i = f - batch_start +1;   
          data(:,:,:,i) = imread(filenames{f});

        end
        m.data(1:131,1:131,1:3,batch_start:batch_end) = data;
        
   
    end
    %save(imdb_target_file, 'imdb', '-v7.3');
end

