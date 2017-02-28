function imdb = init()
%INIT Returns a totally empty imdb struct. This is for convenience to make
%sure you know what this struct looks like
%
% @Author: Christoph Baur

    % Gain access to all methods of the package
    import IMDB.*

    imdb = struct;
    
    
    imdb.meta.name='';
    imdb.meta.pathstr='';
    
    imdb.paths.image='';
    imdb.paths.mask='';
    imdb.paths.orig_segmentation='';
    
    imdb.sets.id=[];
    imdb.sets.name={};
   
    imdb.classes.id=[];
    imdb.classes.name={};
    imdb.classes.images={};
    
    
    imdb.images = struct;
    imdb.images.id=[];
    imdb.images.filenames = {};
    imdb.images.label_filenames = {};
    imdb.images.offset_filenames = {};
    
    imdb.images.data = [];
    imdb.images.mask = [];
    imdb.images.boundary_mask = [];
    imdb.images.set = [];
    imdb.images.size=[];
    imdb.images.labels=[];
    
    
    imdb.images.dataMean = [];
    imdb.images.dataStd = [];

   
   
end

