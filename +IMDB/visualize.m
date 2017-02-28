function handle = visualize( imdb, autopause )
%VISUALIZE 

    if nargin < 2
        autopause = 0;
    end

    handle = figure;
    imdb.images.data = double(normalize2(imdb.images.data));
    for i=1:size(imdb.images.data, 4)
        figure(handle), cla;
        imshow(squeeze(imdb.images.data(:,:,:,i)));
        
        gt = 'n/a';
        try
            gt = num2str(imdb.images.labels_gold(i));
        catch
            disp('No gold label available');
        end
        
        label = 'n/a';
        try
            label = num2str(imdb.images.labels(i));
        catch
            disp('No label available');
        end
        
        set = 'n/a';
        try
            set = num2str(imdb.images.set(i));
        catch
            disp('No label available');
        end
        
        try
            xlabel(['Label: ' label ' - Set: ' set ' - GT: ' gt]);
        catch
            disp('Label could not be determined');
        end
        
        if autopause
            pause(1000);
        else
            pause;
        end
    end
    
end

