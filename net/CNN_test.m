function [info,confusion] = CNN_test()

        close all;
        clear;

        % Do not forget to run Setup first!
        run ..\RGBObjectDetectionSetUp.m
        opts.baseDir = 'F:\Bharti\Thesis\data\' ;
        
        opts.dataDir = [opts.baseDir filesep 'VOC2012'] ;

        opts.expDir=[opts.baseDir filesep 'rgb_object_detection-all-classes-segmentation-L-0.001'];
        opts.epoch=35;
        opts.resultDir=[opts.expDir filesep 'Results' filesep 'val' filesep sprintf('%d',opts.epoch)];
        opts.modelPath = [opts.expDir filesep sprintf('net-epoch-%d.mat',opts.epoch)] ;
        opts.imdbPath = fullfile([opts.baseDir filesep 'imdbVOC2012_segmentation_offsets_ssc256.mat']) ;

        opts.gpus=[1];

        resPath = fullfile(opts.resultDir, 'results.mat') ;

        if not (exist(opts.resultDir,'dir')==7)
                mkdir(opts.resultDir);
        end


        imdb = IMDB.load(opts.imdbPath);
        val = find(imdb.set == 2) ;
        
        net=get_test_net(opts);
       
        predVar = net.getVarIndex('prediction') ;
        inputVar = 'data' ;

        %Confusion matrix
        confusion = zeros(21) ;
        %To store response maps
        %resp_masks=zeros(0,0,0,0);
       
        %Evaluate each image
        for i=1:numel(val)
          j= val(i);
         % imId = imdb.images.id(j) ;
          name=sprintf('img_%d.png',i);
         
          %name = imdb.images.filenames{j} ;


          %rgb=imdb.images.data(:,:,:,j);
          rgb=imdb.images{j};
          im_ =single(rgb) ;
          %orig_mask =imdb.images.mask(:,:,:,j) ;
          %extract class information
          %lb=orig_mask(:,:,[3],:);;
         % lb=get_labeled_mask(orig_mask);
          lb=imdb.labels{j};
          lb=single(lb);
          

          %lb = mod(lb + 1, 256) ; % 0 = ignore, 1 = bkg

          % Subtract the mean (color)
         % im = bsxfun(@minus, single(rgb), net.meta.normalization.averageImage) ;

          if ~isempty(opts.gpus)
            im_ = gpuArray(im_) ;
          end

          center_point=[floor(size(im_,1)/2)+1, floor(size(im_,2)/2)+1];
          im_=crop_at_point( im_,center_point,256,256 );
          lb=crop_at_point( lb,center_point,256,256 );
          
          net.eval({inputVar, im_}) ;
          scores_ = gather(net.vars(predVar).value) ;
          [~,pred] = max(scores_,[],3) ;


          % Accumulate errors
          ok = lb > 0 ;
          
          c=accumarray([lb(ok),pred(ok)],1,[21 21]) ;
          confusion = confusion +c;

          % Plots
       if mod(i - 1,30) == 0 || i == numel(val)
            clear info ;
            %[info.iu, info.miu, info.pacc, info.macc] = getAccuracies(confusion) ;
            
            %fprintf('IU %4.1f ', 100 * info.iu) ;
            %fprintf('\n meanIU: %5.2f pixelAcc: %5.2f, meanAcc: %5.2f\n',100*info.miu, 100*info.pacc, 100*info.macc) ;
            disp('\nConfusion Matrix :: ');
            disp(confusion);
         
        end
       
            % Print segmentation
            f=figure(100) ;clf ;
            displayImage(rgb, lb, pred) ;
            drawnow ;

            % Save segmentation
            mask=uint8(pred);
           % resp_masks(:,:,:,i)=mask;
            
                    
           
            savefig(f,fullfile(opts.resultDir, sprintf('%s.fig',name(1:end-4))));
            saveas(f,fullfile(opts.resultDir, name) );
            
            save(fullfile(opts.resultDir, sprintf('%s.mat',name(1:end-4))),'mask');


            close(f);
       
       
        end   
        
 %       f1=figure(1) ; clf;
%        imagesc(normalizeConfusion(confusion)) ;
        
    %    axis image ;
    %    colormap(jet) ;
    %    colorbar ;
    %    drawnow ;
            
       % Save results
        %c_m_path=fullfile(opts.resultDir,'confusion_m.png');
       % saveas(f1,c_m_path);
       % close(f1);
        
        %save(resPath,'info','confusion','-v7.3') ;
        save(resPath,'confusion','-v7.3') ;


end 

function[net]= get_test_net(opts)
        orig_net=load(opts.modelPath);
        net = dagnn.DagNN.loadobj(orig_net.net) ;

        if ~isempty(opts.gpus)
          gpuDevice(opts.gpus(1)) ;
          net.move('gpu') ;
        end 

        for name = {'loss'} % 'accuracy'
              net.removeLayer(name) ;
        end
        %    net.meta.normalization.averageImage = reshape(net.meta.normalization.rgbMean,1,1,3) ;

        net.mode = 'test' ;
end 