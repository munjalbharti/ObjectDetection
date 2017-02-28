function [] = CNN_test_offset_class()

        close all;
        clear;

        % Do not forget to run Setup first!
        run ..\RGBObjectDetectionSetUp.m

        opts.baseDir = 'E:\Bharti\Code\Thesis\data\' ;

        opts.expDir=[opts.baseDir filesep 'rgb_object_detection-offset-all-class-L-0.001-normalised'];
       
        opts.epoch=486; %124;
        opts.resultDir=[opts.expDir filesep 'Results1' filesep 'val' filesep sprintf('%d',opts.epoch)];
        opts.modelPath = [opts.expDir filesep sprintf('net-epoch-%d.mat',opts.epoch)] ;
        opts.imdbPath = fullfile([opts.baseDir filesep  'imdbVOC2012_segmentation_offsets_ssc256.mat']) ;
       
        opts.gpus=[];
        opts.results_file=fullfile(opts.resultDir,'results.mat');

      
        if not (exist(opts.resultDir,'dir')==7)
                mkdir(opts.resultDir);
        end


        imdb = IMDB.load(opts.imdbPath);
        val = find(imdb.set == 2) ;
  
        net=get_test_net(opts);
       
        predVar1 = net.getVarIndex('prediction1') ;
        predVar2 = net.getVarIndex('prob') ;
     
        inputVar = 'data' ;
        offsets={};
        class_probs={};
        filenames={};
        orig_classes={};
        orig_offsets={};
        images=[];
        
       for i=1286:numel(val)
          j= val(i);
        %  j=1353;
          name=sprintf('img_%d.png',i);
         % name = imdb.images.filenames{j} ;
          %rgb=imdb.images.data(:,:,:,j);
          rgb=imdb.images{j};
          im_ =single(rgb) ;
        
          if ~isempty(opts.gpus)
            im_ = gpuArray(im_) ;
          end

         
          net.eval({inputVar, im_}) ;
          offset= gather(net.vars(predVar1).value) * 256 ;
          prob = gather(net.vars(predVar2).value) ;
          
        %  mask =imdb.images.mask(:,:,:,j) ;
        %  orig_offset=mask(:,:,[1,2],:);
        % orig_class=mask(:,:,[3],:);
          
          orig_offset=imdb.offsets{j};
          orig_class=imdb.labels{j};
 
          offsets{i} = offset;
          class_probs{i} = prob;
          filenames{i}=name;
          
          orig_offsets{i}=orig_offset;
          orig_classes{i}=orig_class;
          
          images{i}=rgb;
          
             f=figure ;
            
             subplot(2,2,1);
             imshow(uint8(im_));
             impixelinfo;
             hold on ;
             title('Ground');
             
             quiver([1:size(orig_offset,2)],[1:size(orig_offset,1)],orig_offset(:,:,1),orig_offset(:,:,2));
             hold off ;
       
             subplot(2,2,2);
             image(uint8(orig_class-1)) ;
             
             subplot(2,2,3);
             imshow(uint8(im_));
                      
             hold on ;          
             quiver([1:size(offset,2)],[1:size(offset,1)],offset(:,:,1),offset(:,:,2));
             title('Prediction');
          
           
            [prob_pred,class_pred] = max(prob,[],3) ;
    
             subplot(2,2,4);
             image(uint8(class_pred-1)) ;
             
             colormap(labelColors(21)) ;

             savefig(f,fullfile(opts.resultDir, sprintf('%s.fig',name(1:end-4))));
             saveas(f,fullfile(opts.resultDir, name));
             close(f);
          
       end
           
      save(opts.results_file,'images','offsets','class_probs','orig_offsets','orig_classes','filenames');
        
end 

function[net]= get_test_net(opts)
        orig_net=load(opts.modelPath);
        net = dagnn.DagNN.loadobj(orig_net.net) ;

        if ~isempty(opts.gpus)
          gpuDevice(opts.gpus(1)) ;
          net.move('gpu') ;
        end 

        
       % for name = {'loss'}
       %       net.removeLayer(name) ;
       % end
        %    net.meta.normalization.averageImage = reshape(net.meta.normalization.rgbMean,1,1,3) ;

       
        net.addLayer('Softmax',dagnn.SoftMax(),{'prediction2'}, {'prob'});
        net.mode = 'test' ;
end 