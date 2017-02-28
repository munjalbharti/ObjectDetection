function [] = CNN_test_offset_class_size_voc_segmentation()
 
        close all;
        clear;
 
        % Do not forget to run Setup first!
        run ..\RGBObjectDetectionSetUp.m
 
        opts.baseDir = 'E:\Bharti\Code\Thesis\data\' ;
 
        opts.expDir=[opts.baseDir filesep 'rgb_object_detection-coco-voc-normalised-L2AdaptiveLoss'];
        opts.epoch=51; %124;
        opts.resultDir=[opts.expDir filesep 'Results' filesep 'VOCval' filesep sprintf('%d',opts.epoch) filesep 'Segmentation' filesep 'comp5_val_cls'];
        opts.modelPath = [opts.expDir filesep sprintf('net-epoch-%d.mat',opts.epoch)] ;
              
        opts.gpus=[1];
      
        if not (exist(fullfile(opts.resultDir),'dir')==7)
                mkdir(fullfile(opts.resultDir));
        end
 
 
VOCPATH=[opts.baseDir filesep 'VOC2012'];
%VOCPATH='F:\Bharti\Thesis\voc\VOCdevkit\VOC2007Test\';
opts.imgpath=[VOCPATH filesep 'JPEGImages' ];
opts.imgsetpath= fullfile(VOCPATH, 'ImageSets', 'Segmentation' , 'val.txt');
%opts.imgsetpath= fullfile(VOCPATH, 'ImageSets', 'Main' , 'val.txt');
 
net=get_test_net(opts);     
 
predVar2 = net.getVarIndex('prob') ;
inputVar = 'data' ;
       
 [ids,~]=textread(opts.imgsetpath,'%s %d');
    
 for i=1:length(ids)
             fprintf('Evaluating image %d\n',i);
          
             filename=sprintf('%s.png',ids{i});
             rgb= imread(fullfile(opts.imgpath,sprintf('%s.jpg',ids{i})));
     
             im_ =single(rgb) ;
             
             im_height=size(im_,1);
             im_width=size(im_,2);
             
             if(im_height < im_width)
                  im_=imresize(im_,[256,NaN]);
             else 
                  im_=imresize(im_,[NaN,256]);
             end 
        
             if ~isempty(opts.gpus)
                im_ = gpuArray(im_) ;
             end
   
          tic;
             net.eval({inputVar, im_}) ;
             toc ;
             prob = gather(net.vars(predVar2).value) ;
             
             prob_r= imresize(prob, [im_height, im_width],'nearest');
            
             [prob_pred,class_pred] = max(prob_r,[],3) ;
             cmap=VOClabelcolormap(256);
             %cmap=labelColors(21);
             %rgbImage = ind2rgb(class_pred, cmap);
          
             imwrite(class_pred,cmap,fullfile(opts.resultDir,filename));
 
          
       end
       
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
 
 

