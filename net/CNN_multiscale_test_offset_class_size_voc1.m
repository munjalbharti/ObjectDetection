function [] = CNN_multiscale_test_offset_class_size_voc1()
 
        close all;
        clear;
 
        % Do not forget to run Setup first!
        run ..\RGBObjectDetectionSetUp.m
 
        opts.baseDir = 'E:\Bharti\Code\Thesis\data\' ;
 
        opts.expDir=[opts.baseDir filesep 'rgb_object_detection-voc-trainval-L2AdaptiveLoss'];
       
        opts.epoch=1305; %124;
        opts.resultDir=[opts.expDir filesep 'ResultsNow' filesep 'VOCval07' filesep sprintf('%d',opts.epoch)];
        opts.modelPath = [opts.expDir filesep sprintf('net-epoch-%d.mat',opts.epoch)] ;
       % 2008_000075
        opts.gpus=[1];
        opts.scales = [128 192 256 320 384];
        opts.votingScale = 3;
     
 
        opts.threshold=5;
        opts.bin_size=5;
        opts.non_m_win_size=9;
        post_process_pref=sprintf('hough_th_%d_bin_%d_win_%d_with_nmx_conf_avg_multi_scale',opts.threshold,opts.bin_size(1),opts.non_m_win_size);
        
 
        if not (exist(fullfile(opts.resultDir,post_process_pref),'dir')==7)
                mkdir(fullfile(opts.resultDir,post_process_pref));
        end
        
     opts.classes={...
    'aeroplane'
    'bicycle'
    'bird'
    'boat'
    'bottle'
    'bus'
    'car'
    'cat'
    'chair'
    'cow'
    'diningtable'
    'dog'
    'horse'
    'motorbike'
    'person'
    'pottedplant'
    'sheep'
    'sofa'
    'train'
    'tvmonitor'};
 
opts.nclasses=length(opts.classes);
fids={};
%VOCPATH=[opts.baseDir filesep 'VOC2012'];
VOCPATH='E:\Bharti\Code\Thesis\data\VOC2012\';
opts.annopath=[VOCPATH  filesep 'Annotations'];
opts.imgpath=[VOCPATH filesep 'JPEGImages' ];
opts.imgsetpath= fullfile(VOCPATH, 'ImageSets', 'Main' , 'val.txt');
 
 
   
 net=get_test_net(opts);     
 predVar1 = net.getVarIndex('prediction1') ;
 predVar2 = net.getVarIndex('prob') ;
 predVar3 = net.getVarIndex('prediction3') ;
 
 %predVar = net.getVarIndex('prediction') ;
 inputVar = 'data' ;
       
 [ids,~]=textread(opts.imgsetpath,'%s %d');
    
 for i=1:length(ids)
             fprintf('Evaluating image %d\n',i);
          
             name=sprintf('%s.png',ids{i});
             rgb= imread(fullfile(opts.imgpath,sprintf('%s.jpg',ids{i})));
     
             I =single(rgb) ;
             
             im_height=size(I,1);
             im_width=size(I,2);
             
             %prob_scales=[];
             votes=[];
             for currentScale = opts.scales      
                 if(im_height < im_width)
                      votingResolution = round([im_height, im_width] * 256 / im_height);           
                      im_=imresize(I,[currentScale,NaN]);
                 else 
                      votingResolution = round([im_height, im_width] * 256 / im_width);
                      im_=imresize(I,[NaN,currentScale]);
                 end 
                 
                 if ~isempty(opts.gpus)
                    im_ = gpuArray(im_) ;
                 end
 
             tStart=tic;
             
       
             net.eval({inputVar, im_}) ;
             
             tElapsed = toc(tStart);
             fprintf('Time elapsed %f sec for image %d\n',tElapsed,i);
           
   
             %prediction= gather(net.vars(predVar).value)  ;   
             %offset=prediction(:,:,[22,23])*256;
             offset= gather(net.vars(predVar1).value)  ;
             offset=offset*256;
            
             bbox_size1 = gather(net.vars(predVar3).value) ;
             bbox_size1= bbox_size1*256;
            
             prob = gather(net.vars(predVar2).value) ;    
             votes = cat(1, votes, houghVoting(votingResolution, offset, bbox_size1, prob));
             %prob_scales = cat(4, prob_scales, imresize(prob, [im_height, im_width]));             
             
            end      
              
            [heatmap, bins] = getCenters(I, votes, votingResolution, opts.bin_size, 0);              
            
            heatmap_nms = nms(heatmap,  opts.non_m_win_size,opts.threshold); %heatmap can be 2D or 3D (per class)
            [confidence, class, BB] = getBoundingBoxes(heatmap_nms, votes, bins, votingResolution, I, 0);             
            
            [BB, class, confidence] = mergeCenters1(BB, class, confidence);
            
            for k=1:opts.nclasses
                    class_name= opts.classes{k};
                    det_result_file2=sprintf('%s_det_val_%s.txt','comp3',class_name);
                    fid=fopen(fullfile(opts.resultDir,post_process_pref,det_result_file2),'at');
                    fids{k}=fid;        
            end 
             
            for k=1:length(class)
                class_ass=class(k);
                fid2=fids{class_ass};       
                r_min_x= BB(1,k);
                r_min_y= BB(2,k);
                r_max_x= BB(3,k);
                r_max_y= BB(4,k);
                fprintf(fid2,'%s %f %f %f %f %f \n',name(1:end-4),r_min_x,r_min_y,r_max_x,r_max_y,confidence(k));
            end
            
            for k=1:opts.nclasses
                      fid=fids{k};
                      fclose(fid);
             end 
             
      
              %{
    
             title('Prediction');
             hold off ;
       
 
    
             subplot(2,2,4);
             image(uint8(class_pred-1)) ;
             
             colormap(labelColors(21)) ;
 
             savefig(f,fullfile(opts.resultDir,post_process_pref, sprintf('%s.fig',name(1:end-4))));
             saveas(f,fullfile(opts.resultDir,post_process_pref, name));
             close(f);
             %}
          
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
 

