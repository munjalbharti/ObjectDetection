% Options and parameters for the current CNN
opts.imdbPath = fullfile('F:\Bharti\Thesis\data\imdbVOC2012_segmentation_offsets_ssc256.mat');
opts.getBatchFunction = @CNN.getOffsetBatchMultiClass1;

%lr = logspace(-1, -4, 30) ;
lr1= ones(1,5)*0.001 ;
lr2= ones(1,25)*0.0001 ;

lr = [lr1,lr2,0.0001] ;
% Training options

%0.001 for 48 epochs, then 0.0001
opts.train.expDir = 'F:\Bharti\Thesis\data\rgb_object_detection-all-classes-segmentation-L-0.001/';
opts.train.batchSize = 16 %50;
opts.train.numEpochs = 1000;
opts.train.learningRate = 0.0001;
opts.train.weightDecay = 0.0005 ;
opts.train.continue = true;
opts.train.gpus = [1];
%opts.train.errorFunction = @CNN.ErrorFn;
opts.train.useGpu= true;
opts.train.momentum = 0.9 ; % this guy should be 0.9 
opts.train.numSubBatches =1  %5;

