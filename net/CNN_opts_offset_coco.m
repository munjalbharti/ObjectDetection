% Options and parameters for the current CNN
%rgb_object_detection-15-person-offset-4794.imdb-again.mat
%rgb_object_detection-15-person-offset-class-4794-imdb
opts.imdbPath = fullfile('C:\Users\Bharti\Thesis\data\rgb_object_detection-coco-voc-imdb-ssd.mat');
opts.getBatchFunction = @CNN.getCOCOBatch1;

%lr = logspace(-1, -4, 30) ;
%lr1= ones(1,5)*0.0001 ;
%lr2= ones(1,25)*0.00001 ;

%lr = [lr1,lr2,0.00001] ;
% Training options

iter= [1:150];
lr=0.0001 * (1 - iter/max(iter)) .^ 0.9;

%rgb_object_detection-offset-all-class-L-poly-normalised
opts.train.expDir = 'E:\Bharti\Code\Thesis\data\rgb_object_detection-coco-voc-normalised-L2AdaptiveLoss\' ;


opts.train.batchSize = 8 ;%50;
opts.train.numEpochs = 10000;
%.00001;
opts.train.learningRate = .0001;
opts.train.weightDecay = 0.0005 ;
opts.train.continue = true;
opts.train.gpus = [1];
%opts.train.errorFunction = @CNN.ErrorFn;
opts.train.useGpu= true;
opts.train.momentum = 0.9 ; % this guy should be 0.9 
opts.train.numSubBatches =1  %5;

