% Clean up
close all;
clear;

% Do not forget to run Setup first!
run ../RGBObjectDetectionSetUp.m

% Options
CNN_opts_offset;

% Create the net

net=cnn_initialization_ResNetUnpool2InterConv_offset_class_size();

%net=cnn_initialization_ResNetUnpool2InterConv_offset_and_class();


% 1. Load the IMDB file containing all patches
imdb = IMDB.load(opts.imdbPath);

%parpool;

% 2. Learn on the complete IMDB
[net,info] = cnn_train_dag_offset_and_class_and_size1(net, imdb, opts.getBatchFunction, ...
                       'expDir', opts.train.expDir, ...
                       'batchSize', opts.train.batchSize, ...
                       'numEpochs', opts.train.numEpochs, ...
                       'continue', opts.train.continue, ...
                       'gpus', opts.train.gpus, ...
                       'learningRate', opts.train.learningRate, ...
                       'weightDecay', opts.train.weightDecay, ...
                        'momentum', opts.train.momentum, ...
                       'numSubBatches',opts.train.numSubBatches);
                      % 'errorFunction', opts.train.errorFunction, ...
                     

% Plot progress
%vl_simplenn_diagnose(net);
