function net = cnn_initialization_ResNetUnpool2InterConv_offset(varargin)

opts.scale = 1;
opts.weightDecay = [1,0];

%% Load ResNet and remove last layers

net = load('ResNet\imagenet-resnet-50-dag.mat');
net = dagnn.DagNN.loadobj(net) ;
net.removeLayer('pool5');   %last pooling
net.removeLayer('prob');    %softmax
net.removeLayer('fc1000');  %fully-connected layer
n_previous_layers = numel(net.layers);

% % Fix BatchNorm kernel issues (already fixed - this was for an older version)
% for i=1:numel(net.layers)
%     if isa(net.layers(i).block,'dagnn.BatchNorm')
%         p = net.getParamIndex(net.layers(i).params);
%         net.params(p(1)).value = reshape(net.params(p(1)).value, [], 1);
%         net.params(p(2)).value = reshape(net.params(p(2)).value, [], 1);
%     end
% end

%% ------------------------------------------------------------------------
%% Repurpose net as FCN
%% ------------------------------------------------------------------------

%% ---------- Transit layer -----------------------------------------------
 net.addLayer('layer1', ...
    dagnn.Conv('size', [1,1,2048,1024], 'pad', 0, 'stride', 1), ...
    {'res5cx'}, ...
    {'x1'}, ...
    {'layer1_f','layer1_b'});
params = net.params(net.layers(numel(net.layers)).paramIndexes);
[params.learningRate] = deal(opts.scale, opts.scale);
[params.weightDecay] = deal(opts.weightDecay(1), opts.weightDecay(1));
net.params(net.layers(numel(net.layers)).paramIndexes) = params;

net.addLayer('layer1_BN', ...
    dagnn.BatchNorm('numChannels', 1024), ...
    {'x1'}, ...
    {'x1BN'}, ...
    {'layer1BN_mult', 'layer1BN_bias', 'layer1BN_moments'});
params = net.params(net.layers(numel(net.layers)).paramIndexes);
[params.learningRate] = deal(opts.scale, opts.scale, opts.scale);
[params.weightDecay] = deal(opts.weightDecay(1), opts.weightDecay(1), opts.weightDecay(1));
net.params(net.layers(numel(net.layers)).paramIndexes) = params;

%% Add up-projection blocks as interleaving convolutions

id1 = '2x';
input = {'x1BN'};
upProject(net, [3,3,1024,512], input, id1, 'pad', 1, 'stride', 1);  %2x
id2 = '4x';
upProject(net, [3,3,512,256], net.layers(end).outputs, id2, 'pad', 1, 'stride', 1);  %4x
id3 = '8x';
upProject(net, [3,3,256,128], net.layers(end).outputs, id3, 'pad', 1, 'stride', 1);  %8x
id4 = '16x';
upProject(net, [3,3,128,64], net.layers(end).outputs, id4, 'pad', 1, 'stride', 1);  %16x

id5 = '32x';
upProject(net, [3,3,64,32], net.layers(end).outputs, id5, 'pad', 1, 'stride', 1);  %32x

net.addLayer('drop', ...
    dagnn.DropOut(), ...
    net.layers(end).outputs, ...
    {'dropped'});

%% Prediction

net.addLayer('ConvPred', ...
    dagnn.Conv('size', [3,3,32,2], 'pad', 1, 'stride', 1), ...
    'dropped', ...
    {'prediction'}, ...
    {'ConvPred_f','ConvPred_b'});
params = net.params(net.layers(numel(net.layers)).paramIndexes);
[params.learningRate] = deal(opts.scale, opts.scale);
[params.weightDecay] = deal(opts.weightDecay(1), opts.weightDecay(1));
net.params(net.layers(numel(net.layers)).paramIndexes) = params;

%% Initialize parameters for new layers

for l = n_previous_layers+1:numel(net.layers)
  p = net.getParamIndex(net.layers(l).params) ;
  params = net.layers(l).block.initParams() ;
  switch net.device
    case 'cpu'
      params = cellfun(@gather, params, 'UniformOutput', false) ;
    case 'gpu'
      params = cellfun(@gpuArray, params, 'UniformOutput', false) ;
  end
  [net.params(p).value] = deal(params{:}) ;
end

%% ---------- Loss - Errors -----------------------------------------------

%net.addLayer('loss', ...              
%net.addLayer('loss', dagnn.SegmentationLoss('loss', 'softmaxlog'),{'prediction', 'label'}, 'objective') ;

%net.addLayer('accuracy',dagnn.SegmentationAccuracy(), {'prediction', 'label'}, 'accuracy') ;

net.addLayer('loss',dagnn.L2LossAdaptiveBoundary(), {'prediction', 'label'}, 'objective') ;

net.addLayer('error',dagnn.RMSError(), {'prediction', 'label'}, 'error') ;


%if 1
 % figure(100) ; clf ;
 % n = numel(net.vars) ;
 % for i=1:n
   % vl_tightsubplot(n,i) ;
   % showRF(net, 'input', net.vars(i).name) ;
   % title(sprintf('%s', net.vars(i).name)) ;
  %  drawnow ;
 % end
%end




    
end


%% ------------------------------------------------------------------------
function upProject(net, size, input, id, varargin)
%% ------------------------------------------------------------------------
% Implements Res-block for up-sampling by interleaving convolutions

%% Options
opts.lr = [1,1];                %Learning rate for Conv layer
opts.BNlr = [1,1,1];            %Learning rate for BatchNorm layer
opts.weightDecay = [1,1];       %Weight decay for Conv layer
opts.BNweightDecay = [1,1,1];   %Weight decay for BatchNorm layer
opts.pad = 0;                   %Padding
opts.stride = 1;                %Stride
opts.BN = true;                 %Optionally include batch normalization
opts.ReLU = true;               %Optionally include ReLU
opts = vl_argparse(opts, varargin);


% -- Branch 1 -------------------------------------------------------------

id_br1 = sprintf('%s_br1', id);

%Interleaving Convs of 1st branch
unpool_as_conv(net, size, input, id_br1, 'pad', opts.pad, 'stride', opts.stride);

% Following Conv of 1st branch
layerName = sprintf('layer%s_Conv', id);
net.addLayer(layerName, ...
    dagnn.Conv('size', [size(1),size(2),size(4),size(4)], 'pad', opts.pad, 'stride', opts.stride), ...   
    net.layers(end).outputs, ...                                          
    {sprintf('x_%s_conv', id)}, ...                                         
    {sprintf('%s_filter',layerName), sprintf('%s_bias',layerName)});    
%set parameters
if numel(opts.lr) == 1, opts.lr(2) = opts.lr(1); end
if numel(opts.weightDecay) == 1, opts.weightDecay(2) = opts.weightDecay(1); end
p = net.getParamIndex(net.layers(end).params) ;
[net.params(p).learningRate] = deal(opts.lr(1), opts.lr(2));
[net.params(p).weightDecay] = deal(opts.weightDecay(1), opts.weightDecay(2));

% Batch Norm
layerName = sprintf('layer%s_BN', id);
net.addLayer(layerName, ...
    dagnn.BatchNorm('numChannels', size(4)), ...
    {sprintf('x_%s_conv', id)}, ...
    {sprintf('x_%s_BN', id)}, ...
    {sprintf('%s_mult',layerName), sprintf('%s_bias',layerName), sprintf('%s_moments',layerName)});
%set parameters
if numel(opts.BNlr) == 1, opts.BNlr(2:3) = opts.BNlr(1); end
if numel(opts.BNweightDecay) == 1, opts.BNweightDecay(2:3) = opts.BNweightDecay(1); end
p = net.getParamIndex(net.layers(end).params) ;
[net.params(p).learningRate] = deal(opts.BNlr(1), opts.BNlr(2), opts.BNlr(3));
[net.params(p).weightDecay] = deal(opts.BNweightDecay(1), opts.BNweightDecay(2), opts.BNweightDecay(3));

branch1_output = net.layers(end).outputs; 

% -- Branch 2 -------------------------------------------------------------

id_br2 = sprintf('%s_br2', id);
%Interleaving Convs of 2nd branch
unpool_as_conv(net, size, input, id_br2, 'pad', opts.pad, 'stride', opts.stride, 'ReLU', false);
branch2_output = net.layers(end).outputs; 

% Sum branches
net.addLayer(sprintf('layer%s_Sum', id), ...
    dagnn.Sum(), ...
    {branch1_output{1}, branch2_output{1}}, ...
    {sprintf('x_%s_sum', id)});

net.addLayer(sprintf('layer%s_ReLU', id), ...
    dagnn.ReLU(), ...
    {sprintf('x_%s_sum', id)}, ...
    {sprintf('x_%s_ReLU', id)});
end


%% ------------------------------------------------------------------------
function unpool_as_conv(net, size, input, id, varargin)
%% ------------------------------------------------------------------------
% Implements interleaving convolutions for up-sampling

%% Options
opts.lr = [1,1];                %Learning rate for Conv layer
opts.BNlr = [1,1,1];            %Learning rate for BatchNorm layer
opts.weightDecay = [1,1];       %Weight decay for Conv layer
opts.BNweightDecay = [1,1,1];   %Weight decay for BatchNorm layer
opts.pad = 0;                   %Padding
opts.stride = 1;                %Stride
opts.BN = true;                 %Optionally include batch normalization
opts.ReLU = true;               %Optionally include ReLU
opts = vl_argparse(opts, varargin);


%% Building block

% -- Convolution A --------------------------------------------------------
layerName = sprintf('layer%s_ConvA',id);
net.addLayer(layerName, ...                                                 %Layer name
    dagnn.Conv('size', [3,3,size(3),size(4)], 'pad', opts.pad, 'stride', opts.stride), ...   %Layer type
    input, ...                                                              %input name
    {sprintf('x_%s_A', id)}, ...                                            %output name
    {sprintf('%s_filter',layerName), sprintf('%s_bias',layerName)});        %parameter names
%set parameters
if numel(opts.lr) == 1, opts.lr(2) = opts.lr(1); end
if numel(opts.weightDecay) == 1, opts.weightDecay(2) = opts.weightDecay(1); end
p = net.getParamIndex(net.layers(end).params) ;
[net.params(p).learningRate] = deal(opts.lr(1), opts.lr(2));
[net.params(p).weightDecay] = deal(opts.weightDecay(1), opts.weightDecay(2));

% -- Convolution B --------------------------------------------------------
layerName = sprintf('layer%s_ConvB',id);
net.addLayer(layerName, ...                                                 %Layer name
    dagnn.Conv('size', [2,3,size(3),size(4)], 'pad', [1,0,1,1], 'stride', opts.stride), ...   %Layer type
    input, ...                                                              %input name
    {sprintf('x_%s_B', id)}, ...                                            %output name
    {sprintf('%s_filter',layerName), sprintf('%s_bias',layerName)});        %parameter names
%set parameters
if numel(opts.lr) == 1, opts.lr(2) = opts.lr(1); end
if numel(opts.weightDecay) == 1, opts.weightDecay(2) = opts.weightDecay(1); end
p = net.getParamIndex(net.layers(end).params) ;
[net.params(p).learningRate] = deal(opts.lr(1), opts.lr(2));
[net.params(p).weightDecay] = deal(opts.weightDecay(1), opts.weightDecay(2));

% -- Convolution C --------------------------------------------------------
layerName = sprintf('layer%s_ConvC',id);
net.addLayer(layerName, ...                                                 %Layer name
    dagnn.Conv('size', [3,2,size(3),size(4)], 'pad', [1,1,1,0], 'stride', opts.stride), ...   %Layer type
    input, ...                                                              %input name
    {sprintf('x_%s_C', id)}, ...                                            %output name
    {sprintf('%s_filter',layerName), sprintf('%s_bias',layerName)});        %parameter names
%set parameters
if numel(opts.lr) == 1, opts.lr(2) = opts.lr(1); end
if numel(opts.weightDecay) == 1, opts.weightDecay(2) = opts.weightDecay(1); end
p = net.getParamIndex(net.layers(end).params) ;
[net.params(p).learningRate] = deal(opts.lr(1), opts.lr(2));
[net.params(p).weightDecay] = deal(opts.weightDecay(1), opts.weightDecay(2));

% -- Convolution D --------------------------------------------------------
layerName = sprintf('layer%s_ConvD',id);
net.addLayer(layerName, ...                                                 %Layer name
    dagnn.Conv('size', [2,2,size(3),size(4)], 'pad', [1,0,1,0], 'stride', opts.stride), ...   %Layer type
    input, ...                                                              %input name
    {sprintf('x_%s_D', id)}, ...                                            %output name
    {sprintf('%s_filter',layerName), sprintf('%s_bias',layerName)});        %parameter names
%set parameters
if numel(opts.lr) == 1, opts.lr(2) = opts.lr(1); end
if numel(opts.weightDecay) == 1, opts.weightDecay(2) = opts.weightDecay(1); end
p = net.getParamIndex(net.layers(end).params) ;
[net.params(p).learningRate] = deal(opts.lr(1), opts.lr(2));
[net.params(p).weightDecay] = deal(opts.weightDecay(1), opts.weightDecay(2));


% -- Interleaving feature maps --------------------------------------------
layerName = sprintf('layer%s_Interleave', id);
net.addLayer(layerName, ... 
    dagnn.Combine(), ...
    {sprintf('x_%s_A', id), sprintf('x_%s_B', id), sprintf('x_%s_C', id), sprintf('x_%s_D', id)}, ...
    {sprintf('x_%s_Inter', id)});

% -- Batch Norm -----------------------------------------------------------
if opts.BN
    layerName = sprintf('layer%s_BN', id);
    net.addLayer(layerName, ...
        dagnn.BatchNorm('numChannels', size(4)), ...
        {sprintf('x_%s_Inter',id)}, ...
        {sprintf('x_%s_BN', id)}, ...
        {sprintf('%s_mult',layerName), sprintf('%s_bias',layerName), sprintf('%s_moments',layerName)});
    %set parameters
    if numel(opts.BNlr) == 1, opts.BNlr(2:3) = opts.BNlr(1); end
    if numel(opts.BNweightDecay) == 1, opts.BNweightDecay(2:3) = opts.BNweightDecay(1); end
    p = net.getParamIndex(net.layers(end).params) ;
    [net.params(p).learningRate] = deal(opts.BNlr(1), opts.BNlr(2), opts.BNlr(3));
    [net.params(p).weightDecay] = deal(opts.BNweightDecay(1), opts.BNweightDecay(2), opts.BNweightDecay(3));
end

% -- ReLU -----------------------------------------------------------------
if opts.ReLU
    layerName = sprintf('layer%s_ReLU', id);
    net.addLayer(layerName, ...
        dagnn.ReLU(), ...
        {sprintf('x_%s_BN',id)}, ...
        {sprintf('x_%s_ReLU', id)});
end


end
