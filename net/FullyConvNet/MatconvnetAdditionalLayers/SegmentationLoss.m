classdef SegmentationLoss < dagnn.Loss
    
    properties
    numClass = 21;
   
  end
 
  methods
    function outputs = forward(obj, inputs, params)
      predictions=inputs{1};
      true_labels=inputs{2};
      %true_labels=cat(4,true_labels{:});
      true_labels=true_labels(:,:,[1],:);
      
      mass = sum(sum(true_labels > 0,2),1) + 1 ;
      outputs{1} = vl_nnloss(predictions, true_labels, [],'loss', obj.loss,'instanceWeights', 1./mass) ;
      %outputs{1} = vl_nnloss(predictions, true_labels, [],'loss', obj.loss) ; 
      n = obj.numAveraged ;
      m = n + size(predictions,4) ;
      obj.average = (n * obj.average + double(gather(outputs{1}))) / m ;
      obj.numAveraged = m ;
    end
 
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        predictions=inputs{1};
        true_labels=inputs{2};
       % true_labels=cat(4,true_labels{:});
        true_labels=true_labels(:,:,[1],:);
        mass = sum(sum(true_labels > 0,2),1) + 1 ;
        
         derInputs{1} = vl_nnloss(predictions, true_labels, derOutputs{1}, 'loss', obj.loss, 'instanceWeights', 1./mass) ;
        % derInputs{1} = vl_nnloss(predictions, true_labels, derOutputs{1}, 'loss', obj.loss) ;
       
  
        derInputs{2} = [] ;
        derParams = {} ;
    end
 
    function obj = SegmentationLoss(varargin)
      obj.load(varargin) ;
    end
  end
end

