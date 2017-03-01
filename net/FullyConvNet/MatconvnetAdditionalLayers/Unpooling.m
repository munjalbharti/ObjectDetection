classdef Unpooling < dagnn.Filter
  properties
    poolSize = [2 2]
    opts = {'cuDNN'}
  end

  methods
      function Y = unpool(X, pool, varargin)

            backMode = numel(varargin) > 0 ;
            s = [size(X,1), size(X,2), size(X,3), size(X,4)];
            pool = pool(1); %supports only summetric pooling

            if ~backMode
                Y = zeros(s(1)*pool, s(2)*pool, s(3), s(4), 'like', X);  %output has pool-times the original size
                Y(1:pool:s(1)*pool, 1:pool:s(2)*pool, :, :) = X;
            else
                dzdy = varargin{1}; %backpropagate derivatives
                Y = dzdy(1:pool:s(1), 1:pool:s(2),:,:); %return back to size(X)
            end

      end
      
     function outputs = forward(self, inputs, params)
        outputs{1} = self.unpool(inputs{1}, self.poolSize);               
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
        derInputs{1} = self.unpool(derOutputs{1}, self.poolSize, derOutputs{1});
        derParams = {} ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.poolSize ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1}(1) = inputSizes{1}(1)*obj.poolSize(1);
      outputSizes{1}(2) = inputSizes{1}(2)*obj.poolSize(2);
      outputSizes{1}(3) = inputSizes{1}(3);
      outputSizes{1}(4) = inputSizes{1}(4);
    end

    function obj = Unpooling(varargin)
      obj.load(varargin) ;
    end
  end
end
