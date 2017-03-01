classdef L2LossOffset < dagnn.Loss

  methods
    function outputs = forward(obj, inputs, params)
      predictions=inputs{1};
      true_labels=inputs{2};
      true_labels=true_labels(:,:,[2,3],:); %for size
      %mask= abs(true_labels) ~= single(0.1) ;
      mask = ~isnan(true_labels);
 
      
      diff = (predictions - true_labels).^2; %This will add NaN for background
     
      % to train background for zero
      %diff(~mask)= predictions(~mask).^2;     
      %full_mask= mask|~mask ;
      
      %to ignore background uncomment this
      diff(~mask) = 0;
     
     % loss_per_img = squeeze((sum(sum(sum((predictions-true_labels).^2 .* mask,1),2),3))) ; %is a vector
      loss_per_img = squeeze((sum(sum(sum(diff,1),2),3))) ; %is a vector
     
   
      n_pixels_per_img= squeeze(sum(sum(sum(mask,1),2),3)); %no of valid pixels per image..is a vector
      n_imgs=size(predictions,4);
      
      avg_loss_per_img =loss_per_img ./ n_pixels_per_img;
      avg_loss_per_img(n_pixels_per_img == 0)=0;
      outputs{1}=avg_loss_per_img;
     
      n = obj.numAveraged ;
      m = n + n_imgs;
      obj.average = (n * obj.average + double(gather(sum(outputs{1})))) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        predictions=inputs{1};
        true_labels=inputs{2};
        true_labels=true_labels(:,:,[2,3],:);
       % mask= abs(true_labels) ~= single(0.1) ;
        mask = ~isnan(true_labels);
     
        diff = predictions - true_labels;
        
        % to train background for zero  
       % diff(~mask)= predictions(~mask);       
       % full_mask= mask|~mask ;
     
        %to ignore background uncomment this
        diff(~mask) = 0;
      
        n_pixels_per_img= sum(sum(sum(mask,1),2),3); %
        
      % Y = 2 * bsxfun(@rdivide, ((predictions-true_labels) .* mask),  n_pixels_per_img);
        Y = 2 * bsxfun(@rdivide, diff,  n_pixels_per_img);
        
         %squeeze after gradient calculation
         %n_pixels_per_img=squeeze(n_pixels_per_img);
         %zero_pixels_img_indexes= find(n_pixels_per_img == 0);
         %Y(:,:,:,zero_pixels_img_indexes)=0;
        
         Y(~mask) = 0;
           
        %Y(~mask) = 0;  %this will just fill gradients with zeros and remove any NaNs that were caused by division with 0 
        derInputs{1} = Y * derOutputs{1};
       
        derInputs{2} = [] ;
        derParams = {} ;
    end

    function obj = L2LossOffset(varargin)
      obj.load(varargin) ;
    end
  end
end
