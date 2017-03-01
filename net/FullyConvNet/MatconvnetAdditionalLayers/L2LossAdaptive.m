classdef L2LossAdaptive < dagnn.Loss

  methods
    function outputs = forward(obj, inputs, params)
      predictions=inputs{1};
      true_labels=inputs{2};
      true_labels=true_labels(:,:,[2,3],:);
      %mask= abs(true_labels) ~= single(0.1) ;
      mask = ~isnan(true_labels);
      [pos_y,pos_x,pos_z]= bkg2obj(predictions,mask);
      
      pos1 = sub2ind(size(mask),pos_y,pos_x,ones(size(pos_y,1),1),pos_z);
      pos2 = sub2ind(size(mask),pos_y,pos_x,ones(size(pos_y,1),1)+1,pos_z);
      pos=[pos1;pos2];
      
      %mask1 contains background pixels that point to foreground
      mask1=logical(false(size(mask)));
      mask1(pos)=1;
      mask1(find(mask))=0;
      
      diff = (predictions - true_labels);
      diff(~mask) = 0;
      diff(mask1) = predictions(mask1);
      
     
      
      full_mask=mask|mask1;
      
     % loss_per_img = squeeze((sum(sum(sum((predictions-true_labels).^2 .* mask,1),2),3))) ; %is a vector
      loss_per_img = squeeze((sum(sum(sum(diff.^2,1),2),3))) ; %is a vector
   
      n_pixels_per_img= squeeze(sum(sum(sum((full_mask),1),2),3)); %no of valid pixels per image..is a vector
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
        
        [pos_y,pos_x,pos_z]= bkg2obj(predictions,mask);
        pos1 = sub2ind(size(mask),pos_y,pos_x,ones(size(pos_y,1),1),pos_z);
        pos2 = sub2ind(size(mask),pos_y,pos_x,ones(size(pos_y,1),1)+1,pos_z);
        pos=[pos1;pos2];
      
        %mask1 contains background pixels that point to foreground
        mask1=logical(false(size(mask)));
        mask1(pos)=1;
        mask1(find(mask))=0;
        
      
         diff = (predictions - true_labels);
         diff(~mask) = 0;
         diff(mask1) = predictions(mask1);
      
         full_mask=mask|mask1;

         n_pixels_per_img= sum(sum(sum(full_mask,1),2),3); %
        
      % Y = 2 * bsxfun(@rdivide, ((predictions-true_labels) .* mask),  n_pixels_per_img);
        Y = 2 * bsxfun(@rdivide, diff,  n_pixels_per_img);
        
        %squeeze after gradient calculation
       n_pixels_per_img=squeeze(n_pixels_per_img);
       zero_pixels_img_indexes= find(n_pixels_per_img == 0);
       Y(:,:,:,zero_pixels_img_indexes)=0;
        
       % Y(~mask) = 0;  %this will just fill gradients with zeros and remove any NaNs that were caused by division with 0 
        derInputs{1} = derOutputs{1} *  Y;
       
        derInputs{2} = [] ;
        derParams = {} ;
    end

    
    function obj = L2Loss(varargin)
      obj.load(varargin) ;
    end
  end
end
