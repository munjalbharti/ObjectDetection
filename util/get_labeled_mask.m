function [ labeled_mask ] = get_labeled_mask( mask )
%GET_LABELED_MASK Summary of this function goes here
%   Detailed explanation goes here
 
 labeled_mask=ones(size(mask));
 labeled_mask(find(mask ==0))=1;
 labeled_mask(find(mask ==255))=2;
    

end

