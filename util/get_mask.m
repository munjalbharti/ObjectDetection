function[new_mask]= get_mask(labeled_mask,display)

%labeled_mask has 1 2 values in it.
%This function turns it into a displayable maask

new_mask=zeros(size(labeled_mask));

new_mask(find(labeled_mask == 2))=255;
new_mask(find(labeled_mask ==1))=0;

if(display)
    figure; imshow(new_mask);
end 
end 