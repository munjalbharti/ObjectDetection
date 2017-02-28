function [ I_conn,Label ] = get_connected_components( I )

%Connect the components with intensity value 1
%expects a binary image
    min_conected_comp_size=15;
    CC = bwconncomp(I);
    numPixels = cellfun(@numel,CC.PixelIdxList);    
    I_conn=I;
    for k=1:CC.NumObjects
         if(numPixels(k) <= min_conected_comp_size)
             I_conn(CC.PixelIdxList{k}) = 0;
         end 
    end
    
end

