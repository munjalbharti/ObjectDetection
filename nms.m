function Is = nms(I, non_m_win_size, th)

% Non-maximum suppression

if ~exist('non_m_win_size', 'var')
    non_m_win_size = ones(1, size(size(I)));  %half-window
end

if ~exist('th', 'var')
    th = 200;
end

Is = I;
I(I < th) = 0;

%Is = nlfilter(I, [w w], @(x) x((w^2+1)/2)*all(x((w^2+1)/2)) > x(:));

mask = ones(non_m_win_size); 
  
maxNeighbour = imdilate(I, mask);
tmp = Is;
tmp(I < maxNeighbour | maxNeighbour == 0) = 0;
Is = tmp;
    %     for i = 1:sz(1)
    %         for j = 1:sz(2)
    %             win = I(max(1,i-w):min(sz(1),i+w), max(1,j-w):min(sz(2),j+w));
    %             if isempty(find(win > I(i,j)))
    %                 % I(i,j) is a peak
    %                 Is(i,j,k) = 1; %create mask
    %             end
    %         end
    %     end
    
fprintf(' found %d centers ', numel(find(Is)));

end