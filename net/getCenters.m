function [heatmap, bins] = getCenters(im, votes, votingResolution, bin, displayLevel)
% -------------------------------------------------------------------------
% Heatmap representing offsets voting for the center(s)
% -------------------------------------------------------------------------

szH = ceil(votingResolution/bin);
% create vote map
x = ceil(votes(:,1)/bin);   % ceil because indexing starts at 1
y = ceil(votes(:,2)/bin);
ind = sub2ind([szH(1) szH(2)], y, x);
%heatmap = accumarray(ind, votes(:,5), [prod(szH),1]);
heatmap = accumarray(ind,1, [prod(szH),1]);
heatmap = reshape(heatmap, szH);
bins = cat(2,x,y);

if displayLevel >= 1
    figure(50);
    cmap = colormap(jet(256));
    H = imresize(heatmap, [size(im, 1) size(im,2)], 'bilinear');
    H = uint8( 255 * (H - min(H(:))) / (max(H(:)) - min(H(:))) );
    RGB = ind2rgb(H, cmap);
    imshowpair(im, RGB, 'blend'); axis equal
end
