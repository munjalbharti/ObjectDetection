function [BBprob, BBclass, BB] = getBoundingBoxes(centers, votes, bins, votingResolution, I, displayLevel)
% -------------------------------------------------------------------------
% find all pixels that voted for a given center c
% -------------------------------------------------------------------------

if displayLevel >= 1
    VOCinit;
    figure(10), hold on
    cmap = VOClabelcolormap();  % supports up to 256 values
end

[yc, xc] = find(centers);
center_x = zeros(1,numel(xc));
center_y = zeros(1,numel(yc));

if displayLevel >= 2
    r = zeros(size(I,1), size(I,2));
    g = r;
    b = r;
    Ig = double(rgb2gray(uint8(I))); %convert image to gray scale
    Ig = repmat(Ig, 1,1,3);   %3-channel grayscale
    Ig = imnormalize(double(Ig), [0, 0.75]) + 0.25;
end

BB = [];
BBclass = [];
BBprob = [];
k = 1;
%seg(:,:,1) = seg(:,:,1)*0.5;
[~, voteClasses] = max(votes(:,5:end), [], 2);

for i = 1:length(xc)
    if displayLevel >= 2
        % voters (color-coded)
        r(mask > 0) = cmap(i+1,1);
        g(mask > 0) = cmap(i+1,2);
        b(mask > 0) = cmap(i+1,3);
        voters = cat(3, r, g, b);
    end
    
    % get votes
    voterIndices = bins(:,1) == xc(i) & bins(:,2) == yc(i);
    
    % remove voters from the background
    voterIndices = voterIndices & (voteClasses ~= 1);
    
    % compute sum of vote weights
    weights = votes(voterIndices,5);
    weightSum = sum(weights);
    
    % find class
    averageConfidence = sum(bsxfun(@times, votes(voterIndices,6:end), weights)) / weightSum;
    averageConfidence(1) = 0;
    [prob,class] = max(averageConfidence);
    
    if  (prob <= 0.2 || class == 1)
        continue; % discard if low probability or background
    end
    
    % compute center from votes
    currCenter = sum(bsxfun(@times, votes(voterIndices,1:2), weights)) / weightSum;
    center_x(i) = currCenter(1) * size(I,2) / votingResolution(2);
    center_y(i) = currCenter(2) * size(I,1) / votingResolution(1);
    
    % compute median bounding box from votes
    bbWidth = sum(votes(voterIndices,3) .* weights) / weightSum * size(I,2) / votingResolution(2);
    bbHeight = sum(votes(voterIndices,4) .* weights) / weightSum * size(I,2) / votingResolution(2);
    
    % final information about predicted bounding box
    %BB = [BB; ...
    %    [max(center_x(i) - mean_w/2, 1); max(center_y(i) - mean_h/2, 1); ...
    %    min(center_x(i) + mean_w/2, size(votes,2)); min(center_y(i) + mean_h/2, size(votes,1))]];
    %BBprob = [BBprob, prob];
    %BBclass = [BBclass, class-1];
    
    BB(1,k) = max(center_x(i) - bbWidth/2, 1);
    BB(2,k) = max(center_y(i) - bbHeight/2, 1);
    BB(3,k) = min(center_x(i) + bbWidth/2, size(I,2));
    BB(4,k) = min(center_y(i) + bbHeight/2, size(I,1));
    
    if BB(3,k)-BB(1,k)+1 <= 0 || BB(4,k)-BB(2,k)+1 <= 0
        continue; % discard if bb degenerate
    end
    
    BBprob(k) = prob;
    BBclass(k) = class - 1;
    
%     % visualization
%     if displayLevel >= 1
%         colorID = cmap(class, :); %class color
%         scatter(center_x(i), center_y(i), 'MarkerFaceColor', colorID, 'MarkerEdgeColor', 'none', 'LineWidth', 2);
%         rectangle('Position', [BB(1,k), BB(2,k), BB(3,k)-BB(1,k)+1, BB(4,k)-BB(2,k)+1], ...
%             'EdgeColor', colorID, 'LineWidth', 2);
%         line([BB(1,k), BB(3,k)], [BB(2,k), BB(4,k)], 'Linewidth', 0.5, 'Color', colorID);
%         line([BB(3,k), BB(1,k)], [BB(2,k), BB(4,k)], 'Linewidth', 0.5, 'Color', colorID);
%         drawnow;
%     end
    k = k + 1;
    %pause();
end

% if displayLevel >= 1
%     hold off
% end

if displayLevel >= 2
    figure(500);
    imshowpair(Ig, voters, 'blend'); axis on
    hold on
    scatter(center_x, center_y, [], cmap(2:length(center_x)+1,:), 'LineWidth', 3);
    hold off
end

end
