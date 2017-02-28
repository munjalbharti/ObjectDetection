function [votes] = houghVoting(votingResolution, offsets, bbSizes, confidences)

    sz = [size(offsets,1), size(offsets,2)];
    scaleFac = [votingResolution(2) / sz(2) votingResolution(1) / sz(1)];
    % compute the vote locations
    offsetsX = offsets(:,:,1);
    offsetsY = offsets(:,:,2);
    [rr,cc] = meshgrid(1:sz(1), 1:sz(2));
    locationsX = (offsetsX + cc') .* scaleFac(1);
    locationsY = (offsetsY + rr') .* scaleFac(2);

    % create votes
    votes(:,1) = locationsX(:); % vote locations
    votes(:,2) = locationsY(:);
    bbs = reshape(bbSizes, [], size(bbSizes,3));
    votes(:,3:4) = bsxfun(@times, bbs, scaleFac); % bounding box size
    votes(:,5) = 1 / (size(offsets,1) * size(offsets,2)); % weight dependent on the number of pixels in current scale
    votes(:,6:5+size(confidences,3)) = reshape(confidences, [], size(confidences,3)); % confidences
    
    % remove votes ouside the image
    outIndices = (votes(:,1) < 1) | (votes(:,2) < 1) | ...
        (votes(:,1) > votingResolution(2)) | (votes(:,2) > votingResolution(1)); % | ... %(bbs(:,1) < 256*0.05) | (bbs(:,2) < 256*0.05) | ... % votes for a extremely small bbox %(bbs(:,1) > 256*0.95) | (bbs(:,2) > 256*0.95);% | ... % votes for a extremely large bbox | %(abs(offsetsX(:)) > 0.5 * bbs(:,1)) | (abs(offsetsY(:)) > 0.5 * bbs(:,2)); % remove votes from outside their own boundingbox
    votes = removerows(votes, 'ind', outIndices);
    %votes(:,5) = 1 / size(votes,1); % weight dependent on the number of pixels in current scale
end