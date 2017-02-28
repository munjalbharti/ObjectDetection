function [bbox, class, prob] = mergeCenters(bbox, class, prob)

ovmin = 0.5;
weak = [];
k = 1;
numBB = numel(class);

for i = 1:numBB
    %Candidate bounding box
    bb_candidate = bbox(:,i);
    %Compare to the rest of the bounding boxes
    for j = i+1:numBB
        bb = bbox(:,j);
        bi = [max(bb(1),bb_candidate(1)) ; max(bb(2),bb_candidate(2)) ; min(bb(3),bb_candidate(3)) ; min(bb(4),bb_candidate(4))];
        iw = bi(3)-bi(1)+1;
        ih = bi(4)-bi(2)+1;
        if iw>0 && ih>0
            % compute overlap as IoU
            ua = (bb(3)-bb(1)+1)*(bb(4)-bb(2)+1) + (bb_candidate(3)-bb_candidate(1)+1)*(bb_candidate(4)-bb_candidate(2)+1) - iw*ih;
            ov = iw*ih / ua;  %overlap bb_i, bb_j
            if ((ov > ovmin) && (class(i) == class(j))) % ||((ov > 0.7))
                if prob(i) > prob(j)
                    weak(k) = j;
                else
                    weak(k) = i;
                end
                k = k + 1;
            end
        end
    end
end

if ~isempty(weak)
    fprintf('(Rejecting %d box(es))', numel(weak));
    bbox = (removerows(bbox', 'ind', weak))';
    class(weak) = [];
    prob(weak) = [];
end

