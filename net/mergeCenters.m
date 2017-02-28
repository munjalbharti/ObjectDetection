function [bbox, class, prob,contri_x,contri_y,centers_x,centers_y ] = mergeCenters(bbox, class,prob,contri_x,contri_y,centers_x,centers_y )

  %Result=struct('min_x',[],'min_y',[],'max_x',[],'max_y',[],'contri_x',{},'contri_y',{},'avg_prob1',[],'per_class_pixels1',[],'avg_prob2',[],'per_class_pixels2',[],'contri_by_total',[],'classes',[]);
   
 % return ;
ovmin = 0.5;
weak = [];
k = 1;
numBB = numel(class);

for i = 1:numBB
    %Candidate bounding box
    bb_candidate = bbox(:,i);
   % bb_candidate=[Result.min_x(i);Result.min_y(i);Result.max_x(i);Result.max_y(i)];
    %Compare to the rest of the bounding boxes
    for j = i+1:numBB
       % bb = [Result.min_x(j);Result.min_y(j);Result.max_x(j);Result.max_y(j)];
        bb=bbox(:,j);
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
    contri_x(weak)=[];
    contri_y(weak)=[];
    centers_x(weak)=[];
    centers_y(weak)=[];
end

