function [labels, falseNegative, assignment] = ...
    assignDetectionsToGroundTruth3(bbox, groundTruthBoxes, threshold, scores)
% Assigns ground truth labels to bounding boxes. 
% Input:
%  bbox and scores are detector outputs
%  groundTruth are ground truth bboxes
%  threshold is the overlap threshold for assignment
%
% Output:
%  labels is the "label" assigned to each detection.
%  falseNegative is the number of false negatives
%  assigment is the ground truth box that the detection is assigned to.

% Copyright 2016-2017 The MathWorks, Inc.
% Modified 2018 Matthew R. Eicholtz

% arbitrary background label
bgLabel = 0;

% foreground object label
fgLabel = 1;

if ~isempty(scores)
    % sort detections by score
    [~, idx] = sort(scores, 'descend');
    bbox = bbox(idx,:);
end

iou = bboxOverlapRatio3(bbox, groundTruthBoxes, 'union');

numDetections = size(bbox,1);
labels = repelem(bgLabel,numDetections, 1);

% track which gt a detection is matched to
assignment = zeros(numDetections,1);

% track which gt was detected
wasDetected = false(size(groundTruthBoxes,1),1);

% greedily assign detections to ground truth
for i = 1:numDetections        
    
    [v, assignment(i)] = max( iou(i,:) );
        
    if v >= threshold                
        
        labels(i) = fgLabel;   
                
        wasDetected(assignment(i)) = true;
        
        % remove gt from future consideration. This penalizes multiple
        % detections that overlap one ground truth box, i.e. it is considered
        % a false positive.
        iou(:,assignment(i)) = -inf;  
    else
        % detection not assigned to any ground truth
        labels(i) = bgLabel;
        assignment(i) = NaN;
    end
end

% return false negatives, these are ground detections that were missed. 
falseNegative = sum(~wasDetected);

if ~isempty(scores)
    % unsort to keep original order of input.
    unsort(idx) = 1:numel(scores);
    labels = labels(unsort);
    assignment = assignment(unsort);
end

