function evaluationInputValidation3(detectionResults, groundTruth, mfilename, checkScore, varargin)
% validate inputs for detection evaluation functions

if checkScore
    % verify detection result
    checkDetectionResultsTable(detectionResults, groundTruth, mfilename);
else
    % verify bounding boxes
    checkBoundingBoxTable(detectionResults, groundTruth, mfilename);
end

% verify ground truth table
checkGroundTruthTable(groundTruth, height(detectionResults), mfilename);

% check additional inputs
if ~isempty(varargin)
    validateattributes(varargin{1}, {'single','double'}, ...
        {'real','scalar','nonsparse','>=',0,'<=',1}, mfilename, 'threshold');
end

end

%==========================================================================
function checkDetectionResultsTable(detectionResults, groundTruth, mfilename)

    validateattributes(detectionResults, {'table'},{'nonempty'}, mfilename, 'detectionResults');

    if width(detectionResults) < 2
        error(message('vision:ObjectDetector:detectionResultsTableWidthLessThanTwo'));
    end
    
    ismulcls = (width(detectionResults) > 2);
    if ismulcls
        classNames = categorical(groundTruth.Properties.VariableNames);
        msg = '{';
        for n = 1:numel(classNames)
            msg = [msg char(classNames(n)) ','];
        end
        msg = [msg(1:end-1) '}'];
    end

    for i = 1:height(detectionResults)
        % check bounding boxes
        try
            if ~isempty(detectionResults{i,1}{1})
                bbox = detectionResults{i,1}{1};
                validateattributes(bbox, ...
                    {'numeric'},{'real','nonsparse','2d', 'size', [NaN, 6]});
                if (any(bbox(:,4)<=0) || any(bbox(:,5)<=0) || any(bbox(:,6)<=0))
                    error(message('vision:visionlib:invalidBboxHeightWidth'));
                end
            end
        catch ME
            error(message('vision:ObjectDetector:invalidBboxInDetectionTable', i, ME.message(1:end-1)));
        end
        
        % check scores
        try
            if ~isempty(detectionResults{i,2}{1})
                validateattributes(detectionResults{i,2}{1},{'single','double'},...
                    {'vector','real','nonsparse','numel',size(detectionResults{i,1}{1},1)});
            end
        catch ME
            error(message('vision:ObjectDetector:invalidScoreInDetectionTable', i, ME.message(1:end-1)));
        end
        
        % for multi-class detection, check labels
        if ismulcls
            try
                if ~isempty(detectionResults{i,3}{1})
                    validateattributes(detectionResults{i,3}{1},{'categorical'},...
                        {'vector','numel',size(detectionResults{i,1}{1},1)});
                end
            catch ME
                error(message('vision:ObjectDetector:invalidLabelInDetectionTable', i, ME.message(1:end-1)));
            end
            
            if ~isempty(detectionResults{i,3}{1})
                labels = categories(detectionResults{i,3}{1});
                if any(isundefined(detectionResults{i,3}{1}))||~all(ismember(labels,classNames))
                    error(message('vision:ObjectDetector:undefinedLabelInDetectionTable', i, msg));
                end
            end
        end
        
    end  
end

%==========================================================================
function checkGroundTruthTable(groundTruthData,numExpectedRows,mfilename)

    validateattributes(groundTruthData,{'table'},...
        {'nonempty','nrows',numExpectedRows},mfilename,'groundTruthData');

    for n=1:width(groundTruthData)
        for ii=1:numExpectedRows
            try
                if ~isempty(groundTruthData{ii,n}{1})
                    bbox = groundTruthData{ii,n}{1};
                    validateattributes(bbox,{'numeric'},{'real','nonsparse','2d','size',[NaN,6]});
                    if (any(bbox(:,4)<=0) || any(bbox(:,5)<=0) || any(bbox(:,6)<=0))
                        error(message('vision:visionlib:invalidBboxHeightWidth'));
                    end
                end
            catch ME
                error(message('vision:ObjectDetector:invalidBboxInTrainingDataTable',ii,n,ME.message(1:end-1)));
            end        
        end  
    end
end 

%==========================================================================
function checkBoundingBoxTable(boundingBoxes, groundTruth, mfilename)

    validateattributes(boundingBoxes, {'table'},{'nonempty'}, mfilename, 'boundingBoxes');
    
    ismulcls = (width(boundingBoxes) >= 2);
    if ismulcls
        classNames = categorical(groundTruth.Properties.VariableNames);
        msg = '{';
        for n = 1:numel(classNames)
            msg = [msg char(classNames(n)) ','];
        end
        msg = [msg(1:end-1) '}'];
    end

    for i = 1:height(boundingBoxes)
        bbox = boundingBoxes{i, 1}{1};
        % check bounding boxes
        try
            if ~isempty(bbox)
                validateattributes(bbox, ...
                    {'numeric'},{'real','nonsparse','2d', 'size', [NaN, 6]});
                if (any(bbox(:,4)<=0) || any(bbox(:,5)<=0) || any(bbox(:,6)<=0))
                    error(message('vision:visionlib:invalidBboxHeightWidth'));
                end                
            end
        catch ME
            error(message('vision:ObjectDetector:invalidBboxInBboxTable', i, ME.message(1:end-1)));
        end
                
        % for multi-class, check labels
        if ismulcls
            try
                if ~isempty(bbox)
                    validateattributes(boundingBoxes{i, 2}{1},{'categorical'},...
                        {'vector','numel',size(bbox,1)});
                end
            catch ME
                error(message('vision:ObjectDetector:invalidLabelInBboxTable', i, ME.message(1:end-1)));
            end
            
            if ~isempty(bbox)
                labels = categories(boundingBoxes{i, 2}{1});
                if any(isundefined(boundingBoxes{i, 2}{1}))||~all(ismember(labels, classNames))
                    error(message('vision:ObjectDetector:undefinedLabelInBboxTable', i, msg));
                end
            end
        end
        
    end  
end