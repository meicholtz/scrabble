%computeAveragePrecision2
%   Trying to compute AP similar to COCO version on test output for 2D
%   shapes.
%
%   Link: http://cocodataset.org/#detection-eval
%
%   See also evaluateDetectionPrecision.

% Copyright 2018 Matthew R. Eicholtz
clearvars -except images boxes output;
clc; close all;

%% Set parameters
inputpath = '\\sshfs\mve@deepblue\git\yad2k\images\test_shapes_2d_plain';
outputfile = '\\sshfs\mve@deepblue\git\yad2k\testing\test_shapes_2d_plain.mat';
anchorfile = '\\sshfs\mve@deepblue\git\yad2k\model_data\yolo_anchors.txt';
classfile = '\\sshfs\mve@deepblue\git\yad2k\model_data\shape_classes.txt';

reload = true; %should we reload the output file? Set to false if using same output files as prior run. Will be faster.

thold_confidence = 0; %threshold for bounding box confidence
thold_nms = 1; %threshold for percent overlap during non-maxima suppression

num_pred_max = 50; %maximum number of predictions per image (affects the speed of computing AP)

%% Read data from file (created by python test script)
if ~exist('images','var') || reload
    status('reading network input from directory (%s)...*',inputpath); tic;
    d = dir(fullfile(inputpath,'*.png'));
    filenames = {d(:).name}';
    images = [];
    for ii=1:length(filenames)
        status('image %d of %d',ii,length(filenames));
        images = cat(3,images,imread(fullfile(inputpath,filenames{ii})));
    end
    
    d = dir(fullfile(inputpath,'*.txt'));
    filenames = {d(:).name}';
    boxes = {};
    for ii=1:length(filenames)
        status('boxes %d of %d',ii,length(filenames));
        boxes = cat(1,boxes,{dlmread(fullfile(inputpath,filenames{ii}),' ')});
    end
    status('*complete (%0.3f seconds)',toc);
end

if ~exist('output','var') || reload
    status('reading network output from file (%s)...*',outputfile); tic;
    load(outputfile);
    output = shiftdim(output,1);
    status('*complete (%0.3f seconds)',toc);
end
num_images = size(images,3);
[rows,cols,channels,batches] = size(output);

if ~exist('anchors','var') || reload
    status('reading anchors from file (%s)...*',anchorfile); tic;
    anchors = dlmread(anchorfile,' ');
    status('*complete (%0.3f seconds)',toc);
end
num_anchors = size(anchors,1);

if ~exist('classes','var') || reload
    status('reading classes from file (%s)...*',classfile); tic;
    fid = fopen(classfile,'r');
    classes = textscan(fid,'%s');
    classes = classes{1};
    fclose(fid);
    status('*complete (%0.3f seconds)',toc);
end
num_classes = length(classes);

tags = classes; %tags for rectangle objects
colors = randcolors(num_classes);

%% Make groundTruthData table for evaluateDetectionPrecision
status('making table of groundtruth data...*'); tic;
S = struct([]); %initialize structure to store ground truth bounding boxes
for ii=1:num_images
    % Get current image and bounding boxes
    I = images(:,:,ii);
    bbox = boxes{ii};

    % Extract class label from bounding boxes
    label = bbox(:,1);
    label = label+1; %convert from 0-indexing
    
    % Create rectangle vectors for each bounding box
    topleft = bbox(:,2:3);
    bottomright = bbox(:,4:5);
    rect0 = [topleft, bottomright-topleft];
    
    % Make a structure separated by class
    for jj=1:num_classes
        S(ii).(classes{jj}) = rect0(label==jj,:);
    end
end
groundTruthData = struct2table(S);
clear I bbox position shape label rect0 S ii jj
status('*complete (%0.3f seconds)',toc);

%% Make detectionResults table for evaluateDetectionPrecision
status('making table of detection results...*'); tic;
S(num_images) = struct('Boxes',[],'Scores',[],'Labels',[]);
for ii=1:num_images
    y = permute(reshape(output(:,:,:,ii),rows,cols,[],num_anchors),[1 2 4 3]);
    pred = [];%zeros(rows*cols*num_anchors,6); %each pred -> [x y w h confidence class]
    for row=1:rows
        for col=1:cols
            y0 = reshape(y(row,col,:,:),num_anchors,[]);
            xy = 32*bsxfun(@plus,1./(1+exp(-y0(:,1:2))),[col-1,row-1]);
            wh = 32*anchors.*exp(y0(:,3:4));
            confidence = 1./(1+exp(-y0(:,5)));
            [~,whichclass] = max(softmax(y0(:,6:8)),[],2);
            pred = cat(1,pred,double([xy,wh,confidence,whichclass]));
        end
    end

    % Apply confidence threshold
    pred(pred(:,5)<thold_confidence,:) = [];

    % Apply non-maxima suppression
    pred = sortrows(pred,5,'descend');
    X = []; %store local optima data
    while ~isempty(pred) %if there are still bboxes, keep going
        X = [X; pred(1,:)];

        % Compute the percent overlap of the highest-scoring bounding box with 
        % each of the remaining bounding boxes
        bboxes = [pred(:,1)-pred(:,3)/2, pred(:,2)-pred(:,4)/2, pred(:,3), pred(:,4)];
        areaAB = rectint(bboxes,bboxes(1,:)); %intersection area
        areaA = pred(:,3).*pred(:,4);
        areaB = pred(1,3).*pred(1,4);
        overlapAB = areaAB./areaB(:);
        overlapBA = areaAB./areaA(:);

        % Discard bounding boxes with overlap > threshold
        mask = overlapAB(:)>=thold_nms | overlapBA(:)>=thold_nms;
        pred(mask,:) = [];
    end
    pred = X;
    clear X mask bboxes area* overlap*

    rect1 = [pred(:,1:2)-pred(:,3:4)/2, pred(:,3:4)];
    
    N = min(num_pred_max,size(rect1,1));
    
    % Store results in required structure for evaluateDetectionPrecision
    S(ii).Boxes = {rect1(1:N,:)};
    S(ii).Scores = {pred(1:N,end-1)};
    S(ii).Labels = {categorical(classes(pred(1:N,end)))};
end
detectionResults = struct2table(S);
clear ii row col y0 xy wh confidence whichclass S
status('*complete (%0.3f seconds)',toc);

%% Compute AP
status('computing average precision...*'); tic;
thold = 0.5:0.05:0.95;
ap = zeros(num_classes,length(thold));
for ii=1:length(thold)
    [ap(:,ii),recall,precision] = evaluateDetectionPrecision(detectionResults,groundTruthData,thold(ii));
end
status('*complete (%0.3f seconds)',toc);
AP = mean(mean(ap));
status('AP = %0.3f',AP);

