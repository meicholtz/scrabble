%computeAveragePrecision3
%   Trying to compute AP similar to COCO version on test output for 3D
%   shapes.
%
%   Link: http://cocodataset.org/#detection-eval
%
%   See also evaluateDetectionPrecision3.

% Copyright 2018 Matthew R. Eicholtz
clearvars -except images boxes output;
clc; close all;

%% Set parameters
inputfile = '\\sshfs\mve@deepblue\git\yad2k\images\spheres_overfit_large.mat';
outputfile = '\\sshfs\mve@deepblue\git\yad2k\testing\derek.mat';
anchorfile = '\\sshfs\mve@deepblue\git\yad2k\model_data\yolo3d_anchors.txt';
classfile = '\\sshfs\mve@deepblue\git\yad2k\model_data\shape3d_classes.txt';

reload = true; %should we reload the output file? Set to false if using same output files as prior run. Will be faster.

thold_confidence = 0; %threshold for bounding box confidence
thold_nms = 1; %threshold for percent overlap during non-maxima suppression

num_pred_max = 50; %maximum number of predictions per image

%% Read data from file (created by python test script)
if ~exist('images','var') || reload
    status('reading network input from file (%s)...*',inputfile); tic;
    load(inputfile);
    status('*complete (%0.3f seconds)',toc);
end

if ~exist('output','var') || reload
    status('reading network output from file (%s)...*',outputfile); tic;
    load(outputfile);
    output = shiftdim(output,1);
    status('*complete (%0.3f seconds)',toc);
end
num_images = size(images,4);
[rows,cols,layers,channels,batches] = size(output);

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

%% Make groundTruthData table for evaluateDetectionPrecision3
status('making table of groundtruth data...*'); tic;
S = struct([]); %initialize structure to store ground truth bounding boxes
for ii=1:num_images
    % Get current 3D image volume and bounding boxes
    I = images(:,:,:,ii);
    bbox = boxes{ii};

    % Extract class label from bounding boxes
    label = bbox(:,1);
    label = label+1; %convert from 0-indexing
    
    % Create rectangle vectors for each bounding box
    topleft = bbox(:,2:4);
    bottomright = bbox(:,5:7);
    rect0 = [topleft, bottomright-topleft];
    
    % Make a structure separated by class
    for jj=1:num_classes
        S(ii).(classes{jj}) = {rect0(label==jj,:)};
    end
end
groundTruthData = struct2table(S);
clear I bbox position shape label rect0 S ii jj
status('*complete (%0.3f seconds)',toc);

%% Make detectionResults table for evaluateDetectionPrecision3
status('making table of detection results...*');
S(num_images) = struct('Boxes',[],'Scores',[],'Labels',[]);
for ii=1:num_images
    y = permute(reshape(output(:,:,:,:,ii),rows,cols,layers,[],num_anchors),[1 2 3 5 4]);
    pred = [];%zeros(rows*cols*layers*num_anchors,8); %each pred -> [x y z w h d confidence class]
    for row=1:rows
        for col=1:cols
            for layer=1:layers
                y0 = reshape(y(row,col,layer,:,:),num_anchors,[]);
                xyz = 32*bsxfun(@plus,1./(1+exp(-y0(:,1:3))),[col-1,row-1,layer-1]);
                whd = 32*anchors.*exp(y0(:,4:6));
                confidence = 1./(1+exp(-y0(:,7)));
                [~,whichclass] = max(softmax(y0(:,8:end)),[],2);
                pred = cat(1,pred,double([xyz,whd,confidence,whichclass]));
            end
        end
    end

    % Apply confidence threshold
    pred(pred(:,7)<thold_confidence,:) = [];

    % Apply non-maxima suppression
    pred = sortrows(pred,7,'descend');
    X = []; %store local optima data
    while ~isempty(pred) %if there are still bboxes, keep going
        X = [X; pred(1,:)];

        % Compute the percent overlap of the highest-scoring bounding box with 
        % each of the remaining bounding boxes
        bboxes = [pred(:,1:3)-pred(:,4:6)/2, pred(:,4:6)];
        volumeAB = rectint3(bboxes,bboxes(1,:)); %intersection volume
        volumeA = prod(pred(:,4:6),2);
        volumeB = prod(pred(1,4:6),2);
        overlapAB = volumeAB./volumeB(:);
        overlapBA = volumeAB./volumeA(:);

        % Discard bounding boxes with overlap > threshold
        mask = overlapAB(:)>=thold_nms | overlapBA(:)>=thold_nms;
        pred(mask,:) = [];
    end
    pred = X;
    clear X mask bboxes area* overlap*

    rect1 = [pred(:,1:3)-pred(:,4:6)/2, pred(:,4:6)];
    
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
status('computing average precision...');
thold = 0.5:0.05:0.95;
ap = zeros(num_classes,length(thold));
for ii=1:length(thold)
    [ap(:,ii),recall,precision] = evaluateDetectionPrecision3(detectionResults,groundTruthData,thold(ii));
end
AP = mean(mean(ap));
status('AP = %0.3f',AP);

