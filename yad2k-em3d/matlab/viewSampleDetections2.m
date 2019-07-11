%viewSampleDetections2
%   Working on Keras model output for YOLO on 2D synthetic shapes.

% Copyright 2018 Matthew R. Eicholtz
clearvars -except images boxes output;
clc; close all;

inputpath = '\\sshfs\mve@deepblue\git\yad2k\images\test_shapes_2d_plain';
outputfile = '\\sshfs\mve@deepblue\git\yad2k\testing\test_shapes_2d_plain.mat';
anchorfile = '\\sshfs\mve@deepblue\git\yad2k\model_data\yolo_anchors.txt';
classfile = '\\sshfs\mve@deepblue\git\yad2k\model_data\shape_classes.txt';

reload = true; %should we reload the output file? Set to false if using same output files as prior run. Will be faster.

whichsample = 2;

thold = 0.5; %threshold for bounding box confidence
iou = 0.5; %threshold for intersection over union during non-maxima suppression

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

%% Ground truth
% Get current image and bounding boxes
I = images(:,:,whichsample);
bbox = boxes{whichsample};

% Extract class label from bounding boxes
label = bbox(:,1);
label = label+1; %convert from 0-indexing

% Create rectangle vectors for each bounding box
topleft = bbox(:,2:3);
bottomright = bbox(:,4:5);
rect0 = [topleft, bottomright-topleft];

% Show image with bounding boxes overlayed
h1 = figure(1); imshow(I,'Border','tight');
h1.Name = 'ground truth';
u = unique(label);
for ii=1:length(u)
    rectangles(rect0(label==u(ii),:),'EdgeColor',colors(ii,:),'Tag',tags{ii});
end

%% Predictions
% figure; imagesc(output(:,:,1,1));
y = permute(reshape(output(:,:,:,whichsample),rows,cols,[],num_anchors),[1 2 4 3]);
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
pred(pred(:,5)<thold,:) = [];

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
    mask = overlapAB(:)>=iou | overlapBA(:)>=iou;
    pred(mask,:) = [];
end
pred = X;
clear X mask bboxes area* overlap*

h2 = figure(2); imshow(I,'Border','tight');
h2.Name = 'predictions';
rect1 = [pred(:,1)-pred(:,3)/2, pred(:,2)-pred(:,4)/2, pred(:,3:4)];
% for ii=1:length(u)
%     rectangles(rect1(pred(:,end)==ii,:),'EdgeColor',colors(ii,:),'Tag',tags{ii},'LineWidth',pred(:,5));
% end
for ii=1:size(rect1,1)
    clr = colors(pred(ii,end),:);
    tag = tags{pred(ii,end)};
    linewidth = 1.5*pred(ii,5)^2;
    rectangles(rect1(ii,:),'EdgeColor',clr,'Tag',tag,'LineWidth',linewidth);
end

