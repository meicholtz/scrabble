%viewSampleDetections3
%   Working on Keras model output for YOLO on 3D synthetic shapes.
clearvars -except images boxes output;
clc; close all;

inputfile = '\\sshfs\mve@deepblue\git\yad2k\images\spheres_overfit_large.mat';
outputfile = '\\sshfs\mve@deepblue\git\yad2k\testing\derek.mat';
anchorfile = '\\sshfs\mve@deepblue\git\yad2k\model_data\yolo3d_anchors.txt';
classfile = '\\sshfs\mve@deepblue\git\yad2k\model_data\shape3d_classes.txt';

reload = true; %should we reload the output file? Set to false if using same output files as prior run. Will be faster.

whichsample = 1;

thold = 0.5; %threshold for bounding box confidence
iou = 0.5; %threshold for intersection over union during non-maxima suppression

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

%% Ground truth
% Get current 3D image volume and bounding boxes
I = images(:,:,:,whichsample);
bbox = boxes{whichsample};

% Extract class label from bounding boxes
label = bbox(:,1);
label = label+1; %convert from 0-indexing

% Create rectangle vectors for each bounding box
topleft = bbox(:,2:4);
bottomright = bbox(:,5:7);
rect0 = [topleft, bottomright-topleft];

video = VideoWriter('spheres_groundtruth.avi');
video.FrameRate = 10; %frames per second (in the output video)
open(video);
figure;
scale = 3;
for frame=1:size(I,3)
    imshow(imresize(I(:,:,frame),scale));
    mask = frame-rect0(:,3)>=0 & rect0(:,3)+rect0(:,6)-frame>=0; %which bounding boxes should be visible at the current layer?
    
    rect0_temp = rect0(mask,[1,2,4,5]);
    if ~isempty(rect0_temp)
        for ii=1:length(u)
            rectangles(rect0_temp(label(mask)==u(ii),:)*scale,'EdgeColor',colors(ii,:),'Tag',tags{ii});
        end
    end
    pause(0.05);
    writeVideo(video,getframe);
end
close(video);

return
%% Show predictions
% figure; imagesc(output(:,:,1,1));
y = permute(reshape(output(:,:,:,:,whichsample),rows,cols,layers,[],num_anchors),[1 2 3 5 4]);
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
pred(pred(:,7)<thold,:) = [];

% Apply non-maxima suppression
pred = sortrows(pred,7,'descend');
% X = []; %store local optima data
% while ~isempty(pred) %if there are still bboxes, keep going
%     X = [X; pred(1,:)];
%     
%     % Compute the percent overlap of the highest-scoring bounding box with 
%     % each of the remaining bounding boxes
%     bboxes = [pred(:,1)-pred(:,3)/2, pred(:,2)-pred(:,4)/2, pred(:,3), pred(:,4)];
%     areaAB = rectint(bboxes,bboxes(1,:)); %intersection area
%     areaA = pred(:,3).*pred(:,4);
%     areaB = pred(1,3).*pred(1,4);
%     overlapAB = areaAB./areaB(:);
%     overlapBA = areaAB./areaA(:);
% 
%     % Discard bounding boxes with overlap > threshold
%     mask = overlapAB(:)>=iou | overlapBA(:)>=iou;
%     pred(mask,:) = [];
% end
% pred = X;
% clear X mask bboxes area* overlap*

video = VideoWriter('spheres_overfit.avi');
video.FrameRate = 10; %frames per second (in the output video)
open(video);
figure;
scale = 3;
rect1 = [pred(:,1)-pred(:,4)/2, pred(:,2)-pred(:,5)/2, pred(:,3)-pred(:,6)/2, pred(:,4:6)];
for frame=1:size(I,3)
    imshow(imresize(I(:,:,frame),scale));
    mask = frame-rect1(:,3)>=0 & rect1(:,3)+rect1(:,6)-frame>=0; %which bounding boxes should be visible at the current layer?
    
    rect1_temp = rect1(mask,[1,2,4,5]);
    pred_temp = pred(mask,:);
    if ~isempty(rect1_temp)
        for ii=1:size(rect1_temp)
            clr = colors(pred_temp(ii,end),:);
            tag = tags{pred_temp(ii,end)};
            linewidth = 1.5*pred_temp(ii,7)^2;
            rectangles(rect1_temp(ii,:)*scale,'EdgeColor',clr,'Tag',tag,'LineWidth',linewidth);
        end
%         for ii=1:length(u)
%             rectangles(rect1_temp(label(mask)==u(ii),:)*3,'EdgeColor',colors(ii,:),'Tag',tags{ii});
%         end
    end
    pause(0.05);
    writeVideo(video,getframe);
end
close(video);

