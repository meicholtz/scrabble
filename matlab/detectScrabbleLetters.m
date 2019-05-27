% detectScrabbleLetters
clear; clc; close all;

% Setup processing parameters
root = fullfile(cd,'data');  % root directory where images are stored
ind = 10218;  % index of image
dark = true;  % are the letters dark or light relative to the background?
scale = 2;  % how much to resize the image
thold = [2000 20000]; %[min, max] number of pixels in the object

% Get image files
d = dir(root);
filenames = fullfile(root,{d(3:end).name}');

% Show an image
I = imread(filenames{ind});
figure(1);
imshow(I);
title(ind);

% Resize the image to help OCR
I = imresize(I,scale);
% figure;
% imshow(I);

% Binarize image
bw = im2bw(I,graythresh(I));
figure;
imshow(bw);
if ~dark; bw = ~bw; end

% Connected component analysis to remove unwanted stuff
cc = bwconncomp(~bw);
numpixels = cellfun(@numel, cc.PixelIdxList);
bad = find(numpixels < thold(1) | numpixels > thold(2));
for ii=bad(:)'
    bw(cc.PixelIdxList{ii}) = 1;
end
figure;
imshow(bw);

% Region properties to filter out objects that are missing critical
% properties of letters
cc = bwconncomp(~bw);
stats = regionprops(cc,'BoundingBox','EquivDiameter');
bbox = reshape([stats(:).BoundingBox],4,[])';
ed = [stats(:).EquivDiameter];
% bad = find(prod(bbox(:,3:4),2)<7500);
bad = find(bbox(:,4)<80);
for ii=bad(:)'
    bw(cc.PixelIdxList{ii}) = 1;
end
figure;
imshow(bw);

% Get ROIs to help OCR
bw2 = imdilate(~bw,strel('disk',4)); % dilate the letters for ROI
s = regionprops(bw2,'BoundingBox');
roi = vertcat(s(:).BoundingBox);

% Apply OCR
bw3 = imerode(~bw, strel('square',1)); % thin 'blocky' letters a bit
txt = ocr(bw3,roi,'TextLayout','Word','CharacterSet','A':'Z');
letter = cell(1,numel(txt));
for ii=1:numel(txt)
    letter{ii} = deblank(txt(ii).Text);
end

% Remove unrecognized letters
mask = cellfun('isempty',letter);
roi(mask,:) = [];
letter(mask) = [];

%% Show results
% J = insertObjectAnnotation(im2uint8(bw),'Rectangle',roi,letter);
figure;
imshow(I);
hold on;
txtparams = {'FontSize',6,'Color','k','FontWeight','bold'};
for ii=1:length(letter)
    pos = [roi(ii,1)-80, roi(ii,2)-100, 110, 110];
    rectangle('Position',pos,'FaceColor',[255,200,200]/255,'EdgeColor','r');
    text(roi(ii,1)-60,roi(ii,2)-60,letter(ii),txtparams{:});
end
hold off;


