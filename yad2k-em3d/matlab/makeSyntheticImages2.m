%makeSyntheticImages2
%   Create a dataset of 2D images containing basic shapes (e.g., circles,
%   squares, triangles).
%
%   The output is a directory of png images and text files. Each text file
%   contains bounding boxes for the corresponding image in the format
%   expected by yad2k:
%
%   class1 xmin1 ymin1 xmax1 ymax1
%   class2 xmin2 ymin2 xmax2 ymax2
%   ...
clear; clc; close all;

%% Set parameters
N = 100; %number of training images
minobj = 10; %minimum number of object per image
maxobj = 30; %maximum number of object per image
sz = [416 416]; %size of each training image

shapeparams0 = {'Color','white'}; %applies to all shapes; for insertShape()
radius = [16 16]; %[min max] allowable effective radius of shapes
opacity = [1 1]; %[min max] allowable opacity of shapes

noisy = false; %add noise or not
noiseparams = {'gaussian'};

boundaries = false; %allow objects to extend beyond boundaries (true) or not (false)
overlap = false; %allow overlapping objects (true) or not (false)

showresults = true; %do you want to visualize the results?
saveresults = true; %do you want to save the results (png and txt files)?

% saveroot = '\\sshfs\mve@deepblue\git\yad2k\images\test_shapes_2d_plain';
saveroot = 'C:\Users\meicholtz\Documents\python\yad2k-em3d\data';

seed = sum('test'+0);%floor(now); %use the current date as seed for random number generation

%% Check inputs
assert(diff(radius)>=0,'The minimum radius must be less than or equal to the maximum radius: [min,max]=[%d,%d]',radius(1),radius(2));
assert(diff(opacity)>=0,'The minimum opacity must be less than or equal to the maximum opacity: [min,max]=[%0.3f,%0.3f]',opacity(1),opacity(2));

%% Generate images
rng(seed); %set random number generator for reproducibility

I = zeros([sz,N],'uint8'); %initialize image volume
numobj = randi([minobj,maxobj],N,1); %randomly pick number of objects for each frame
bbox = cell(N,1); %bounding boxes as yad2k wants it [class xmin ymin xmax ymax]
rect = cell(N,1); %bounding boxes as rectangles [xmin ymin wid hei]
for ii=1:N
    status('making image %d of %d with %d objects',ii,N,numobj(ii));
    
    % Initialize synthetic image properties
    img = zeros(sz); %blank image
    mask = true(sz); %all positions are initially valid
    bbox{ii} = zeros(numobj(ii),5); %[class xmin ymin xmax ymax]
    rect{ii} = zeros(numobj(ii),4); %[xmin ymin wid hei]
    
    % Restrict objects from being too close to the boundary
    if ~boundaries
        mask([1 end],:) = false;
        mask(:,[1 end]) = false;
    end
    
    % Iteratively insert objbects
    for jj=1:numobj(ii)
        % Randomly pick effective radius
        r = radius(1)+diff(radius).*rand;
        maskj = ~imdilate(~mask,strel('square',ceil(2*r)));
        
        if any(maskj(:)) %check to make sure there are available locations to place an object
            % Randomly pick object location and opacity
            [y,x] = ind2sub(size(maskj),randsample(find(maskj),1));
            shapeparams = cat(2,shapeparams0,{'Opacity',opacity(1)+diff(opacity).*rand});
            
            % Randomly pick object class
            switch randi([0,2])
                case 0 %circle
                    img = insertShape(img,'FilledCircle',[x,y,r],shapeparams{:});
                    
                    bbox{ii}(jj,:) = [0, x-r, y-r, x+r, y+r];
                    rect{ii}(jj,:) = [x-r, y-r, 2*r, 2*r];
                    
                case 1 %square
                    wid = sqrt(2)*r;
                    hei = sqrt(2)*r;
                    img = insertShape(img,'FilledRectangle',[x-wid/2,y-hei/2,wid,hei],shapeparams{:});
                    
                    bbox{ii}(jj,:) = [1, x-wid/2, y-hei/2, x+wid/2, y+hei/2];
                    rect{ii}(jj,:) = [x-wid/2, y-hei/2, wid, hei];
                    
                case 2 %triangle
                    p1 = [x,y-r];
                    p2 = [x-r*sind(60), y+r*cosd(60)];
                    p3 = [x+r*sind(60), y+r*cosd(60)];
                    img = insertShape(img,'FilledPolygon',[p1,p2,p3],shapeparams{:});
                    
                    bbox{ii}(jj,:) = [2, x-r*sind(60), y-r, x+r*sind(60), y+r*cosd(60)];
                    rect{ii}(jj,:) = [x-r*sind(60), y-r, 2*r*sind(60), (1+cosd(60))*r];
            end
            
            % Update the allowable locations for new objects
            if ~overlap
                xmin = floor(max(1,x-r));
                xmax = ceil(min(sz(2),x+r));
                ymin = floor(max(1,y-r));
                ymax = ceil(min(sz(1),y+r));
                mask(ymin:ymax,xmin:xmax) = false;
%                 imoverlay3(rgb2gray(img),mask);
            end

        else
            bbox{ii}(jj:end,:) = [];
            rect{ii}(jj:end,:) = [];
            break;
        end
        
%         figure(1);
%         imshow(mask);
%         figure(2);
%         imshow(maskj);
%         pause
    end
    
    % Convert to grayscale uint8
    img = im2uint8(img(:,:,1));
    
    % Add noise, if requested
    if noisy
        img = imnoise(img,noiseparams{:});
    end
    
    % Add the grayscale image to the 3D volume of synthetic images
    I(:,:,ii) = img;
end
bbox = cellfun(@round,bbox,'uni',0);
rect = cellfun(@round,rect,'uni',0);

%% Show results, if requested
if showresults
    status('showing a sampling of synthetic images');
    imshow3(I(:,:,1:min(size(I,3),100)));
end

%% Save results, if requested
if saveresults
    status('saving results to file...');
    assert(~isempty(saveroot),'saveroot must be non-empty to save results');
    
    % Create directory, if needed
    if ~exist(saveroot,'dir')
        status('making new directory: %s',saveroot);
        mkdir(saveroot);
    end
    
    [~,folder] = fileparts(saveroot);
    
    % Write images to file
    for ii=1:N
        savefile = sprintf(['%0' num2str(floor(log10(N))+1) 'd.png'],ii);
        status('writing image %d of %d to file: %s',ii,N,savefile);
        imwrite(I(:,:,ii),fullfile(saveroot,savefile),'png');
    end
    
    % Write bounding boxes to file
    for ii=1:N
        savefile = sprintf(['%0' num2str(floor(log10(N))+1) 'd.txt'],ii);
        status('writing bounding boxes for image %d of %d to file: %s',ii,N,savefile);
        fid = fopen(fullfile(saveroot,savefile),'w');
        fprintf(fid,'%d %d %d %d %d\r\n',bbox{ii}');
        fclose(fid);
        pause(0.1);
    end
end

