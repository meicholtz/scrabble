%viewSampleLetters Show samples of manually labeled letters.
clear; clc; close all;

% Set relevant parameters
root = cd;
letter = 'A';
board = 'all'; % index of labeled board (or 'all')
sz = 825; % size of warped Scrabble board
maxcount = 1000; % how many letters to look at total

labelfiles = {'labels.txt','labels1.txt'};

seed = 4; % for random number generation

showresults = true;

% Get label files containing letters and corresponding positions
d = dir(fullfile(root,'labels'));
filenames = {d(3:end).name};
mask = cellfun('isempty', regexp(filenames, 'labels\d*.txt'));
filenames = filenames(mask);
N = length(filenames);

% Shuffle filenames
rng(seed);
order = randperm(N);
filenames = filenames(order);

% Parse label files for a given letter
letters = {};
for ii=14%:N
    status('%s (%d of %d)',filenames{ii},ii,N);
    % Read data from file
    fid = fopen(fullfile(root,'labels',filenames{ii}),'r');
    C = fscanf(fid,'%s %f %f %f %f',[5 Inf])';
    fclose(fid);
    
    % Find instances of letter
    pos = C(C(:,1)==letter+0,2:end); %[x1 y1 x2 y2]
    
    % Get original image
    imgfile = regexprep(filenames{ii},'.txt','.jpg');
    I = imread(fullfile(root,'data',imgfile));
%     I = imrotate(I,-90);
    [m,n,~] = size(I);
    if showresults
        figure(1);
        imshow(I);
    end
    
    % Get corners of Scrabble board from label files
    for jj=1:length(labelfiles)
        % Read data from file
        labelfile = labelfiles{jj};
        fid = fopen(fullfile(root,'labels',labelfile),'r');
        C = textscan(fid,'%s %f %f %f %f %f %f %f %f',[Inf 8]);
        names = C{1};
        allcorners = cell2mat(C(2:end));
        fclose(fid);
        
        % Check if current board is in the list
        ind = find(contains(upper(names),upper(imgfile)),1); % upper corrects for .jpg vs .JPG
        if ~isempty(ind)
            corners = allcorners(ind,:);
            break;
        end
    end
    if isempty(ind)
        error('Corner data not found for %s',imgfile);
    end
    if showresults
        hold on;
        scatter(n*corners(1:2:end),m*corners(2:2:end),50,'r','filled');
    end
    
    % Warp the original image
    x0 = n*corners(1:2:end);
    y0 = m*corners(2:2:end);
    x1 = [0,sz,0,sz];
    y1 = [0,0,sz,sz];
    tform = fitgeotrans([x0;y0]',[x1;y1]','projective');
    R = imref2d([sz,sz],[0,sz],[0,sz]);
    J = imwarp(I,tform,'OutputView',R);
    if showresults
        figure(2);
        imshow(J);
    end
    
    % Extract letters from board
    for jj=1:size(pos,1)
        rect = [pos(jj,1:2),pos(jj,3:4)-pos(jj,1:2)]*sz;
        K = imcrop(J,rect-[0 0 1 1]);
        K = imresize(K,[sz/15,sz/15]);
        letters = cat(1,letters,{K});
    end
    
    if length(letters)>=maxcount
        letters = letters(1:maxcount);
        break;
    end
end

% Show montage of letters (in batches of 100)
layers = ceil(maxcount/100);
T = zeros(sz/15*10,sz/15*10,3,layers,'uint8');
for ii=1:layers
    T(:,:,:,ii) = imtile(letters(100*(ii-1)+1:100*ii));
end
imshow3(T);

