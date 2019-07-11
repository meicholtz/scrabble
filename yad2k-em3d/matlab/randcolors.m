function colors = randcolors(N,ignore)
%RANDCOLORS Generate set of perceptually distinct pseudorandom colors.
%   COLORS = RANDCOLORS(N)
%   COLORS = RANDCOLORS(N,IGNORE)

% Copyright 2018 Matthew R. Eicholtz

%% Parse inputs
if nargin<2 %user did not provide any colors to ignore
    ignore = [1 1 1]; %default is to ignore white colors
elseif iscell(ignore) %user provided multiple colors to ignore
    ignore = cell2mat(cellfun(@checkcolor,ignore(:),'uni',0));
else %user provided one color to ignore
    ignore = checkcolor(ignore);
end

%% Make candidate RGB triples
x = linspace(0,1,20);
[R,G,B] = ndgrid(x,x,x);
rgb = [R(:), G(:), B(:)];
assert(N<size(rgb,1),'%s cannot generate %d random distinct colors',mfilename,N);

%% Convert to Lab color space
C = makecform('srgb2lab');
lab = applycform(rgb,C);
lab0 = applycform(ignore,C);

%% Select requested number of colors
d = min(pdist2(lab,lab0),[],2); %distance of each color to closest ignored color
colors = zeros(N,3); %initialize output array
prev = lab0(end,:); %initialize "previous" color
for ii=1:N
    d = min(d,pdist2(lab,prev));
    [~,ind] = max(d);
    colors(ii,:) = rgb(ind,:);
    prev = lab(ind,:);
end

end

%% Helper functions
function y = checkcolor(x)
    if ischar(x)
        y = str2rgb(x);
    else
        y = x;
    end
end
