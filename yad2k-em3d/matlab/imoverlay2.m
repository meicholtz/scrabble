function varargout = imoverlay2(I,map,varargin)
%IMOVERLAY2 Superimpose a heatmap on an image.
%   IMOVERLAY2(I,MAP) generates a figure for visualizing slices of an
%   input volume with a superimposed mask. The size of both input arrays 
%   must be equal.
%
%   H = IMOVERLAY2(___) returns the image handles for the image and the
%   heatmap as outputs.
%
%   IMOVERLAY2(___,Name,Value) uses additional parameter name-value pairs.
%   Valid parameters include:
%
%       'Colormap'      String indicating which colormap to use. Valid
%                       strings include: 'parula', 'jet', 'hsv', 'hot',
%                       'cool', 'spring', 'summer', 'autumn', 'winter',
%                       'gray', 'bone', 'copper', 'pink', 'lines',
%                       'colorcude', 'prism', 'flag', and 'white'.
%
%                       Default: 'parula'
%
%       'Opacity'       Scalar indicating the alpha value for the
%                       superimposed mask. Must be >=0 and <=1. A value of
%                       0 indicates full transparency, while a value of 1
%                       indicates no transparency.
%
%                       Default: 0.4
%
%   See also IMOVERLAY.

% Copyright 2018 Matthew R. Eicholtz

% Default parameter values
default = struct(...
    'colormap','parula',...
    'opacity',0.4);

% Parse inputs
[cmap,opacity] = parseinputs(default,varargin{:});

rgb = label2rgb(round(255*map/max(map(:)))+0,cmap);
mask = all(rgb==255,3);
alphadata = ones(size(map))*opacity;
alphadata(mask) = 0;

h1 = imshow(I,'Border','tight');
hold on;
h2 = image(rgb,'AlphaData',alphadata);
hold off;

% Return the figure handle, if requested
if nargout>0
    varargout = {cat(2,h1,h2)};
end

end

%% Helper functions
function varargout = parseinputs(default,varargin)
%PARSEINPUTS Custom input parsing function.
    p = inputParser;
    
    p.addParameter('colormap',default.colormap,...
        @(x) ismember(x,{'parula','jet','hsv','hot','cool','spring','summer',...
        'autumn','winter','gray','bone','copper','pink','lines','colorcude',...
        'prism','flag','white'}));
    p.addParameter('opacity',default.opacity,...
        @(x) validateattributes(x,{'numeric'},{'scalar','nonempty','nonsparse','>=',0,'<=',1}));
    
    p.parse(varargin{:});
    
    [cmap,opacity] = struct2vars(p.Results);
    
    varargout = {cmap,opacity};
end

