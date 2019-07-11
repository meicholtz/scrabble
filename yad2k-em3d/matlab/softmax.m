function y = softmax(x)
%SOFTMAX Softmax, or normalized exponential, function.
%   Y = SOFTMAX(X) applies the softmax function, defined as
%
%       f(x)_i = exp(x_i) / sum(exp(x_j))
%
%   to an input (X), which can either be a numeric array or a cell array of
%   numeric arrays. In the latter case, the output will also be a cell
%   array of numeric arrays.
%
%   Examples:
%   1) Apply the softmax to a numeric vector:
%
%       x = [1 2 3 4];
%       y = softmax(x);
%
%   yields the result
%
%       y = [0.0321    0.0871    0.2369    0.6439]

% Copyright 2017 Matthew R. Eicholtz

if iscell(x)
    y = cellfun(@(x) bsxfun(@rdivide,exp(x),sum(exp(x),2)),x,'uni',0);
else
    y = bsxfun(@rdivide,exp(x),sum(exp(x),2));
end

end

