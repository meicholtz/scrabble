function out = rectint3(A,B)
%RECTINT3 Rectangular prism intersection volume.
%   VOLUME = RECTINT3(A,B) returns the volume of intersection of the
%   rectangular prisms specified by position vectors A and B.  
%
%   If A and B each specify one rectangular prism, the output VOLUME is a 
%   scalar.
%
%   A and B can also be matrices, where each row is a position vector.
%   VOLUME is then a matrix giving the intersection of all rectangular 
%   prisms specified by A with all the rectangular prisms specified by B. 
%   That is, if A is M-by-6 and B is N-by-6, then VOLUME is an M-by-N 
%   matrix where VOLUME(P,Q) is the intersection volume of the rectangular
%   prisms specified by the Pth row of A and the Qth row of B.
%
%   Note: A position vector is a six-element vector [X,Y,Z,W,H,D], where
%   the point defined by (X,Y,Z) specifies one corner of the rectangular
%   prism, and (W,H,D) defines the size in units along the x-, y-, and
%   z-axes, respectively.
%
%   Class support for inputs A,B: 
%      float: double, single
%
%   See also RECTINT.

% Copyright 1984-2004 The MathWorks, Inc.
% Modified 2018 Matthew R. Eicholtz

leftA = A(:,1);
bottomA = A(:,2);
frontA = A(:,3);
rightA = leftA + A(:,4);
topA = bottomA + A(:,5);
backA = frontA + A(:,6);

leftB = B(:,1)';
bottomB = B(:,2)';
frontB = B(:,3)';
rightB = leftB + B(:,4)';
topB = bottomB + B(:,5)';
backB = frontB + B(:,6)';

numRectA = size(A,1);
numRectB = size(B,1);

leftA = repmat(leftA, 1, numRectB);
bottomA = repmat(bottomA, 1, numRectB);
frontA = repmat(frontA, 1, numRectB);
rightA = repmat(rightA, 1, numRectB);
topA = repmat(topA, 1, numRectB);
backA = repmat(backA, 1, numRectB);

leftB = repmat(leftB, numRectA, 1);
bottomB = repmat(bottomB, numRectA, 1);
frontB = repmat(frontB, numRectA, 1);
rightB = repmat(rightB, numRectA, 1);
topB = repmat(topB, numRectA, 1);
backB = repmat(backB, numRectA, 1);

out = (max(0, min(rightA, rightB) - max(leftA, leftB))) .* ...
    (max(0, min(topA, topB) - max(bottomA, bottomB))) .* ...
    (max(0, min(backA, backB) - max(frontA, frontB)));

end

