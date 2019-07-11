function I = imsphere(sz,x,y,z,r,opacity)
%IMSPHERE Generate a 3D image volume containing a sphere.
%   I = IMSPHERE(SZ,X,Y,Z,R)

% Copyright 2018 Matthew R. Eicholtz

I = zeros(sz); %initialize image volume

[X,Y,Z] = meshgrid(1:sz(1),1:sz(2),1:sz(3)); %generate voxel coordinates

d = sqrt((X-x).^2 + (Y-y).^2 + (Z-z).^2); %compute distance to center of sphere

% Interior voxels of sphere must be <= radius of sphere
mask = d<=r; 
I(mask) = 1;

% Compute exponential decay approximation of voxel intensity for 'halo'
% near sphere boundary
halo = d-r;
thold = 0.1;
mask = halo>0 & halo<thold*r;
c = -log(0.001)/(thold*r);
I(mask) = exp(-c*halo(mask));

I = I*opacity;

end
