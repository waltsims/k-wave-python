function [lin_ind, is, js, ks] = tolStar(tolerance, kgrid, point, debug)
%TOLSTAR Compute spatial extent of BLI above given tolerance.
%
% DESCRIPTION:
%     tolStar computes a set of subscripts corresponding to locations where
%     the magnitude of a multidimensional sinc function has not yet decayed
%     below a specified tolerance. These subscripts are relative to the 
%     nearest grid node to a given Cartesian point. Note, a sinc
%     approximation is used for the band-limited interpolant (BLI).
%
% USAGE:
%     [lin_ind, x_ind] = tolStar(tolerance, kgrid, point)
%     [lin_ind, x_ind, y_ind] = tolStar(tolerance, kgrid, point)
%     [lin_ind, x_ind, y_ind, z_ind] = tolStar(tolerance, kgrid, point)
%
% INPUTS:
%     tolerance             - Scalar value controlling where the spatial
%                             extent of the BLI at each point is trunctated
%                             as a  portion of the maximum value.
%     kgrid                 - Object of the kWaveGrid class defining the
%                             Cartesian and k-space grid fields.
%     point                 - Cartesian coordinates defining location of
%                             the BLI.
% 
% OUTPUTS:
%     lin_ind               - Linear indices following MATLAB's column-wise
%                             linear matrix index ordering.
%     x_ind, y_ind, z_ind   - Grid indices (y and z only returned in 2D and
%                             3D).
%     
% ABOUT:
%     author                - Elliott Wise and Bradley Treeby
%     date                  - 16th March 2017
%     last update           - 31st July 2021
%
% This function is part of the k-Wave Toolbox (http://www.k-wave.org)
% Copyright (C) 2017-2021 Elliott Wise and Bradley Treeby

% This file is part of k-Wave. k-Wave is free software: you can
% redistribute it and/or modify it under the terms of the GNU Lesser
% General Public License as published by the Free Software Foundation,
% either version 3 of the License, or (at your option) any later version.
% 
% k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
% more details. 
% 
% You should have received a copy of the GNU Lesser General Public License
% along with k-Wave. If not, see <http://www.gnu.org/licenses/>.

% debug option
if nargin < 4
    debug = true;
    disp(debug);
end

% tolerance value to decide if a specified point is on grid or not as a
% proportion of kgrid.dx (within a thousandth)
ongrid_threshold = kgrid.dx * 1e-3;

% store a canonical star for use whenever tol remains unchanged
persistent tol subs0

% compute a new canonical star if the tolerance changes, where the star
% gives indices relative to [0 0 0]
if isempty(tol) || (tolerance ~= tol) || (length(subs0) ~= kgrid.dim)
% (no subs yet)    (new tolerance)       (new dimensionality)
    
    % assign tolerance value
    tol = tolerance;
    
    % the on-axis decay of the BLI is given by dx/(pi*x), thus the grid 
    % point at which the BLI has decayed to the tolerance value can be 
    % calculated by 1/(pi*tolerance)
    decay_subs = ceil(1/(pi*tol));
    
    % compute grid indices along axes
    lin_ind = -decay_subs:decay_subs;          
    
    % replicate grid vectors
    switch kgrid.dim
        case 1
            is0 = lin_ind;
        case 2
            [is0, js0] = ndgrid(lin_ind, lin_ind);
        case 3
            [is0, js0, ks0] = ndgrid(lin_ind, lin_ind, lin_ind);
    end
    
    % only keep grid indices where the BLI is above the tolerance value
    % (i.e., within the star)
    switch kgrid.dim
        case 1
            subs0 = {is0};
        case 2
            instar = logical(abs(is0 .* js0) <= decay_subs);
            is0 = is0(instar);
            js0 = js0(instar);
            subs0 = {is0, js0};
        case 3
            instar = logical(abs(is0.*js0.*ks0) <= decay_subs);
            is0 = is0(instar);
            js0 = js0(instar);
            ks0 = ks0(instar);
            subs0 = {is0, js0, ks0};
    end
    
    % plot in debug mode
    if debug && kgrid.dim == 2
        figure; 
        imagesc(lin_ind, lin_ind, instar);
        title('Canonical Star');
        xlabel('Grid points');
        ylabel('Grid points');
        axis image;
    end
    
end

% assign canonical star
is = subs0{1};
if kgrid.dim > 1
    js = subs0{2};
else
    js = [];
end
if kgrid.dim > 2
    ks = subs0{3};
else
    ks = [];
end

% if the point lies on the grid, then truncate the canonical star as the
% BLI will evaluate to zero for all grid points in that direction
[x_closest, x_closest_ind] = findClosest(kgrid.x_vec, point(1));
if abs(x_closest - point(1)) < ongrid_threshold
    switch kgrid.dim
        case 1
            is(is ~= 0) = [];
        case 2
            js(is ~= 0) = [];
            is(is ~= 0) = [];
        case 3
            ks(is ~= 0) = [];
            js(is ~= 0) = [];
            is(is ~= 0) = [];
    end
end
if kgrid.dim > 1
    [y_closest, y_closest_ind] = findClosest(kgrid.y_vec, point(2));
    if abs(y_closest - point(2)) < ongrid_threshold
        switch kgrid.dim
            case 2
                is(js ~= 0) = [];
                js(js ~= 0) = [];
            case 3
                ks(js ~= 0) = [];
                is(js ~= 0) = [];
                js(js ~= 0) = [];
        end
    end
end
if kgrid.dim > 2
    [z_closest, z_closest_ind] = findClosest(kgrid.z_vec, point(3));
    if abs(z_closest - point(3)) < ongrid_threshold
        js(ks ~= 0) = [];
        is(ks ~= 0) = [];
        ks(ks ~= 0) = [];
    end
end

% get nearest subs to given point and shift canonical subs by this amount
is = is + x_closest_ind;
if kgrid.dim > 1
    js = js + y_closest_ind;
end
if kgrid.dim > 2
    ks = ks + z_closest_ind;
end

% plot in debug mode
if debug && kgrid.dim == 2
    figure;
    tolstar_on_grid = zeros(kgrid.Nx, kgrid.Ny);
    tolstar_on_grid(sub2ind([kgrid.Nx, kgrid.Ny], is, js)) = 1;
    imagesc(tolstar_on_grid);
    title('Tol Star (truncated BLI) on grid');
    xlabel('Grid points');
    ylabel('Grid points');
    clear tolstar_on_grid;
    axis image;
end

% check all points are within the simulation grid - if any are found,
% remove them
switch kgrid.dim
    case 1
        if (min(is, [], 'all') < 1) || (max(is, [], 'all') > kgrid.Nx)
            inbounds = (is >= 1) & (is <= kgrid.Nx);
            is = is(inbounds);
        end
    case 2
        if (min(is, [], 'all') < 1) || (max(is, [], 'all') > kgrid.Nx) || ...
           (min(js, [], 'all') < 1) || (max(js, [], 'all') > kgrid.Ny)
            inbounds = (is >= 1) & (is <= kgrid.Nx) & (js >= 1) & (js <= kgrid.Ny);
            is = is(inbounds);
            js = js(inbounds); 
        end
    case 3
        if (min(is, [], 'all') < 1) || (max(is, [], 'all') > kgrid.Nx) || ...
           (min(js, [], 'all') < 1) || (max(js, [], 'all') > kgrid.Ny) || ...
           (min(ks, [], 'all') < 1) || (max(ks, [], 'all') > kgrid.Nz)
            inbounds = (is >= 1) & (is <= kgrid.Nx) & (js >= 1) & (js <= kgrid.Ny) & (ks >= 1) & (ks <= kgrid.Nz);
            is = is(inbounds);
            js = js(inbounds);
            ks = ks(inbounds);
        end        
end
            
% convert to indices
switch kgrid.dim
    case 1
        lin_ind = is;
    case 2
        lin_ind = kgrid.Nx .* (js - 1) + is;
        % lin_ind = sub2ind([kgrid.Nx, kgrid.Ny], is, js);
    case 3
        lin_ind = kgrid.Nx .* kgrid.Ny .* (ks - 1) + kgrid.Nx .* (js - 1) + is;
        % lin_ind = sub2ind([kgrid.Nx, kgrid.Ny, kgrid.Nz], is, js, ks);
end
