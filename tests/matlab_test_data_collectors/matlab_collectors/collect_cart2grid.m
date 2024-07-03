dims = [1, 2, 3];
list_num_points = [10, 19, 5];
list_d = [0.1, 0.5, 1.0];
kgrid_dims = [10, 20, 49];

dx = 0.1;
dy = 0.5;
dz = 1.0;

idx = 0;

recorder = utils.TestRecorder('collectedValues/cart2grid.mat');
for dim = dims
    
    if dim == 2
        axisymmetric = [false, true];
    else
        axisymmetric = [false];
    end
    
    for is_axisymmetric = axisymmetric
        for num_points = list_num_points

            kgrid = {};
            kgrid.dim = dim;

            kgrid.Nx = kgrid_dims(1);
            kgrid.dx = list_d(1);

            x_bounds = [-kgrid.Nx/2  * kgrid.dx + kgrid.dx/2, kgrid.Nx/2  * kgrid.dx - kgrid.dx/2];
            points_x = utils.rand_vector_in_range(x_bounds(1), x_bounds(2), num_points);

            if dim == 2 || dim == 3
                kgrid.Ny = kgrid_dims(2);
                kgrid.dy = list_d(2);
                if is_axisymmetric
                    y_bounds = [0, (kgrid.Ny * kgrid.dy) - kgrid.dy/2];
                else
                    y_bounds = [-kgrid.Ny/2  * kgrid.dy + kgrid.dy/2, kgrid.Ny/2  * kgrid.dy - kgrid.dy/2];
                end
                
                points_y = utils.rand_vector_in_range(y_bounds(1), y_bounds(2), num_points);
            end

            if dim == 3
                kgrid.Nz = kgrid_dims(3);
                kgrid.dz = list_d(3);
                z_bounds = [-kgrid.Nz/2  * kgrid.dz + kgrid.dz/2, kgrid.Nz/2  * kgrid.dz - kgrid.dz/2];
                points_z = utils.rand_vector_in_range(z_bounds(1), z_bounds(2), num_points);
            end

            if dim == 1
                cart_data = points_x';
            elseif dim == 2
                cart_data = [points_x points_y]';
            elseif dim == 3
                cart_data = [points_x points_y points_z]';
            end

            [grid_data, order_index, reorder_index] = cart2grid(kgrid, cart_data, is_axisymmetric);

            recorder.recordObject('kgrid', kgrid);
            recorder.recordVariable('cart_data', cart_data);
            recorder.recordVariable('grid_data', grid_data);
            recorder.recordVariable('order_index', order_index);
            recorder.recordVariable('reorder_index', reorder_index);
            recorder.recordVariable('is_axisymmetric', is_axisymmetric);
            recorder.increment();

        end
    end
    
end
recorder.saveRecordsToDisk();
disp('Done.')
