dims = [2, 3];
thresholds = [0.5, 0.25, 0.1];
list_d = [0.1, 0.5, 1.0];
kgrid_dims = [10, 20, 49];

recorder = utils.TestRecorder('collectedValues/grid2cart.mat');
for dim = dims

    for threshold = thresholds

        kgrid = {};
        kgrid.dim = dim;

        kgrid.Nx = kgrid_dims(1);
        kgrid.dx = list_d(1);
        kgrid.Ny = kgrid_dims(2);
        kgrid.dy = list_d(2);

        if dim == 3
            kgrid.Nz = kgrid_dims(3);
            kgrid.dz = list_d(3);
            grid_data = rand([Nx, Ny, Nz]) < threshold;
        else
            grid_data = rand([Nx, Ny]) < threshold;
        end

        [cart_data, order_index] = grid2cart(kgrid, grid_data);

        recorder.recordObject('kgrid', kgrid);
        recorder.recordVariable('cart_data', cart_data);
        recorder.recordVariable('grid_data', grid_data);
        recorder.recordVariable('order_index', order_index);
        recorder.increment();

    end
    
end
recorder.saveRecordsToDisk();
disp('Done.')