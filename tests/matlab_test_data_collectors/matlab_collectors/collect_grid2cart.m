dims = [2, 3];
thresholds = [0.5, 0.25, 0.1];
list_d = [0.1, 0.5, 1.0];
kgrid_dims = [10, 20, 49];

recorder = utils.TestRecorder('collectedValues/grid2cart.mat');
for dim = dims

    for threshold = thresholds

        kgrid = {};
        kgrid.dim = kgrid_dims;


        Nx = kgrid_dims(1);
        dx = list_d(1);
        Ny = kgrid_dims(2);
        dy = list_d(2);

        if dim == 3
            Nz = kgrid_dims(3);
            dz = list_d(3);
            kgrid_m = kWaveGrid(Nx, dx, Ny, dy, Nz, dz);
            grid_data = rand([kgrid_m.Nx, kgrid_m.Ny, kgrid_m.Nz]) < threshold;

            kgrid.x = kgrid_m.x;
            kgrid.y = kgrid_m.y;
            kgrid.z = kgrid_m.z; 
        else
            kgrid_m = kWaveGrid(Nx, dx, Ny, dy);
            grid_data = rand([kgrid_m.Nx, kgrid_m.Ny]) < threshold;

            kgrid.x = kgrid_m.x;
            kgrid.y = kgrid_m.y;
        end

        [cart_data, order_index] = grid2cart(kgrid_m, grid_data);

        recorder.recordObject('kgrid', kgrid);
        recorder.recordVariable('cart_data', cart_data);
        recorder.recordVariable('grid_data', grid_data);
        recorder.recordVariable('order_index', order_index);
        recorder.increment();

    end
    
end
recorder.saveRecordsToDisk();
disp('Done.')