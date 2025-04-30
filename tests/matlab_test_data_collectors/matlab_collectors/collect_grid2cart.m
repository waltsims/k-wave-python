dims = [1, 2, 3];
list_num_points = [10, 19, 5];
list_d = [0.1, 0.5, 1.0];
kgrid_dims = [10, 20, 49];

dx = 0.1;
dy = 0.5;
dz = 1.0;

idx = 0;

recorder = utils.TestRecorder('collectedValues/grid2cart.mat');
for dim = dims
    
    for num_points = list_num_points



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