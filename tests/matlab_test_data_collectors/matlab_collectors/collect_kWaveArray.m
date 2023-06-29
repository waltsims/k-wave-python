

output_file = 'collectedValues/kWaveArray.mat';
recorder = utils.TestRecorder(output_file);

kwave_array = kWaveArray();

% Useful for checking if the defaults are set correctly
recorder.recordObject('kwave_array', kwave_array);
recorder.increment();

% Useful for checking if different constructor param values are internally 
% set correctly 
kwave_array = kWaveArray('Axisymmetric', true, 'BLITolerance', 0.5, 'BLIType', 'sinc', 'SinglePrecision', true, 'UpsamplingRate', 20);
recorder.recordObject('kwave_array', kwave_array);
recorder.increment();
kwave_array = kWaveArray('Axisymmetric', false, 'BLITolerance', 0.5, 'BLIType', 'exact', 'SinglePrecision', false, 'UpsamplingRate', 1);
recorder.recordObject('kwave_array', kwave_array);
recorder.increment();

% Useful for testing addAnnularArray with 2x1 diameters
kwave_array.addAnnularArray([3, 5, 10], 5, [1.2; 0.5], [12, 21, 3]);
recorder.recordObject('kwave_array', kwave_array);
recorder.increment();

% Useful for testing addAnnularArray with 2x2 diameters
kwave_array.addAnnularArray([3, 5, 10], 5, [1.2, 5.3; 0.5, 1.0], [12, 21, 3]);
recorder.recordObject('kwave_array', kwave_array);
recorder.increment();

% Useful for testing addAnnularElement
kwave_array.addAnnularElement([0, 0, 0], 5, [0.001, 0.03], [1, 5, -3]);
recorder.recordObject('kwave_array', kwave_array);
recorder.increment();

% Useful for testing addBowlElement
kwave_array.addBowlElement([0, 0, 0], 5, 4.3, [1, 5, -3]);
recorder.recordObject('kwave_array', kwave_array);
recorder.increment();

% Useful for testing addRectElement in 2D
kwave_array.addRectElement([12, -8, 0.3], 3, 4, [2, 4, 5]);
recorder.recordObject('kwave_array', kwave_array);
recorder.increment();

% Useful for testing addArcElement
% cannot add to previous kWaveArray because it contains 3D elements
% and ArcElement is a 2D element.
kwave_array = kWaveArray();
kwave_array.addArcElement([0, 0.3], 5, 4.3, [1, 5]);
recorder.recordObject('kwave_array', kwave_array);
recorder.increment();

% Useful for testing addDiscElement
kwave_array.addDiscElement([0, 0.3], 5, [1, 5]);
recorder.recordObject('kwave_array', kwave_array);
recorder.increment();

% Useful for testing addRectElement in 2D
kwave_array.addRectElement([12, -8], 3, 4, 2);
recorder.recordObject('kwave_array', kwave_array);
recorder.increment();

% Useful for testing removeElement -- remove last 3 elements from above
kwave_array.removeElement(3);
recorder.recordObject('kwave_array', kwave_array);
recorder.increment();
kwave_array.removeElement(1);
recorder.recordObject('kwave_array', kwave_array);
recorder.increment();
kwave_array.removeElement(1);
recorder.recordObject('kwave_array', kwave_array);
recorder.increment();

% Useful for testing addLineElement in 1D
kwave_array = kWaveArray();
kwave_array.addLineElement([0], [5]);
recorder.recordObject('kwave_array', kwave_array);
recorder.increment();

% Useful for testing addLineElement in 2D
kwave_array = kWaveArray();
kwave_array.addLineElement([0, 3], [5, 2]);
recorder.recordObject('kwave_array', kwave_array);
recorder.increment();

% Useful for testing addLineElement in 3D
kwave_array = kWaveArray();
kwave_array.addLineElement([0, 3, -3], [5, 2, -9]);
recorder.recordObject('kwave_array', kwave_array);
recorder.increment();

% Useful for testing getElementGridWeights
kgrid = kWaveGrid(100, 0.1, 200, 0.3, 150, 0.4);
grid_weights = kwave_array.getElementGridWeights(kgrid, 1);
recorder.recordVariable('grid_weights', grid_weights);

% Useful for testing getElementBinaryMask
mask = kwave_array.getElementBinaryMask(kgrid, 1);
recorder.recordVariable('mask', mask);
recorder.increment();

% Useful for testing getArrayGridWeights
grid_weights = kwave_array.getArrayGridWeights(kgrid);
recorder.recordVariable('grid_weights', grid_weights);

% Useful for testing getArrayBinaryMask
mask = kwave_array.getArrayBinaryMask(kgrid);
recorder.recordVariable('mask', mask);
recorder.increment();

% Useful for testing getDistributedSourceSignal
source_signal = rand(12, 20);
distributed_source_signal = kwave_array.getDistributedSourceSignal(kgrid, source_signal);
recorder.recordVariable('source_signal', source_signal);
recorder.recordVariable('distributed_source_signal', distributed_source_signal);
recorder.increment();

% Useful for testing combineSensorData
kgrid = kWaveGrid(10, 0.1, 100, 0.1, 100, 0.1);
sensor_data = rand(1823, 20);
combined_sensor_data = kwave_array.combineSensorData(kgrid, sensor_data);
recorder.recordVariable('sensor_data', sensor_data);
recorder.recordVariable('combined_sensor_data', combined_sensor_data);
recorder.increment();

% Useful for testing setArrayPosition
translation = [5, 2, -1];
rotation = [20, 30, 15];
kwave_array.setArrayPosition(translation, rotation);
recorder.recordVariable('translation', translation);
recorder.recordVariable('rotation', rotation);
recorder.recordObject('kwave_array', kwave_array);
recorder.increment();

% Useful for testing setAffineTransform
affine_transform = rand(4, 4);
kwave_array.setAffineTransform(affine_transform);
recorder.recordVariable('affine_transform', affine_transform);
recorder.recordObject('kwave_array', kwave_array);
recorder.increment();

% Useful for testing addAnnularArray
kwave_array = kWaveArray();
kwave_array.addAnnularArray([3, 5, 10], 5, [1.2; 0.5], [12, 21, 3]);
element_pos = kwave_array.getElementPositions();
recorder.recordVariable('element_pos', element_pos);
recorder.increment();

recorder.saveRecordsToDisk();