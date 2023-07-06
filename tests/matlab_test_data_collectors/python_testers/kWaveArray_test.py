import os

import numpy as np
from kwave.kgrid import kWaveGrid

from kwave.utils.kwave_array import kWaveArray
from tests.matlab_test_data_collectors.python_testers.utils.check_equality import check_kwave_array_equality
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_kwave_array():
    # test_record_path = os.path.join(Path(__file__).parent, 'collectedValues/kWaveArray.mat')
    test_record_path = os.path.join('/Users/farid/PycharmProjects/k-wave-python/tests/matlab_test_data_collectors/matlab_collectors', 'collectedValues/kWaveArray.mat')
    reader = TestRecordReader(test_record_path)

    kwave_array = kWaveArray()

    # Useful for checking if the defaults are set correctly
    check_kwave_array_equality(kwave_array, reader.expected_value_of('kwave_array'))
    reader.increment()

    kwave_array = kWaveArray(axisymmetric=True, bli_tolerance=0.5, bli_type='sinc', single_precision=True, upsampling_rate=20)
    check_kwave_array_equality(kwave_array, reader.expected_value_of('kwave_array'))
    reader.increment()

    kwave_array = kWaveArray(axisymmetric=False, bli_tolerance=0.5, bli_type='exact', single_precision=False,
                             upsampling_rate=1)
    check_kwave_array_equality(kwave_array, reader.expected_value_of('kwave_array'))
    reader.increment()

    kwave_array.add_annular_array([3, 5, 10], 5, [[1.2, 0.5]], [12, 21, 3])
    check_kwave_array_equality(kwave_array, reader.expected_value_of('kwave_array'))
    reader.increment()

    kwave_array.add_annular_array([3, 5, 10], 5, [[1.2, 0.5], [5.3, 1.0]], [12, 21, 3])
    check_kwave_array_equality(kwave_array, reader.expected_value_of('kwave_array'))
    reader.increment()

    kwave_array.add_annular_element([0, 0, 0], 5, [0.001, 0.03], [1, 5, -3])
    check_kwave_array_equality(kwave_array, reader.expected_value_of('kwave_array'))
    reader.increment()

    kwave_array.add_bowl_element([0, 0, 0], 5, 4.3, [1, 5, -3])
    check_kwave_array_equality(kwave_array, reader.expected_value_of('kwave_array'))
    reader.increment()

    kwave_array.add_rect_element([12, -8, 0.3], 3, 4, [2, 4, 5])
    check_kwave_array_equality(kwave_array, reader.expected_value_of('kwave_array'))
    reader.increment()

    kwave_array.add_disc_element([0, 0.3, 12], 5, [1, 5, 8])
    check_kwave_array_equality(kwave_array, reader.expected_value_of('kwave_array'))
    reader.increment()

    kwave_array = kWaveArray()
    kwave_array.add_arc_element([0, 0.3], 5, 4.3, [1, 5])
    check_kwave_array_equality(kwave_array, reader.expected_value_of('kwave_array'))
    reader.increment()

    kwave_array.add_disc_element([0, 0.3], 5)
    check_kwave_array_equality(kwave_array, reader.expected_value_of('kwave_array'))
    reader.increment()

    # Useful for testing addRectElement in 2D
    kwave_array.add_rect_element([12, -8], 3, 4, 2)
    check_kwave_array_equality(kwave_array, reader.expected_value_of('kwave_array'))
    reader.increment()

    kwave_array.remove_element(2)
    check_kwave_array_equality(kwave_array, reader.expected_value_of('kwave_array'))
    reader.increment()
    kwave_array.remove_element(0)
    check_kwave_array_equality(kwave_array, reader.expected_value_of('kwave_array'))
    reader.increment()
    kwave_array.remove_element(0)
    check_kwave_array_equality(kwave_array, reader.expected_value_of('kwave_array'))
    reader.increment()

    kwave_array = kWaveArray()
    kwave_array.add_line_element([0], [5])
    check_kwave_array_equality(kwave_array, reader.expected_value_of('kwave_array'))
    reader.increment()

    kwave_array = kWaveArray()
    kwave_array.add_line_element([0, 3], [5, 2])
    check_kwave_array_equality(kwave_array, reader.expected_value_of('kwave_array'))
    reader.increment()

    kwave_array = kWaveArray()
    kwave_array.add_line_element([0, 3, -3], [5, 2, -9])
    check_kwave_array_equality(kwave_array, reader.expected_value_of('kwave_array'))
    reader.increment()

    kgrid = kWaveGrid([100, 200, 150], [0.1, 0.3, 0.4])
    grid_weights = kwave_array.get_element_grid_weights(kgrid, 0)
    print(grid_weights.shape)



    # # Useful for testing getElementGridWeights
    # kgrid = kWaveGrid(100, 0.1, 200, 0.3, 150, 0.4);
    # grid_weights = kwave_array.getElementGridWeights(kgrid, 1);
    # recorder.recordVariable('grid_weights', grid_weights);
    #
    # # Useful for testing getElementBinaryMask
    # mask = kwave_array.getElementBinaryMask(kgrid, 1);
    # recorder.recordVariable('mask', mask);
    # recorder.increment();
    #
    # # Useful for testing getArrayGridWeights
    # grid_weights = kwave_array.getArrayGridWeights(kgrid);
    # recorder.recordVariable('grid_weights', grid_weights);
    #
    # # Useful for testing getArrayBinaryMask+

    # mask = kwave_array.getArrayBinaryMask(kgrid);
    # recorder.recordVariable('mask', mask);
    # recorder.increment();
    #
    # # Useful for testing getDistributedSourceSignal
    # source_signal = rand(12, 20);
    # distributed_source_signal = kwave_array.getDistributedSourceSignal(kgrid, source_signal);
    # recorder.recordVariable('source_signal', source_signal);
    # recorder.recordVariable('distributed_source_signal', distributed_source_signal);
    # recorder.increment();
    #
    # # Useful for testing combineSensorData
    # kgrid = kWaveGrid(10, 0.1, 100, 0.1, 100, 0.1);
    # sensor_data = rand(1823, 20);
    # combined_sensor_data = kwave_array.combineSensorData(kgrid, sensor_data);
    # recorder.recordVariable('sensor_data', sensor_data);
    # recorder.recordVariable('combined_sensor_data', combined_sensor_data);
    # recorder.increment();
    #
    # # Useful for testing setArrayPosition
    # translation = [5, 2, -1];
    # rotation = [20, 30, 15];
    # kwave_array.setArrayPosition(translation, rotation);
    # recorder.recordVariable('translation', translation);
    # recorder.recordVariable('rotation', rotation);
    # recorder.recordObject('kwave_array', kwave_array);
    # recorder.increment();
    #
    # # Useful for testing setAffineTransform
    # affine_transform = rand(4, 4);
    # kwave_array.setAffineTransform(affine_transform);
    # recorder.recordVariable('affine_transform', affine_transform);
    # recorder.recordObject('kwave_array', kwave_array);
    # recorder.increment();
    #
    # # Useful for testing addAnnularArray
    # kwave_array = kWaveArray();
    # kwave_array.addAnnularArray([3, 5, 10], 5, [1.2; 0.5], [12, 21, 3]);
    # element_pos = kwave_array.getElementPositions();
    # recorder.recordVariable('element_pos', element_pos);
    # recorder.increment();
