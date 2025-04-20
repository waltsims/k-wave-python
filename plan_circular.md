# Plan for Porting 2D Time Reversal Reconstruction For A Circular Sensor Example

## Overview
This example demonstrates time-reversal reconstruction of a 2D photoacoustic wave-field recorded over a circular array of sensor elements. The sensor data is simulated and then time-reversed using kspaceFirstOrder2D.

## Key Components to Port

1. **Grid Setup**
   - Create computational grid with PML
   - Set grid size and spacing
   - Create time array

2. **Initial Pressure Distribution**
   - Load and scale initial pressure from image
   - Resize to match grid points
   - Smooth the distribution

3. **Medium Properties**
   - Set sound speed

4. **Sensor Setup**
   - Create centered Cartesian circular sensor
   - Define sensor radius, angle, and position
   - Generate sensor mask

5. **Simulation**
   - Run initial simulation
   - Add noise to sensor data
   - Create second grid for reconstruction
   - Perform time reversal reconstruction
   - Interpolate data for continuous circle

6. **Visualization**
   - Plot initial pressure and sensor distribution
   - Plot simulated sensor data
   - Plot reconstructed pressure
   - Plot comparison profiles

## Implementation Steps

1. **Setup Dependencies**
   ```python
   from kwave.data import Vector
   from kwave.kgrid import kWaveGrid
   from kwave.kmedium import kWaveMedium
   from kwave.ksensor import kSensor
   from kwave.ksource import kSource
   from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
   from kwave.options.simulation_execution_options import SimulationExecutionOptions
   from kwave.options.simulation_options import SimulationOptions
   from kwave.reconstruction import TimeReversal
   from kwave.utils.colormap import get_color_map
   from kwave.utils.filters import smooth
   ```

2. **Grid Creation**
   - Use PML_size = 20
   - Set Nx = Ny = 256 - 2 * PML_size
   - Set grid size to 10e-3 m
   - Create kWaveGrid object

3. **Initial Pressure**
   - Load source image
   - Scale and resize to match grid
   - Apply smoothing

4. **Medium and Sensor**
   - Set medium.sound_speed = 1500
   - Create circular sensor with radius 4.5e-3 m
   - Set sensor angle to 3π/2
   - Generate Cartesian sensor mask

5. **Simulation Setup**
   - Create simulation options
   - Set execution options for GPU
   - Run initial simulation
   - Add noise with SNR = 40 dB

6. **Reconstruction**
   - Create second grid (300x300)
   - Setup time reversal
   - Run reconstruction
   - Create binary sensor mask
   - Interpolate data
   - Run final reconstruction

7. **Visualization**
   - Plot initial pressure with sensor distribution
   - Plot sensor data
   - Plot reconstructions
   - Plot comparison profiles

## Key Differences from MATLAB Version

1. Use Python's numpy arrays instead of MATLAB matrices
2. Use k-wave-python's Vector class for grid dimensions
3. Implement proper GPU simulation options
4. Use Python's matplotlib for visualization
5. Handle data types explicitly (single precision)
6. Use Python's file handling for image loading

## TimeReversal Class Enhancements

### Cartesian Sensor Support
- Added automatic detection of Cartesian sensor masks
- Implemented conversion between Cartesian and binary masks using `cart2grid` and `grid2cart`
- Added interpolation support for Cartesian sensor data using `interp_cart_data`
- Added new parameter `interp_method` to control interpolation method ('nearest' or 'linear')

### Internal Improvements
- Added helper method `_detect_sensor_type()` to identify and convert sensor masks
- Enhanced validation logic to handle both Cartesian and binary masks
- Improved reconstruction process to handle both mask types seamlessly
- Added proper error handling for mask type detection and conversion

### New Dependencies
- Added `cart2grid` from `kwave.utils.conversion` for converting Cartesian points to binary grid
- Added `grid2cart` from `kwave.utils.conversion` for converting binary grid to Cartesian points
- Added `interp_cart_data` for interpolating Cartesian sensor data

### Testing Plan
- Verify Cartesian sensor support works correctly
- Test conversion between Cartesian and binary masks
- Validate interpolation of sensor data
- Ensure backward compatibility with binary sensors
- Test error handling for invalid mask types

### Implementation Notes
- Sensor mask handling matches MATLAB implementation but with added flexibility
- Cartesian masks are automatically converted to binary masks for reconstruction
- Binary masks are used directly without conversion
- Interpolation is only applied when using Cartesian masks
- All mask operations preserve the geometric properties of the original sensor

## References
- Original MATLAB example: example_pr_2D_TR_circular_sensor.m
- Related Python examples:
  - pr_2D_TR_line_sensor
  - pr_3D_TR_planar_sensor
  - at_array_as_sensor

  origonal matlab logic

  ```matlab
                % compute an equivalent sensor mask using nearest neighbour
                % interpolation, if flags.time_rev = false and
                % cartesian_interp = 'linear' then this is only used for
                % display, if flags.time_rev = true or cartesian_interp =
                % 'nearest' this grid is used as the sensor.mask  
                [sensor.mask, order_index, reorder_index] = cart2grid(kgrid, sensor.mask, flags.axisymmetric);

                % if in time reversal mode, reorder the p0 input data in
                % the order of the binary sensor_mask  
                if flags.time_rev

                    % append the reordering data
                    new_col_pos = length(sensor.time_reversal_boundary_data(1, :)) + 1;
                    sensor.time_reversal_boundary_data(:, new_col_pos) = order_index;

                    % reorder p0 based on the order_index
                    sensor.time_reversal_boundary_data = sortrows(sensor.time_reversal_boundary_data, new_col_pos);

                    % remove the reordering data
                    sensor.time_reversal_boundary_data = sensor.time_reversal_boundary_data(:, 1:new_col_pos - 1);

  ``` 