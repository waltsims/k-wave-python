% Define kgrids with different parameters
kgrid1 = struct('x_vec', [1,2,3], 'y_vec', [1,2,3], 'z_vec', [1,2,3], 'dim', 1);
kgrid2 = struct('x_vec', [1,2,3,4,5], 'y_vec', [1,2,3,4,5], 'z_vec', [1,2,3,4,5], 'dim', 2);
kgrid3 = struct('x_vec', [1,2,3,4,5,6], 'y_vec', [1,2,3,4,5,6], 'z_vec', [1,2,3,4,5,6], 'dim', 3);

% Define sets of points to trim
pointsSet1 = [1,2,3,4,5,6; 1,2,3,4,5,6; 1,2,3,4,5,6]; % 3 rows of x,y,z values
pointsSet2 = [1,2; 1,2; 1,2]; % 3 rows of x,y,z values with fewer columns
pointsSet3 = [1,2,3,4; 1,2,3,4; 1,2,3,4]; % 3 rows of x,y,z values with different number of columns

% Define kgrids and points arrays for looping
kgrids = {kgrid1, kgrid2, kgrid3};
pointsSets = {pointsSet1, pointsSet2, pointsSet3};

output_file = 'collectedValues/trimCartPoints.mat';
recorder = utils.TestRecorder(output_file);

recorder.recordVariable('kgrid1', kgrid1);
recorder.recordVariable('kgrid2', kgrid1);
recorder.recordVariable('kgrid3', kgrid1);

recorder.recordVariable('pointsSet1', pointsSet1);
recorder.recordVariable('pointsSet2', pointsSet1);
recorder.recordVariable('pointsSet3', pointsSet1);

recorder.increment();

% Loop over each combination of kgrid and points set
for i = 1:length(kgrids)
    for j = 1:length(pointsSets)
        % Call the function with current parameters
        trimmed_points = trimCartPoints(kgrids{i}, pointsSets{j});        
        
        recorder.recordVariable('i', i);
        recorder.recordVariable('j', j);
        recorder.recordVariable('trimmed_points', trimmed_points);
        recorder.increment();
    end
end

recorder.saveRecordsToDisk();
