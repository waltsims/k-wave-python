all_params = make_params();



output_file = 'collectedValues/makeCartSphericalSegment.mat';
recorder = utils.TestRecorder(output_file);

for idx = 1:length(all_params)
    disp(idx);
    params = all_params{idx};

    coordinates = makeCartSphericalSegment(params{:});

    recorder.recordVariable('params', params);
    recorder.recordVariable('coordinates', coordinates);
    recorder.increment();
end

recorder.saveRecordsToDisk();

function all_params = make_params()
    %% parameters taken from k-wave testing
    bowl_pos = [0.5, 0.5, 0.5];
    focus_pos = [0.5, 0.5, 1.5];
    num_points_vec = [69, 70, 200, 1000];
    radius = 65e-3;
    ap_diam1 = 30e-3;
    ap_diam2 = 45e-3;
    ap_diam3 = 60e-3;
    
    % find position where to split the bowl
    bowl_height1 = radius - sqrt(radius^2 - (ap_diam1/2)^2);
    bowl_height2 = radius - sqrt(radius^2 - (ap_diam2/2)^2);
    all_params = {};
    % loop over some different numbers of points
    for points_ind = 1:length(num_points_vec)
    
        num_points = num_points_vec(points_ind);
    
        % make regular bowl
        bowl_points = makeCartBowl(bowl_pos, radius, ap_diam3, focus_pos, num_points);
    
        % split array
        bowl_points_ann1 = bowl_points(:, bowl_points(3, :) <= bowl_height1);
        bowl_points_ann2 = bowl_points(:, (bowl_points(3, :) <=  bowl_height2) & (bowl_points(3, :) > bowl_height1));
        bowl_points_ann3 = bowl_points(:, bowl_points(3, :) >  bowl_height2);
    
        % count points
        num_points_ann1 = size(bowl_points_ann1, 2);
        num_points_ann2 = size(bowl_points_ann2, 2);
        num_points_ann3 = size(bowl_points_ann3, 2);
    
        % (bowl_pos, radius, inner_diameter, outer_diameter, focus_pos, num_points, plot_bowl, num_points_inner
    
        all_params{end+1} = {bowl_pos, radius, 0, ap_diam1, focus_pos, num_points_ann1, false, 0};
        all_params{end+1} = {bowl_pos, radius, ap_diam1, ap_diam2, focus_pos, num_points_ann2, [], num_points_ann1};
        all_params{end+1} = {bowl_pos, radius, ap_diam2, ap_diam3, focus_pos, num_points_ann3, [], num_points_ann1 + num_points_ann2};
    
    end
end