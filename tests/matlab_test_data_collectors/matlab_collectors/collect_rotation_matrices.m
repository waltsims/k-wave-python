function collect_rotation_matrices()
    % Test angles
    angles = [-180, -90, -45, 0, 45, 90, 180];
    
    output_file = 'collectedValues/rotation_matrices.mat';
    recorder = utils.TestRecorder(output_file);
    
    % Test all angles
    for i = 1:length(angles)
        theta = angles(i);
        
        % Record angle and matrices
        recorder.recordVariable('theta', theta);
        recorder.recordVariable('Rx_matrix', Rx(theta));
        recorder.recordVariable('Ry_matrix', Ry(theta));
        recorder.recordVariable('Rz_matrix', Rz(theta));
        recorder.increment();
    end
    
    recorder.saveRecordsToDisk();
end

% generate 3D rotation matrices
function R = Rx(theta)
    R = [1, 0, 0; 0, cosd(theta), -sind(theta); 0, sind(theta), cosd(theta)];
end

function R = Ry(theta)
    R = [cosd(theta), 0, sind(theta); 0, 1, 0; -sind(theta), 0, cosd(theta)];
end

function R = Rz(theta)
    R = [cosd(theta), -sind(theta), 0; sind(theta), cosd(theta), 0; 0, 0, 1];
end 