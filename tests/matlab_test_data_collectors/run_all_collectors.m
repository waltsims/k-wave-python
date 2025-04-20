clear all;
% ensure k-wave is on the path
addpath(genpath('../../../k-wave'));
directory = pwd + "/matlab_collectors";
files = getListOfFiles(directory);

for idx=1:length(files)
        % ensure collected value directory has been created
        file_parts = split(files(idx),["_","."]);
        collected_value_dir = pwd + ...
            "/matlab_collectors/collectedValues/" + file_parts(2);
        mkdir(collected_value_dir)
    % run value collector
    run(fullfile(directory, files{idx}));
    clearvars -except idx files directory
end

if ~isRunningInCI()
    updatePythonCollectedValues(directory);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function is_running_in_ci = isRunningInCI()
    is_running_in_ci = ~isempty(getenv('CI'));
end

function updatePythonCollectedValues(directory)
    target = pwd + "/python_testers/collectedValues";
    if exist(target, 'dir')
        rmdir(target, 's')
    end
    movefile(directory + "/collectedValues", target)
end

function files = getListOfFiles(directory)
    list    = dir(fullfile(directory, '*.m'));
    files = {list.name};
end