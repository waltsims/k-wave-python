clear all;

directory = pwd + "/matlab_collectors";
files = getListOfFiles(directory);
% remove this file.

for idx=1:length(files)
        % ensure collected value directory has been created
        file_parts = split(files(idx),["_","."]);
        collected_value_dir = pwd + ...
            "/matlab_collectors/collectedValues/" + file_parts(2);
        mkdir(collected_value_dir)
        % run value collector
        run(fullfile(directory, files{idx}));
end

updateCollectedValues(directory);

function updateCollectedValues(directory)
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