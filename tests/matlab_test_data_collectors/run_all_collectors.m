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

function files = getListOfFiles(directory)
    list    = dir(fullfile(directory, '*.m'));
    files = {list.name};
end