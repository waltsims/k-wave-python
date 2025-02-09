types = {
    "Bartlett"
    "Bartlett-Hanning"
    "Blackman"
    "Blackman-Harris"
    "Blackman-Nuttall"
    "Cosine"
    "Flattop"
    "Gaussian"
    "HalfBand"
    "Hamming"
    "Hanning"
    "Kaiser"
    "Lanczos"
    "Nuttall"
    "Rectangular"
    "Triangular"
    "Tukey"
};

dims = [1, 2, 3];
% rotation_symmetric_square
other_params = [
    [false, false, false],
    [false, false, true],
    [false, true, false],
    [false, true, true],
    [true, false, false],
    [true, false, true],
    [true, true, false],
    [true, true, true],
];
% rotation = [true false];
% symmetric = [true false];
% square = [true false];
windows_with_param = {'Tukey', 'Blackman', 'Gaussian', 'Kaiser'};
control_params = [0.5, 0.16, 0.5, 3];

recorder = utils.TestRecorder('collectedValues/getWin.mat');

for dim = dims
    for type_idx = 1:length(types)
        type_ = string(types(type_idx));

        for p=other_params'
            rotation = p(1);
            symmetric = p(2);
            square = p(3);

            for control_param=control_params

                input_args = { ...
                    'Rotation', rotation, ...
                    'Symmetric', symmetric, ...
                    'Square', square ...
                };

                if any(strcmp(windows_with_param, type_))
                    input_args{end+1} = 'Param';
                    input_args{end+1} = control_param;
                end

                if dim == 1
                    sizes = [1, 17, 45, 80];
                elseif dim == 2
                    sizes = [[12, 13]; [40, 40]; [80, 90]]';
                elseif dim == 3
                    sizes = [[17, 23, 42]; [45, 45, 45]; [10, 50, 80]]';
                end
                for N = sizes
                    if idx == 2049
%                         disp(idx);
                    end

                    [win, cg] = getWin(N', type_, input_args{:});
%                     disp(size(win));

                    type_ = char(type_);
                    recorder.recordVariable('N', N);
                    recorder.recordVariable('type_', type_);
                    recorder.recordVariable('input_args', input_args);
                    recorder.recordVariable('win', win);
                    recorder.recordVariable('cg', cg);
                    recorder.increment();

                end
            end
        end
    end
end
recorder.saveRecordsToDisk();
disp('Done.')
