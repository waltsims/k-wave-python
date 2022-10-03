from kwave.utils.ioutils import *
import numpy as np
from pathlib import Path


def test_write_matrix(tmp_path_factory):
    idx = 0
    for dim in range(1, 3):
        for compression_level in range(1, 9):
            tmp_path = tmp_path_factory.mktemp("matrix") / f"{idx}.h5"
            matrix = np.single(10.0 * np.ones([1, dim]))
            write_matrix(tmp_path, matrix=matrix, matrix_name='test')
            local_h5 = h5py.File(tmp_path, 'r')
            ref_path = os.path.join(Path(__file__).parent, f"collectedValues/writeMatrix/{idx}.h5")
            ref_h5 = h5py.File(ref_path, 'r')
            # TODO: this introduces dependency on h5diff
            cmd = f'h5diff -c -d 0.00000001 {tmp_path} {ref_path}'
            print(cmd)
            res = os.system(cmd)
            assert res == 0
            idx = idx + 1
    pass


def test_write_flags(tmp_path_factory):
    idx = 0
    for dim in range(1, 3):
        tmp_path = tmp_path_factory.mktemp("flags") / f"{idx}.h5"
        grid_size = 10 * np.ones([3, 1])
        grid_spacing = 0.1 * np.ones([3, 1])
        pml_size = 2 * np.ones([3, 1])
        pml_alpha = 0.5 * np.ones([3, 1])
        write_grid(tmp_path, grid_size, grid_spacing, pml_size, pml_alpha, 5, 0.5, 1540)
        write_matrix(tmp_path, np.asarray([0.], dtype=np.single), 'sensor_mask_index')
        write_flags(tmp_path)
        ref_path = os.path.join(Path(__file__).parent, f"collectedValues/writeFlags/{idx}.h5")

        # TODO: this introduces dependency on h5diff
        cmd = f'h5diff -c -d 0.00000001 {tmp_path} {ref_path}'
        print(cmd)
        res = os.system(cmd)
        assert res == 0, f"File index {idx} failed"
        idx = idx + 1
    pass


def test_write_attributes(tmp_path_factory):
    idx = 0
    matrix_name = 'test'
    tmp_path = tmp_path_factory.mktemp("attributes") / f"{idx}.h5"
    matrix = np.single(10.0 * np.ones([1, 1]))
    ref_path = os.path.join(Path(__file__).parent, f"collectedValues/writeAttributes/{idx}.h5")
    write_matrix(tmp_path, matrix, matrix_name)
    write_attributes(tmp_path)  
    local_h5 = h5py.File(tmp_path, 'r')
    ref_h5 = h5py.File(ref_path, 'r')
    for key in local_h5.attrs.keys():
        assert key in ref_h5.attrs.keys()
        assert np.isclose(local_h5.attrs[key], ref_h5.attrs[key])

    idx = idx + 1
    pass

def test_writeGrid(tmp_path_factory):
    idx = 0
    for dim in range(1, 3):
        tmp_path = tmp_path_factory.mktemp("flags") / f"{idx}.h5"
        grid_size = 10 * np.ones([3, 1])
        grid_spacing = 0.1 * np.ones([3, 1])
        pml_size = 2 * np.ones([3, 1])
        pml_alpha = 0.5 * np.ones([3, 1])
        write_grid(tmp_path, grid_size, grid_spacing, pml_size, pml_alpha, 5, 0.5, 1540)
        ref_path = os.path.join(Path(__file__).parent, f"collectedValues/writeGrid/{idx}.h5")

        # TODO: this introduces dependency on h5diff
        cmd = f'h5diff -c -d 0.00000001 {tmp_path} {ref_path}'
        print(cmd)
        res = os.system(cmd)
        assert res == 0, f"File index {idx} failed"
        idx = idx + 1
    pass



