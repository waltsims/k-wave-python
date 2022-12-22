import functools
import os
from pathlib import Path

import numpy as np

from kwave.enums import DiscreteCosine, DiscreteSine
from kwave.kgrid import kWaveGrid
from scipy.io import loadmat


class TestRecordReader(object):

    def __init__(self, record_filename):
        recorded_data = loadmat(record_filename, simplify_cells=True)
        self._records = recorded_data
        self._total_steps = recorded_data['total_steps']
        self._step = 0

    def expected_value_of(self, name, squeeze=False):
        record_key = f'step_{self._step}___{name}'
        value = self._records[record_key]
        if squeeze:
            value = np.squeeze(value)
        return value

    def increment(self):
        self._step += 1
        if self._step > self._total_steps:
            raise ValueError("Exceeded total recorded steps. Perhaps something is wrong with logic?")


def recursive_getattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def check_kgrid_equality(kgrid_object: kWaveGrid, expected_kgrid_dict: dict):
    are_totally_equal = True
    for key, expected_value in expected_kgrid_dict.items():

        ignore_if_nan = False
        match key:
            case "kx_vec":
                mapped_key = 'k_vec.x'
            case "ky_vec":
                mapped_key = 'k_vec.y'
                ignore_if_nan = True
            case "kz_vec":
                mapped_key = 'k_vec.z'
                ignore_if_nan = True
            case "kx_max":
                mapped_key = 'k_max.x'
            case "ky_max":
                mapped_key = 'k_max.y'
                ignore_if_nan = True
            case "kz_max":
                mapped_key = 'k_max.z'
                ignore_if_nan = True
            case "k_max":
                mapped_key = "k_max_all"
            case "x_size":
                mapped_key = 'size.x'
            case "xn_vec":
                mapped_key = 'n_vec.x'
            case "yn_vec":
                mapped_key = 'n_vec.y'
                ignore_if_nan = True
            case "zn_vec":
                mapped_key = 'n_vec.z'
                ignore_if_nan = True
            case "xn_vec_sgx":
                mapped_key = 'n_vec_sg.x'
            case "xn_vec_sgy":
                mapped_key = 'n_vec_sg.y'
                ignore_if_nan = True
            case "xn_vec_sgz":
                mapped_key = 'n_vec_sg.z'
                ignore_if_nan = True
            case "dxudxn":
                mapped_key = 'dudn.x'
            case "dyudyn":
                mapped_key = 'dudn.y'
                ignore_if_nan = True
            case "dzudzn":
                mapped_key = 'dudn.z'
                ignore_if_nan = True

            case "dxudxn_sgx":
                mapped_key = 'dudn_sg.x'
            case "dyudyn_sgy":
                mapped_key = 'dudn_sg.y'
                ignore_if_nan = True
            case "dzudzn_sgz":
                mapped_key = 'dudn_sg.z'
                ignore_if_nan = True

            case "yn_vec_sgy":
                mapped_key = 'n_vec_sg.y'
                ignore_if_nan = True
            case "zn_vec_sgz":
                mapped_key = 'n_vec_sg.z'
                ignore_if_nan = True

            case 'y_vec' | 'z_vec' | 'ky' | 'kz' | 'yn' | 'zn':
                mapped_key = key
                ignore_if_nan = True

            case _:
                mapped_key = key

        actual_value = recursive_getattr(kgrid_object, mapped_key, None)
        actual_value = np.squeeze(actual_value)
        expected_value = np.array(expected_value)

        if ignore_if_nan and expected_value.size == 1 and (np.isnan(actual_value)) and (expected_value == 0):
            are_equal = True
        elif (actual_value is None) and (expected_value is not None):
            are_equal = False
        elif np.size(actual_value) >= 2 or np.size(expected_value) >= 2:
            are_equal = np.allclose(actual_value, expected_value)
        else:
            are_equal = (actual_value == expected_value)

        if not are_equal:
            print('Following property does not match:')
            print(f'\tkey: {key}, mapped_key: {mapped_key}')
            print(f'\t\texpected: {expected_value}')
            print(f'\t\tactual: {actual_value}')
            are_totally_equal = False

    return are_totally_equal


def test_get_color_map():
    test_record_path = os.path.join(Path(__file__).parent, 'collectedValues/kWaveGrid.mat')
    reader = TestRecordReader(test_record_path)

    Nx = 10
    dx = 0.1
    Ny = 14
    dy = 0.05
    Nz = 9
    dz = 0.13

    for dim in range(1, 4):
        print('Dim:', dim)
        if dim == 1:
            kgrid = kWaveGrid(Nx, dx)
        elif dim == 2:
            kgrid = kWaveGrid([Nx, Ny], [dx, dy])
        else:
            kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dy, dz])

        check_kgrid_equality(kgrid, reader.expected_value_of('kgrid'))
        reader.increment()

        kgrid.setTime(52, 0.0001)
        check_kgrid_equality(kgrid, reader.expected_value_of('kgrid'))
        reader.increment()

        t_array, dt = kgrid.makeTime(1596)
        check_kgrid_equality(kgrid, reader.expected_value_of('kgrid'))
        assert np.allclose(reader.expected_value_of('returned_t_array'), t_array)
        assert np.allclose(reader.expected_value_of('returned_dt'), dt)
        reader.increment()

        for dtt_type in [*list(DiscreteCosine), *list(DiscreteSine)]:
            print(dtt_type)
            k, M = kgrid.k_dtt([dtt_type] * dim)
            check_kgrid_equality(kgrid, reader.expected_value_of('kgrid'))
            assert np.allclose(reader.expected_value_of('returned_k'), k)
            assert np.allclose(reader.expected_value_of('returned_M'), M)
            reader.increment()

            kx_vec_dtt, M = kgrid.kx_vec_dtt(dtt_type)
            check_kgrid_equality(kgrid, reader.expected_value_of('kgrid'))
            assert np.allclose(reader.expected_value_of('returned_kx_vec_dtt'), kx_vec_dtt)
            assert np.allclose(reader.expected_value_of('returned_M'), M)
            reader.increment()

            ky_vec_dtt, M = kgrid.ky_vec_dtt(dtt_type)
            check_kgrid_equality(kgrid, reader.expected_value_of('kgrid'))
            assert np.allclose(reader.expected_value_of('returned_ky_vec_dtt'), ky_vec_dtt)
            assert np.allclose(reader.expected_value_of('returned_M'), M)
            reader.increment()

            kz_vec_dtt, M = kgrid.kz_vec_dtt(dtt_type)
            check_kgrid_equality(kgrid, reader.expected_value_of('kgrid'))
            assert np.allclose(reader.expected_value_of('returned_kz_vec_dtt'), kz_vec_dtt)
            assert np.allclose(reader.expected_value_of('returned_M'), M)
            reader.increment()

        for axisymmetric in [None, 'WSWA', 'WSWS']:
            if axisymmetric == 'WSWS' and dim == 1:
                continue
            highest_prime_factors = kgrid.highest_prime_factors(axisymmetric)
            check_kgrid_equality(kgrid, reader.expected_value_of('kgrid'))
            assert np.allclose(reader.expected_value_of('returned_highest_prime_factors'), highest_prime_factors)
            reader.increment()

        inp_xn_vec = reader.expected_value_of('inp_xn_vec')
        inp_dxudxn = reader.expected_value_of('inp_dxudxn')
        inp_xn_vec_sgx = reader.expected_value_of('inp_xn_vec_sgx')
        inp_dxudxn_sgx = reader.expected_value_of('inp_dxudxn_sgx')
        kgrid.setNUGrid(dim, inp_xn_vec, inp_dxudxn, inp_xn_vec_sgx, inp_dxudxn_sgx)
        check_kgrid_equality(kgrid, reader.expected_value_of('kgrid'))
        reader.increment()
