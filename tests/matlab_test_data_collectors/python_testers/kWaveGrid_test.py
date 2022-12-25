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

        matlab_to_python_mapping = {
            "kx_vec": "k_vec.x",
            "ky_vec": "k_vec.y",
            "kz_vec": "k_vec.z",
            "kx_max": "k_max.x",
            "ky_max": "k_max.y",
            "kz_max": "k_max.z",
            "k_max": "k_max_all",
            "x_size": "size.x",
            "xn_vec": "n_vec.x",
            "yn_vec": "n_vec.y",
            "zn_vec": "n_vec.z",
            "xn_vec_sgx": "n_vec_sg.x",
            "xn_vec_sgy": "n_vec_sg.y",
            "xn_vec_sgz": "n_vec_sg.z",
            "dxudxn": "dudn.x",
            "dyudyn": "dudn.y",
            "dzudzn": "dudn.z",
            "dxudxn_sgx": "dudn_sg.x",
            "dyudyn_sgy": "dudn_sg.y",
            "dzudzn_sgz": "dudn_sg.z",
            "yn_vec_sgy": "n_vec_sg.y",
            "zn_vec_sgz": "n_vec_sg.z"
        }

        mapped_key = matlab_to_python_mapping.get(key, key)
        if mapped_key in ["k_vec.y", "k_vec.z", "k_max.y", "k_max.z", "n_vec.y", "n_vec.z",
                          "n_vec_sg.y", "n_vec_sg.z", "dudn.y", "dudn.z", "dudn.z", 
                          "dudn_sg.y", "dudn_sg.z", "n_vec_sg.y", "n_vec_sg.z", 
                          'y_vec', 'z_vec' , 'ky' , 'kz', 'yn', 'zn']:
            ignore_if_nan = True
        else:
            ignore_if_nan = False

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

    assert are_totally_equal


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
