import os
from pathlib import Path

import numpy as np

from kwave.enums import DiscreteCosine, DiscreteSine
from kwave.kgrid import kWaveGrid
from tests.matlab_test_data_collectors.python_testers.utils.check_equality import check_kgrid_equality
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_kwave_grid():
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
