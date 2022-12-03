import numpy as np

from kwave.enums import DiscreteCosine
from kwave.kgrid import kWaveGrid
from scipy.io import loadmat


class TestRecordReader(object):

    def __init__(self, record_filename):
        recorded_data = loadmat(record_filename)
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


if __name__ == '__main__':
    recorder = TestRecordReader('/data/code/Work/black_box_testing/x.mat')
    print(recorder.expected_value_of('dx', squeeze=True))

    Nx = 10
    dx = 0.1
    kgrid = kWaveGrid(Nx, dx)

    assert recorder.expected_value_of('Nx', squeeze=True) == kgrid.Nx
    assert recorder.expected_value_of('dx', squeeze=True) == kgrid.dx
    assert np.allclose(recorder.expected_value_of('kx_vec'), kgrid.k_vec.x)
    assert np.allclose(recorder.expected_value_of('k'), kgrid.k)
    assert recorder.expected_value_of('kx_max', squeeze=True) == kgrid.k_max.x

    # recorder.recordExpectedValue('k_max', kgrid.k_max)
    # assert np.allclose(recorder.expected_value_of('k_max'), kgrid.k_max)

    assert np.allclose(recorder.expected_value_of('x'), kgrid.x)
    assert np.allclose(recorder.expected_value_of('y'), kgrid.y)
    assert np.allclose(recorder.expected_value_of('z'), kgrid.z)

    assert np.allclose(recorder.expected_value_of('x_size'), kgrid.size[0])
    # assert np.allclose(recorder.expected_value_of('y_size'), kgrid.size.y)
    # assert np.allclose(recorder.expected_value_of('z_size'), kgrid.size.z)

    assert str(recorder.expected_value_of('t_array')[0]) == kgrid.t_array
    assert recorder.expected_value_of('total_grid_points', squeeze=True) == kgrid.total_grid_points

    recorder.increment()

    kgrid.setTime(52, 0.0001)
    assert recorder.expected_value_of('Nt', squeeze=True) == kgrid.Nt
    assert recorder.expected_value_of('dt', squeeze=True) == kgrid.dt
    assert np.allclose(recorder.expected_value_of('t_array'), kgrid.t_array)

    recorder.increment()
    # t_array, dt = kgrid.makeTime(1596)
    kgrid.makeTime(1596)
    # assert np.allclose(recorder.expected_value_of('returned_t_array'), t_array)
    # assert np.allclose(recorder.expected_value_of('returned_dt'), dt)
    assert np.allclose(recorder.expected_value_of('Nt'), kgrid.Nt)
    assert np.allclose(recorder.expected_value_of('dt'), kgrid.dt)
    assert np.allclose(recorder.expected_value_of('t_array'), kgrid.t_array)

    recorder.increment()
    k, M = kgrid.k_dtt(DiscreteCosine.TYPE_1)
    # assert np.allclose(recorder.expected_value_of('returned_k'), k)
    # assert np.allclose(recorder.expected_value_of('returned_M'), M)

    # recorder.increment()
    # [kx_vec_dtt, M] = kgrid.kx_vec_dtt(1)
    # recorder.recordExpectedValue('returned_kx_vec_dtt', kx_vec_dtt)
    # recorder.recordExpectedValue('returned_M', M)
    #
    # recorder.increment()
    # [ky_vec_dtt, M] = kgrid.ky_vec_dtt(1)
    # recorder.recordExpectedValue('returned_ky_vec_dtt', ky_vec_dtt)
    # recorder.recordExpectedValue('returned_M', M)
    #
    # recorder.increment()
    # [kz_vec_dtt, M] = kgrid.kz_vec_dtt(1)
    # recorder.recordExpectedValue('returned_kz_vec_dtt', kz_vec_dtt)
    # recorder.recordExpectedValue('returned_M', M)
    #
    # recorder.increment()
    # highest_prime_factors = kgrid.highest_prime_factors('WSWA')
    # recorder.recordExpectedValue('returned_highest_prime_factors', highest_prime_factors)
    #
    # recorder.increment()
    # recorder.recordExpectedValue('xn', kgrid.xn)
    # recorder.recordExpectedValue('xn_vec', kgrid.xn_vec)
    # recorder.recordExpectedValue('yn', kgrid.yn)
    # recorder.recordExpectedValue('yn_vec', kgrid.yn_vec)
    # recorder.recordExpectedValue('zn', kgrid.zn)
    # recorder.recordExpectedValue('zn_vec', kgrid.zn_vec)
    #
    # recorder.increment()
    # inp_xn_vec = rand(3, 2)
    # inp_dxudxn = rand(4, 7)
    # inp_xn_vec_sgx = rand(7, 5)
    # inp_dxudxn_sgx = rand(3, 4)
    # kgrid.setNUGrid(1, inp_xn_vec, inp_dxudxn, inp_xn_vec_sgx, inp_dxudxn_sgx)
    #
    # recorder.recordExpectedValue('inp_xn_vec', inp_xn_vec)
    # recorder.recordExpectedValue('inp_dxudxn', inp_dxudxn)
    # recorder.recordExpectedValue('inp_xn_vec_sgx', inp_xn_vec_sgx)
    # recorder.recordExpectedValue('inp_dxudxn_sgx', inp_dxudxn_sgx)
    #
    # recorder.recordExpectedValue('xn_vec', kgrid.xn_vec)
    # recorder.recordExpectedValue('dxudxn', kgrid.dxudxn)
    # recorder.recordExpectedValue('xn_vec_sgx', kgrid.xn_vec_sgx)
    # recorder.recordExpectedValue('dxudxn_sgx', kgrid.dxudxn_sgx)
    # recorder.recordExpectedValue('nonuniform', kgrid.nonuniform)
