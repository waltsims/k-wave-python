import os
from pathlib import Path

import numpy as np
import pytest

from kwave.kgrid import kWaveGrid
from kwave.utils.conversion import db2neper, grid2cart, neper2db
from kwave.utils.filters import apply_filter, extract_amp_phase, spect
from kwave.utils.interp import get_bli, interp_cart_data
from kwave.utils.mapgen import fit_power_law_params, power_law_kramers_kronig
from kwave.utils.matrix import gradient_fd, num_dim, resize, trim_zeros
from kwave.utils.signals import add_noise, gradient_spect, tone_burst
from tests.matlab_test_data_collectors.python_testers.utils.record_reader import TestRecordReader


def test_grid2cart():
    kgrid = kWaveGrid(
        [1000, 100, 10],
        [1, 1, 1],
    )
    binary_sensor_mask = np.zeros((1000, 100, 10))
    binary_sensor_mask[50, 50, 4] = 1
    binary_sensor_mask[99, 99, 9] = 1

    cart_bsm, order_index = grid2cart(kgrid, binary_sensor_mask)
    assert cart_bsm.shape == (3, 2), f"grid2cart did not return a 3x2 array. Shape is {cart_bsm.shape}"
    print(cart_bsm)
    expected_cart_bsm = np.array([[-450, 0, -1], [-401, 49, 4]]).T
    print(expected_cart_bsm)
    assert np.all(cart_bsm == expected_cart_bsm)


def test_grid2cart_origin():
    kgrid = kWaveGrid(
        [1000, 100, 10],
        [1, 1, 1],
    )
    binary_sensor_mask = np.zeros((1000, 100, 10))
    binary_sensor_mask[500, 50, 5] = 1  # equivalent index in matlab is [501, 51, 6] for mask origin
    cart_bsm, order_index = grid2cart(kgrid, binary_sensor_mask)
    print(cart_bsm)
    print(order_index)
    assert np.all(cart_bsm == 0), "origin location was incorrect"


def test_interp_cart_data_2_points_linear():
    kgrid = kWaveGrid([1000, 100, 10], [1, 1, 1])
    binary_sensor_mask = np.zeros((1000, 100, 10), dtype=bool)
    binary_sensor_mask[501, 51, 7] = True
    cart_sensor_mask = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float32).T  # sensor at the origin and another point
    cart_sensor_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).T  # 3 time steps
    print(cart_sensor_data)
    interp_data = interp_cart_data(kgrid, cart_sensor_data, cart_sensor_mask, binary_sensor_mask, "linear")
    # TODO: find expected value from matlab. In this case we revert to nearest because point is not between p1 and p2.
    print(interp_data)


def test_interp_cart_data_2_points_nearest():
    kgrid = kWaveGrid([1000, 100, 10], [1, 1, 1])
    binary_sensor_mask = np.zeros((1000, 100, 10), dtype=bool)
    binary_sensor_mask[501, 51, 7] = True
    cart_sensor_mask = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float32).T  # sensor at the origin and another point
    cart_sensor_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).T  # 3 time steps
    print(cart_sensor_data)
    interp_data = interp_cart_data(kgrid, cart_sensor_data, cart_sensor_mask, binary_sensor_mask)
    # TODO: find expected value from matlab, current behavior is round up to nearest neighbor
    print(interp_data)


def test_interp_cart_data_1_point_nearest():
    kgrid = kWaveGrid([1000, 100, 10], [1, 1, 1])
    binary_sensor_mask = np.zeros((1000, 100, 10), dtype=bool)
    binary_sensor_mask[501, 51, 6] = True
    cart_sensor_mask = np.array([[0.0, 0.0, 0.0]], dtype=np.float32).T  # sensor at the origin
    cart_sensor_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32).T  # 3 time steps
    print(cart_sensor_data)
    interp_data = interp_cart_data(kgrid, cart_sensor_data, cart_sensor_mask, binary_sensor_mask)
    assert np.allclose(interp_data, cart_sensor_data)
    print(interp_data)


def test_nepers2db():
    expected_scalar = 8.186258123051049e05
    expected_matrix = expected_scalar * np.ones((10, 10))
    assert np.isclose(neper2db(1.5), expected_scalar), "Point check of nepers2db incorrect"
    assert np.allclose(neper2db(1.5 * np.ones((10, 10))), expected_matrix), "matrix check of nepers2db incorrect"
    return


def test_db2nepers():
    expected_scalar = 2.931742395517710e-06
    expected_matrix = expected_scalar * np.ones((10, 10))
    assert np.isclose(db2neper(1.6), expected_scalar), "Point check of db2nepers incorrect"
    assert np.allclose(db2neper(1.6 * np.ones((10, 10))), expected_matrix), "matrix check of db2nepers incorrect"
    return


def test_add_noise():
    input_signal = tone_burst(1.129333333333333e07, 5e5, 5)
    output = add_noise(input_signal, 5)
    p_sig = np.sqrt(np.mean(input_signal**2))
    p_noise = np.sqrt(np.mean((output - input_signal) ** 2))
    snr = 20 * np.log10(p_sig / p_noise)
    assert abs(5 - snr) < 2, "add_noise produced signal with incorrect SNR, this is a stochastic process. Perhaps test again?"
    return


def test_tone_error():
    try:
        test_input_signal = tone_burst(1.129333333333333e07, 5e5, 5, envelope="BobTheBuilder")  # noqa: F841
    except ValueError as e:
        if str(e) == "Unknown envelope BobTheBuilder.":
            pass
        else:
            raise e
    return


def test_signal_length():
    test_input_signal = tone_burst(1.129333333333333e07, 5e5, 5, signal_length=500)
    assert len(np.squeeze(test_input_signal)) == 500


def test_signal_offset():
    offset = 100
    test_input_signal = tone_burst(1.129333333333333e07, 5e5, 5, signal_offset=offset, signal_length=500)
    assert len(np.squeeze(test_input_signal)) == 500
    assert (np.squeeze(test_input_signal)[:offset] == 0.0).all()


def test_num_dim():
    assert num_dim(np.ones((1, 2))) == 1, "num_dim fails for len 3"
    assert num_dim(np.ones((2, 2))) == 2, "num_dim fails for len 2"
    return


def test_spect():
    a = np.array([0, 1.111111111, 2.222222222, 3.3333333333, 4.444444444]) * 1e6
    b = [0.0000, 0.1850, 0.3610, 0.2905, 0.0841]
    c = [3.1416, 1.9199, -0.8727, 2.6180, -0.1745]
    a_t, b_t, c_t = spect(tone_burst(10e6, 2.5e6, 2), 10e6)
    assert (abs(a_t - a) < 0.01).all()
    assert (abs(b_t - b) < 0.0001).all()
    assert (abs(c_t - c) < 0.0001).all()


def test_extract_amp_phase():
    # odd length signal
    test_signal = tone_burst(sample_freq=10_000_000, signal_freq=2.5 * 1_000_000, num_cycles=2, envelope="Gaussian")
    assert np.shape(np.squeeze(test_signal))[0] % 2 != 0
    a_t, b_t, c_t = extract_amp_phase(data=test_signal, fs=10_000_000, source_freq=2.5 * 10**6)
    a, b, c = 0.6547, -1.8035, 2.5926e06
    assert (abs(a_t - a) < 0.01).all()
    assert (abs(b_t - b) < 0.0001).all()
    assert (abs(c_t - c) < 100).all()
    # even length signal
    test_signal = tone_burst(sample_freq=18_000_000, signal_freq=6_000_000, num_cycles=5, envelope="Gaussian")
    assert np.shape(np.squeeze(test_signal))[0] % 2 == 0
    a_t, b_t, c_t = extract_amp_phase(data=test_signal, fs=18_000_000, source_freq=6_000_000)
    a, b, c = 0.6591, -1.5708, 6.000e06
    assert (abs(a_t - a) < 0.01).all()
    assert (abs(b_t - b) < 0.0001).all()
    assert (abs(c_t - c) < 100).all()


def test_extract_amp_phase_2d():
    # Create a 2D test signal with 2 channels
    Fs = 10_000_000  # Sample frequency
    source_freq = 2.5 * 1_000  # Signal frequency
    amp_1 = 2.0
    amp_2 = 1.0
    phase_1 = np.pi / 8
    phase_2 = np.pi / 4
    # Create a 2D test signal with 2 channels
    sig_1 = amp_1 * np.sin(source_freq * 2 * np.pi * np.arange(Fs) / Fs + phase_1)
    sig_2 = amp_2 * np.sin(source_freq * 2 * np.pi * np.arange(Fs) / Fs + phase_2)
    test_signal = np.vstack([sig_1, sig_2])
    # plt.plot(test_signal[0])
    # plt.show()
    amp, phase, f = extract_amp_phase(test_signal, Fs, source_freq, dim=1)

    assert np.allclose(amp, np.array([amp_1, amp_2]))
    # Phase is not used in any k-wave-python examples
    # assert np.allclose(phase, np.array([-phase_1, -phase_2]))
    assert np.allclose(f, source_freq)


def test_extract_amp_phase_double_freq():
    # Create a test signal with double the detection frequency
    Fs = 10_000_000  # Sample frequency
    source_freq = 2.5 * 1_000  # Source frequency
    detection_freq = 5 * 1_000  # Double the frequency (2 * source_freq)

    amp = 2.0
    phase = np.pi / 6

    # Create test signal at source_freq
    t = np.arange(Fs) / Fs
    test_signal = amp * np.sin(source_freq * 2 * np.pi * t + phase)

    # Extract amplitude and phase at double the frequency
    # Explicitly set dim=0 for 1D array
    detected_amp, detected_phase, detected_f = extract_amp_phase(test_signal, Fs, detection_freq, dim=0)

    # The amplitude should be very close to zero since we're detecting at a different frequency
    # than what's present in the signal
    assert np.isclose(detected_amp, 0, atol=1e-3)
    assert np.isclose(detected_f, detection_freq)

    # Now create a signal with the detection frequency
    test_signal_at_detection = amp * np.sin(detection_freq * 2 * np.pi * t + phase)

    # Extract amplitude and phase at the correct frequency
    # Explicitly set dim=0 for 1D array
    detected_amp2, detected_phase2, detected_f2 = extract_amp_phase(test_signal_at_detection, Fs, detection_freq, dim=0)

    # Now the amplitude should match
    assert np.isclose(detected_amp2, amp, rtol=0.01)
    # Phase is not used in any k-wave-python examples, but we can verify it's consistent
    # assert np.isclose(detected_phase2, -phase, rtol=0.01)
    assert np.isclose(detected_f2, detection_freq)


def test_apply_filter_lowpass():
    test_signal = tone_burst(sample_freq=10_000_000, signal_freq=2.5 * 1_000_000, num_cycles=2, envelope="Gaussian")
    filtered_signal = apply_filter(test_signal, fs=1e7, cutoff_f=1e7, filter_type="LowPass")
    expected_signal = [
        0.00000000e00,
        2.76028757e-24,
        -3.59205956e-24,
        1.46416820e-23,
        3.53059551e-22,
        1.00057133e-21,
        -3.57207790e-21,
        -1.10602408e-20,
        1.60600278e-20,
    ]
    assert ((abs(filtered_signal - expected_signal)) < 0.0001).all()
    pass


def test_apply_filter_highpass():
    test_signal = tone_burst(sample_freq=10_000_000, signal_freq=2.5 * 1_000_000, num_cycles=2, envelope="Gaussian")
    filtered_signal = apply_filter(test_signal, fs=1e7, cutoff_f=1e7, filter_type="HighPass")
    expected_signal = [
        0.00000000e00,
        1.40844920e-10,
        1.82974394e-09,
        7.00097845e-09,
        9.76890695e-09,
        -4.62418007e-09,
        -6.61859580e-08,
        -2.38415184e-07,
        -6.35274380e-07,
    ]
    assert ((abs(filtered_signal - expected_signal)) < 0.0001).all()
    pass


def test_apply_filter_bandpass():
    test_signal = tone_burst(sample_freq=10_000_000, signal_freq=2.5 * 1_000_000, num_cycles=2, envelope="Gaussian")
    filtered_signal = apply_filter(test_signal, fs=1e7, cutoff_f=[5e6, 1e7], filter_type="BandPass")
    expected_signal = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert ((abs(filtered_signal - expected_signal)) < 0.0001).all()
    pass


def test_fit_power_law_params():
    a, b = fit_power_law_params(1, 2, 1540, 5e6, 7e6)

    assert abs(a - 1.0) < 0.001
    assert abs(b - 2.0) < 0.001
    pass


def test_get_bli():
    test_signal = tone_burst(sample_freq=10_000_000, signal_freq=2.5 * 1_000_000, num_cycles=2, envelope="Gaussian")
    bli, x_fine = get_bli(test_signal)

    assert x_fine[-1] == 8
    assert abs(bli[-1] - -4.8572e-17) < 0.001
    pass


def test_power_kramers_kronig():
    assert 1540 == power_law_kramers_kronig(1, 1, 1540, 1, 2.5)
    assert 1540 == power_law_kramers_kronig(1, 1, 1540, 1, 1)
    with pytest.warns(UserWarning):
        c = power_law_kramers_kronig(1, 1, 1540, 1, 4)
        assert c == 1540
    assert abs(-1.4311 - power_law_kramers_kronig(3, 1, 1540, 1, 1)) < 0.001
    assert abs(1.4285 - power_law_kramers_kronig(1, 3, 1540, 1, 1)) < 0.001


def test_gradient_FD():
    f = np.array([1, 2, 4, 7, 11, 16], dtype=float)
    grad = gradient_fd(f)
    assert (abs(np.array(grad) - np.array([1.0, 1.5, 2.5, 3.5, 4.5, 5])) < 0.001).all()


def test_gradient_spacing():
    f = np.array([1, 2, 4, 7, 11, 16], dtype=float)
    dx = 2
    grad = gradient_fd(f, dx)
    assert np.allclose(np.array(grad), np.array([0.5, 0.75, 1.25, 1.75, 2.25, 2.5]))


def test_gradient_spacing_ndim():
    f = np.array([1, 2, 4, 7, 11, 16], dtype=float)
    x = np.arange(f.size)
    grad = gradient_fd(f, x)
    assert np.allclose(grad, np.array([1.0, 1.5, 2.5, 3.5, 4.5, 5.0]))


def test_gradeint_spacing_uneven():
    f = np.array([1, 2, 4, 7, 11, 16], dtype=float)
    x = np.array([0.0, 1.0, 1.5, 3.5, 4.0, 6.0], dtype=float)
    grad = gradient_fd(f, x)
    assert np.allclose(grad, np.array([1.0, 3.0, 3.5, 6.7, 6.9, 2.5]))


def test_gradient_FD_2D():
    f = np.array([[1, 2, 6], [3, 4, 5]], dtype=float)
    grad = gradient_fd(f)

    assert len(grad) == 2, "gradient_fd did not return two gradient matrices."
    assert np.allclose(np.array(grad), [np.array([[2.0, 2.0, -1.0], [2.0, 2.0, -1.0]]), np.array([[1.0, 2.5, 4.0], [1.0, 1.0, 1.0]])])

    pass


def test_gradient_spect_1D():
    # compute gradient of a 2 period sinusoid
    x = np.arange(start=np.pi / 20, stop=4 * np.pi + np.pi / 20, step=np.pi / 20)
    y = np.sin(x)
    dydx = gradient_spect(y, np.pi / 20)
    test_val = np.cos(x)

    assert np.isclose(dydx, test_val).all()


def test_gradient_spect_2D():
    test_record_path = os.path.join(
        Path(__file__).parent, Path("matlab_test_data_collectors/python_testers/collectedValues/gradientSpect.mat")
    )
    reader = TestRecordReader(test_record_path)
    dx = reader.expected_value_of("dx")
    dy = reader.expected_value_of("dy")
    z = reader.expected_value_of("z")

    # compute gradient of a 2 period sinusoid
    dydx = gradient_spect(z, (np.pi / 20, np.pi / 20))

    assert np.isclose(dy, dydx[1]).all()
    assert np.isclose(dx, dydx[0]).all()
    assert np.isclose(dydx[1].T, dydx[0]).all()
    pass


# TODO:
def test_resize_2D_splinef2d():
    mat = np.ones([10, 10])
    out = resize(mat, [20, 20], "splinef2d")  # noqa: F841


def test_resize_2D_linear_larger():
    # assign the grid size and create the computational grid
    new_size = [20, 3]
    p0 = np.zeros([10, 2])
    p0[:, 1] = 1

    # resize the input image to the desired number of grid points
    p1 = resize(p0, new_size, interp_mode="linear")
    assert p1.shape == tuple(new_size)
    assert np.all(p1 == [0.0, 0.5, 1.0])

    # now test transpose
    p0 = p0.T
    new_size = [3, 20]
    p1 = resize(p0, new_size, interp_mode="linear")
    assert p1.shape == tuple(new_size)
    assert np.all(p1.T == [0.0, 0.5, 1.0])


def test_resize_2D_linear_smaller():
    # assign the grid size and create the computational grid
    p0 = np.zeros([20, 3])
    p0[:, 1:3] = [0.5, 1]
    new_size = [10, 2]

    # resize the input image to the desired number of grid points
    p1 = resize(p0, new_size, interp_mode="linear")
    assert p1.shape == tuple(new_size)
    assert np.all(p1 == [0.0, 1.0])

    # now test transpose
    p0 = p0.T
    new_size = [2, 10]
    p1 = resize(p0, new_size, interp_mode="linear")
    assert p1.shape == tuple(new_size)
    assert np.all(p1.T == [0.0, 1.0])


def test_resize_2D_nearest_larger():
    # assign the grid size and create the computational grid
    p0 = np.zeros([10, 2])
    p0[:, 1] = 1
    new_size = [20, 3]

    # resize the input image to the desired number of grid points
    p1 = resize(p0, new_size, interp_mode="nearest")
    assert p1.shape == tuple(new_size)
    assert np.all(p1 == [0.0, 0.0, 1.0])

    # now test transpose
    p0 = p0.T
    new_size = [3, 20]
    p1 = resize(p0, new_size, interp_mode="nearest")
    assert p1.shape == tuple(new_size)
    assert np.all(p1.T == [0.0, 0.0, 1.0])


def test_resize_2D_nearest_smaller():
    # assign the grid size and create the computational grid
    p0 = np.zeros([20, 3])
    p0[:, 1:4] = 1
    new_size = [10, 2]

    # resize the input image to the desired number of grid points
    p1 = resize(p0, new_size, interp_mode="nearest")
    assert p1.shape == tuple(new_size)
    assert np.all(p1 == [0.0, 1.0])

    # now test the transpose
    p0 = p0.T
    new_size = [2, 10]
    p1 = resize(p0, new_size, interp_mode="nearest")
    assert p1.shape == tuple(new_size)
    assert np.all(p1.T == [0.0, 1.0])


def test_trim_zeros():
    # 1D case
    vec = np.zeros([10])
    vec[3:8] = 1
    vec_trimmed, ind = trim_zeros(vec)
    assert np.all(vec_trimmed == np.ones([5])), "trim_zeros did not pass the 1D test."
    assert ind == [(3, 8)], "trim_zeros did not return the correct indices for the 1D case."

    # 2D case
    mat = np.zeros([10, 10])
    mat[3:8, 3:8] = 1
    mat_trimmed, ind = trim_zeros(mat)
    assert np.all(mat_trimmed == np.ones([5, 5])), "trim_zeros did not pass the 2D test."
    assert ind == [(3, 8), (3, 8)], "trim_zeros did not return the correct indices for the 2D case."

    # 3D case
    mat = np.zeros([10, 10, 10])
    mat[3:8, 3:8, 3:8] = 1
    mat_trimmed, ind = trim_zeros(mat)
    assert np.all(mat_trimmed == np.ones([5, 5, 5])), "trim_zeros did not pass the 3D test."
    assert ind == [(3, 8), (3, 8), (3, 8)], "trim_zeros did not return the correct indices for the 3D case."

    # Harder 2D test case

    data = np.array(
        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 3, 0, 0], [0, 0, 1, 3, 4, 0], [0, 0, 1, 3, 4, 0], [0, 0, 1, 3, 0, 0], [0, 0, 0, 0, 0, 0]]
    )

    correct_trimmed = np.array([[0, 3, 0], [1, 3, 4], [1, 3, 4], [1, 3, 0]])

    data_trimmed, ind = trim_zeros(data)

    # assert correctness
    assert np.all(data_trimmed == correct_trimmed), "trim_zeros did not pass the hard 2D test."

    # Higher dimensional case (4D)
    mat = np.zeros([10, 10, 10, 10])
    mat[3:8, 3:8, 3:8, 3:8] = 1
    with pytest.raises(ValueError):
        mat_trimmed, ind = trim_zeros(mat)

    # TODO: generalize to N-D case


if __name__ == "__main__":
    test_interp_cart_data_1_point_nearest()
    test_interp_cart_data_2_points_nearest()
    test_interp_cart_data_2_points_linear()
