from kwave.utils.checkutils import num_dim
from kwave.utils.maputils import hounsfield2density
from kwave.utils.conversionutils import db2neper, neper2db
from kwave.utils.kutils import toneBurst, add_noise
from kwave.utils.filterutils import extract_amp_phase, spect
import numpy as np
from phantominator import shepp_logan


input_signal = np.array([0., 0.00099663, 0.00646706, 0.01316044, 0.01851998,
                         0.02355139, 0.02753738, 0.02966112, 0.02907933, 0.02502059,
                         0.01690274, 0.00445909, -0.01214073, -0.03219275, -0.05440197,
                         -0.0768851, -0.09724674, -0.1127371, -0.12049029, -0.11783127,
                         -0.10262828, -0.07365651, -0.03093011, 0.02404454, 0.0881411,
                         0.15665039, 0.22350821, 0.28172052, 0.3239688, 0.34335346,
                         0.33420925, 0.29290732, 0.21854807, 0.11344786, -0.01666495,
                         -0.16280571, -0.31326688, -0.45451349, -0.5723795, -0.65346175,
                         -0.68657424, -0.66410437, -0.58311018, -0.4460133, -0.2607779,
                         -0.04051683, 0.19747155, 0.43316678, 0.64565891, 0.81517002,
                         0.92505348, 0.96355291, 0.92512102, 0.81114165, 0.62996031,
                         0.39620447, 0.12945356, -0.14760764, -0.41135944, -0.63951227,
                         -0.81322602, -0.91886114, -0.94918152, -0.90389224, -0.78947066,
                         -0.6183277, -0.40740841, -0.1763986, 0.05425963, 0.26532811,
                         0.44050654, 0.56777338, 0.64021864, 0.65631606, 0.61964583,
                         0.53813765, 0.42295011, 0.28713217, 0.14422376, 0.00694453,
                         -0.11390314, -0.21022661, -0.27708217, -0.3127555, -0.31851672,
                         -0.29811426, -0.257092, -0.20202238, -0.139745, -0.07668802,
                         -0.01832983, 0.03116454, 0.06912143, 0.0943617, 0.10705662,
                         0.10847543, 0.10066873, 0.08613128, 0.06748245, 0.04719401,
                         0.02738443, 0.00968836, -0.00479991, -0.01551776, -0.02237439,
                         -0.02565662, -0.0259173, -0.02386159, -0.02024351, -0.01578123,
                         -0.00892002, -0.00185916, -0.])


def test_nepers2db():
    assert abs(neper2db(1.5) - 8.186258123051049e+05) < 1e-6, "Point check of nepers2db incorrect"
    return


def test_db2nepers():
    assert abs(db2neper(1.6) - 2.931742395517710e-06) < 1e-6, "Point check of nepers2db incorrect"
    return


def test_hounsfield2soundspeed():
    ph = shepp_logan(5)

    expected_density = np.array([[-5.68040401, -5.68040401, -5.68040401, -5.68040401, -5.68040401],
                                 [-5.68040401, -5.4752454, -5.37266609, -5.4752454, -5.68040401],
                                 [-5.68040401, -5.4752454, -5.4752454, -5.4752454, -5.68040401],
                                 [-5.68040401, -5.4752454, -5.4752454, -5.4752454, -5.68040401],
                                 [-5.68040401, -5.68040401, -5.68040401, -5.68040401, -5.68040401]])

    assert ((expected_density - hounsfield2density(ph)) < 1).all(), "Generated denisty does not match expected value"

    return


def test_add_noise():
    output = add_noise(input_signal, 5)
    p_sig = np.sqrt(np.mean(input_signal ** 2))
    p_noise = np.sqrt(np.mean((output - input_signal) ** 2))
    snr = 20 * np.log10(p_sig / p_noise)
    assert (abs(5 - snr) < 2), \
        "add_noise produced signal with incorrect SNR, this is a stochastic process. Perhaps test again?"
    return


def test_tone_burst_gaussian():
    # Test simple case
    expected_input_signal = input_signal
    test_input_signal = toneBurst(1.129333333333333e+07, 5e5, 5)
    assert (abs(test_input_signal - expected_input_signal) < 1e-6).all()


def test_tone_burst_rectangular():
    expected_input_signal = np.array([0., 0.27460719, 0.52810066, 0.74099004, 0.89690692,
                                      0.98386329, 0.99517336, 0.92996752, 0.79325925, 0.59555965,
                                      0.35206924, 0.08150928, -0.19531768, -0.45712725, -0.68378967,
                                      -0.85787755, -0.96600578, -0.99986071, -0.95683934, -0.84024945,
                                      -0.65905528, -0.42718833, -0.16247614, 0.11472836, 0.38311173,
                                      0.62203879, 0.81313914, 0.94171966, 0.99789415, 0.97734355,
                                      0.88164792, 0.71816501, 0.49946454, 0.24236174, -0.03337554,
                                      -0.30654667, -0.55614835, -0.76298943, -0.91116654, -0.98928678,
                                      -0.9913437, -0.91717917, -0.77249548, -0.56841691, -0.32063446,
                                      -0.04819939, 0.22794159, 0.48655683, 0.70776216, 0.87454977,
                                      0.97409586, 0.99874663, 0.94660675, 0.82168511, 0.63358655,
                                      0.39677341, 0.12945356, -0.1478196, -0.41372735, -0.64782484,
                                      -0.83211301, -0.95242249, -0.99950306, -0.96973482, -0.86540656,
                                      -0.69453978, -0.47027191, -0.20984623, 0.06671389, 0.33814459,
                                      0.58357635, 0.78413867, 0.92441092, 0.99360796, 0.98640946,
                                      0.90336888, 0.75087096, 0.54064082, 0.28884242, 0.01483578,
                                      -0.26031153, -0.51544426, -0.73094603, -0.89024753, -0.98110057,
                                      -0.99651971, -0.93531942, -0.80220522, -0.60741185, -0.3659164,
                                      -0.09628673, 0.18074614, 0.44388197, 0.67288905, 0.8501597,
                                      0.96206411, 0.99999828, 0.96104557, 0.84820093, 0.67014066,
                                      0.44055527, 0.17709691, -0.09997791, -0.36936573, -0.61035411,
                                      -0.8044142, -0.93662527, -0.99682203, -0.98037612, -0.88855201,
                                      -0.7284098, -0.51226232, -0.25672853])
    # Test envelope selection options [Gaussian, Rectangular, RingUpDown, Not an option]
    test_input_signal = toneBurst(1.129333333333333e+07, 5e5, 5, 'Rectangular')
    assert (abs(test_input_signal - expected_input_signal) < 1e-6).all()


def test_tone_burst_ring():
    # test_input_signal = toneBurst(1.129333333333333e+07, 5e5, 5, 'RingUpDown')
    try:
        test_input_signal = toneBurst(1.129333333333333e+07, 5e5, 5, envelope='BobTheBuilder')
        # TODO: check correctness
    except ValueError as e:
        if str(e) == 'Unknown envelope BobTheBuilder.':
            pass
        else:
            raise e

    return

    # TODO: Test signal length

    # TODO: Test signal offset


def test_num_dim():
    assert num_dim(np.ones((1, 2))) == 1, " num_dim fails for len 3"
    assert num_dim(np.ones((2, 2))) == 2, "num_dim fails for len 2"
    return


def test_spect():
    a = np.array([0, 1.111111111, 2.222222222, 3.3333333333, 4.444444444]) * 1e6
    b = [0.0000, 0.1850, 0.3610, 0.2905, 0.0841]
    c = [3.1416, 1.9199, -0.8727, 2.6180, -0.1745]
    a_t, b_t, c_t =spect(toneBurst(10e6, 2.5e6, 2), 10e6)
    assert (abs(a_t - a) < 0.01).all()
    assert (abs(b_t - b) < 0.0001).all()
    assert (abs(c_t - c) < 0.0001).all()


def test_extract_amp_phase():
    test_signal = toneBurst(sample_freq=10_000_000, signal_freq=2.5 * 1_000_000, num_cycles=2, envelope='Gaussian')
    a_t, b_t, c_t = extract_amp_phase(data=test_signal, Fs=10_000_000, source_freq=2.5 * 10 ** 6)
    a, b, c = 0.6547, -1.8035, 2.5926e06
    assert (abs(a_t - a) < 0.01).all()
    assert (abs(b_t - b) < 0.0001).all()
    assert (abs(c_t - c) < 100).all()

