import numpy as np
import pytest
import itertools
from kwave.utils.mapgen import water_sound_speed, water_non_linearity, water_density, water_absorption


def test_water_absorption():
    expected_outputs = np.array(
        [
            4.92700598e-03,
            3.10935008e-03,
            2.18550989e-03,
            1.65777741e-03,
            1.29686047e-03,
            1.03350585e-03,
            8.60898169e-04,
            1.97080239e-02,
            1.24374003e-02,
            8.74203957e-03,
            6.63110962e-03,
            5.18744190e-03,
            4.13402339e-03,
            3.44359267e-03,
            4.43430538e-02,
            2.79841507e-02,
            1.96695890e-02,
            1.49199967e-02,
            1.16717443e-02,
            9.30155264e-03,
            7.74808352e-03,
            7.88320956e-02,
            4.97496012e-02,
            3.49681583e-02,
            2.65244385e-02,
            2.07497676e-02,
            1.65360936e-02,
            1.37743707e-02,
            1.23175149e-01,
            7.77337519e-02,
            5.46377473e-02,
            4.14444351e-02,
            3.24215119e-02,
            2.58376462e-02,
            2.15224542e-02,
            1.77372215e-01,
            1.11936603e-01,
            7.86783562e-02,
            5.96799866e-02,
            4.66869771e-02,
            3.72062105e-02,
            3.09923341e-02,
            2.41423293e-01,
            1.52358154e-01,
            1.07089985e-01,
            8.12310929e-02,
            6.35461633e-02,
            5.06417866e-02,
            4.21840103e-02,
            3.15328382e-01,
            1.98998405e-01,
            1.39872633e-01,
            1.06097754e-01,
            8.29990704e-02,
            6.61443743e-02,
            5.50974828e-02,
            3.99087484e-01,
            2.51857356e-01,
            1.77026301e-01,
            1.34279970e-01,
            1.05045698e-01,
            8.37139737e-02,
            6.97327517e-02,
            4.92700598e-01,
            3.10935008e-01,
            2.18550989e-01,
            1.65777741e-01,
            1.29686047e-01,
            1.03350585e-01,
            8.60898169e-02,
            5.96167723e-01,
            3.76231359e-01,
            2.64446697e-01,
            2.00591066e-01,
            1.56920117e-01,
            1.25054208e-01,
            1.04168678e-01,
            7.09488860e-01,
            4.47746411e-01,
            3.14713425e-01,
            2.38719946e-01,
            1.86747908e-01,
            1.48824842e-01,
            1.23969336e-01,
            8.32664010e-01,
            5.25480163e-01,
            3.69351172e-01,
            2.80164382e-01,
            2.19169420e-01,
            1.74662488e-01,
            1.45491790e-01,
            9.65693171e-01,
            6.09432615e-01,
            4.28359939e-01,
            3.24924372e-01,
            2.54184653e-01,
            2.02567146e-01,
            1.68736041e-01,
            1.10857634e00,
            6.99603767e-01,
            4.91739726e-01,
            3.72999916e-01,
            2.91793607e-01,
            2.32538816e-01,
            1.93702088e-01,
        ]
    )

    input_temps = np.arange(0, 70, 10, dtype=float)
    input_fsampling = np.arange(1, 16, dtype=float)
    # idx : int = 0

    # for fs in input_fsampling:
    #    for temp in input_temps:
    for idx, (f_s, temp) in enumerate(itertools.product(input_fsampling, input_temps)):
        assert np.isclose(expected_outputs[idx], water_absorption(f_s, temp)), "Expected value deviates from expected absorption value "

    n: int = np.size(input_temps)
    for idx, f_s in enumerate(input_fsampling):
        assert np.allclose(expected_outputs[idx * n : (idx + 1) * n], water_absorption(f_s, input_temps)), (
            "Expected value deviates from expected " "absorption value "
        )
    return


def test_water_density():
    expected_values = np.array(
        [
            999.96069266,
            999.93636812,
            999.89732966,
            999.84393424,
            999.7765297,
            999.69545466,
            999.60103861,
            999.49360184,
            999.37345547,
            999.24090148,
            999.09623264,
            998.93973257,
            998.77167573,
            998.59232737,
            998.40194361,
            998.20077138,
            997.98904845,
            997.76700339,
            997.53485564,
            997.29281544,
            997.04108388,
            996.77985284,
            996.50930509,
            996.22961417,
            995.94094449,
            995.64345126,
            995.33728055,
            995.02256923,
            994.69944501,
            994.36802644,
            994.02842288,
            993.68073453,
            993.32505242,
            992.9614584,
            992.59002517,
            992.21081622,
        ]
    )
    input_temp = np.arange(5, 41, dtype=float)

    for idx, value in enumerate(expected_values):
        assert np.isclose(value, water_density(input_temp[idx])), "The expected value deviates from water_density output"

    assert np.allclose(expected_values, water_density(input_temp)), "An expected value deviates from water_density output vector"
    assert np.allclose(
        expected_values.reshape(6, 6), water_density(input_temp.reshape(6, 6))
    ), "An expected value deviates from water_density output matrix"
    with pytest.raises(ValueError):
        _ = water_density(3.0)


def test_water_sound_speed():
    expected_values = np.array(
        [
            1402.385,
            1407.36634896,
            1412.23406799,
            1416.99007945,
            1421.63627279,
            1426.17450488,
            1430.60660038,
            1434.93435204,
            1439.15952103,
            1443.28383731,
            1447.30899994,
            1451.2366774,
            1455.06850798,
            1458.80610003,
            1462.45103239,
            1466.00485465,
            1469.46908751,
            1472.84522313,
            1476.13472544,
            1479.33903048,
            1482.45954675,
            1485.49765554,
            1488.45471124,
            1491.3320417,
            1494.13094858,
            1496.85270762,
            1499.49856905,
            1502.06975787,
            1504.56747424,
            1506.99289373,
            1509.34716775,
            1511.63142381,
            1513.8467659,
            1515.99427481,
            1518.07500845,
            1520.09000221,
            1522.04026926,
            1523.92680095,
            1525.75056705,
            1527.51251618,
            1529.21357606,
            1530.85465393,
            1532.4366368,
            1533.96039184,
            1535.42676671,
            1536.83658985,
            1538.19067089,
            1539.4898009,
            1540.73475281,
            1541.92628166,
            1543.065125,
            1544.1520032,
            1545.18761977,
            1546.17266173,
            1547.10779991,
            1547.9936893,
            1548.83096938,
            1549.62026447,
            1550.36218404,
            1551.05732306,
            1551.70626234,
            1552.30956883,
            1552.86779602,
            1553.3814842,
            1553.85116085,
            1554.27734094,
            1554.66052729,
            1555.00121089,
            1555.29987123,
            1555.55697664,
            1555.77298465,
            1555.94834228,
            1556.0834864,
            1556.17884406,
            1556.23483284,
            1556.25186113,
            1556.23032855,
            1556.17062622,
            1556.07313709,
            1555.93823634,
            1555.76629165,
            1555.55766354,
            1555.31270576,
            1555.03176554,
            1554.71518402,
            1554.3632965,
            1553.97643281,
            1553.55491766,
            1553.09907095,
            1552.60920812,
            1552.08564046,
            1551.52867549,
            1550.93861723,
            1550.31576661,
            1549.66042173,
            1548.97287827,
        ]
    )
    input_temp = np.arange(96, dtype=float)

    for idx, value in enumerate(expected_values):
        assert np.isclose(value, water_sound_speed(idx)), "Expected value deviates from water_sound_speed output"

    assert np.allclose(expected_values, water_sound_speed(input_temp)), "Expected value deviates from water_sound_speed output vector"
    assert np.allclose(
        expected_values.reshape(12, 8), water_sound_speed(input_temp.reshape(12, 8))
    ), "Expected value deviates from water_sound_speed output matrix"

    with pytest.raises(ValueError):
        _ = water_sound_speed(96.0)


def test_water_non_linearity():
    expected_values = np.array(
        [
            4.18653394,
            4.23941757,
            4.29049232,
            4.33981942,
            4.38745897,
            4.43347001,
            4.47791044,
            4.52083708,
            4.56230564,
            4.60237073,
            4.64108587,
            4.67850347,
            4.71467483,
            4.74965016,
            4.78347857,
            4.81620807,
            4.84788556,
            4.87855684,
            4.90826661,
            4.93705849,
            4.96497496,
            4.99205743,
            5.01834619,
            5.04388044,
            5.06869829,
            5.09283671,
            5.11633161,
            5.13921778,
            5.16152891,
            5.18329759,
            5.2045553,
            5.22533245,
            5.24565831,
            5.26556108,
            5.28506782,
            5.30420454,
            5.32299611,
            5.34146631,
            5.35963783,
            5.37753224,
            5.39517003,
            5.41257056,
            5.42975212,
            5.44673187,
            5.46352591,
            5.48014918,
            5.49661558,
            5.51293787,
            5.52912771,
            5.54519568,
            5.56115125,
            5.57700277,
            5.59275752,
            5.60842166,
            5.62400026,
            5.63949726,
            5.65491554,
            5.67025686,
            5.68552187,
            5.70071013,
            5.71582009,
            5.73084912,
            5.74579346,
            5.76064827,
            5.77540761,
            5.79006441,
            5.80461054,
            5.81903673,
            5.83333264,
            5.84748682,
            5.8614867,
            5.87531864,
            5.88896788,
            5.90241855,
            5.9156537,
            5.92865527,
            5.9414041,
            5.95387992,
            5.96606138,
            5.977926,
            5.98945022,
            6.00060938,
            6.0113777,
            6.02172833,
            6.03163328,
            6.04106348,
            6.04998878,
            6.05837788,
            6.06619841,
            6.07341691,
            6.0799988,
            6.08590839,
            6.0911089,
            6.09556247,
            6.0992301,
            6.10207171,
            6.10404612,
            6.10511105,
            6.10522312,
            6.10433782,
            6.10240959,
        ]
    )
    for idx, value in enumerate(expected_values):
        assert np.isclose(value, water_non_linearity(idx)), "The expected value deviates from water_non_linearity output"

    input_temp = np.arange(101, dtype=float)
    assert np.allclose(
        expected_values, water_non_linearity(input_temp)
    ), "An expected value deviates from the output vector from water_non_linearity"

    assert np.allclose(
        expected_values[1:101].reshape(10, 10), water_non_linearity(input_temp[1:101].reshape(10, 10))
    ), "An expected values deviates from the output matrix from water_non_linearity output matrix"

    with pytest.raises(ValueError):
        _ = water_non_linearity(101.0)
