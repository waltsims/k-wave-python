import functools
import logging
import dataclasses

import numpy as np

from kwave.utils.kwave_array import kWaveArray, Element


def recursive_getattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def check_element_equality(
    actual_element: Element,
    expected_element: Element,
) -> bool:
    for field in dataclasses.fields(expected_element):
        expected = getattr(expected_element, field.name)
        actual = getattr(actual_element, field.name)
        if isinstance(expected, np.ndarray):
            if not np.allclose(actual, expected):
                return False
        elif isinstance(expected, float):
            if not np.isclose(actual, expected):
                return False
        else:
            if actual != expected:
                return False
    return True


def check_kgrid_equality(kgrid_object: kWaveArray, expected_kgrid_dict: dict):
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
            "zn_vec_sgz": "n_vec_sg.z",
        }

        mapped_key = matlab_to_python_mapping.get(key, key)
        if mapped_key in [
            "k_vec.y",
            "k_vec.z",
            "k_max.y",
            "k_max.z",
            "n_vec.y",
            "n_vec.z",
            "n_vec_sg.y",
            "n_vec_sg.z",
            "dudn.y",
            "dudn.z",
            "dudn.z",
            "dudn_sg.y",
            "dudn_sg.z",
            "n_vec_sg.y",
            "n_vec_sg.z",
            "y_vec",
            "z_vec",
            "ky",
            "kz",
            "yn",
            "zn",
        ]:
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
            are_equal = actual_value == expected_value

        if not are_equal:
            logging.log(logging.INFO, "Following property does not match:")
            logging.log(logging.INFO, f"\tkey: {key}, mapped_key: {mapped_key}")
            logging.log(logging.INFO, f"\t\texpected: {expected_value}")
            logging.log(logging.INFO, f"\t\tactual: {actual_value}")
            are_totally_equal = False

    assert are_totally_equal


def check_kwave_array_equality(kwave_array_object: kWaveArray, expected_kwave_array_dict: dict):
    are_totally_equal = True

    for key, expected_value in expected_kwave_array_dict.items():
        matlab_to_python_mapping = {}

        mapped_key = matlab_to_python_mapping.get(key, key)
        if mapped_key in []:
            ignore_if_nan = True
        else:
            ignore_if_nan = False

        actual_value = recursive_getattr(kwave_array_object, mapped_key, None)

        if key == "elements":
            are_equal = True
            if isinstance(expected_value, dict):
                expected_value = [Element(**expected_value)]
                for actual, expected in zip(actual_value, expected_value):
                    are_equal &= check_element_equality(
                        actual_element=actual,
                        expected_element=expected,
                    )
            elif isinstance(expected_value, list):
                expected_value = [Element(**val) for val in expected_value]
                for actual, expected in zip(actual_value, expected_value):
                    are_equal &= check_element_equality(
                        actual_element=actual,
                        expected_element=expected,
                    )
        else:
            actual_value = np.squeeze(actual_value)
            expected_value = np.array(expected_value)

            if ignore_if_nan and expected_value.size == 1 and (np.isnan(actual_value)) and (expected_value == 0):
                are_equal = True
            elif (actual_value is None) and (expected_value is not None):
                are_equal = False
            elif np.size(actual_value) >= 2 or np.size(expected_value) >= 2:
                are_equal = np.allclose(actual_value, expected_value)
            elif np.size(actual_value) == np.size(expected_value) == 0:
                are_equal = True
            else:
                # check if both are float (either native or numpy), if so, check if they are close
                if isinstance(actual_value, float) and isinstance(expected_value, float):
                    are_equal = np.isclose(actual_value, expected_value)
                else:
                    are_equal = actual_value == expected_value

        if not are_equal:
            logging.log(logging.INFO, "Following property does not match:")
            logging.log(logging.INFO, f"\tkey: {key}, mapped_key: {mapped_key}")
            logging.log(logging.INFO, f"\t\texpected: {expected_value}")
            logging.log(logging.INFO, f"\t\tactual: {actual_value}")
            are_totally_equal = False

    assert are_totally_equal
