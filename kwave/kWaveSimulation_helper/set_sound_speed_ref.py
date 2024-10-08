import logging
import numpy as np

from kwave.kmedium import kWaveMedium
from kwave.options.simulation_options import SimulationType


def set_sound_speed_ref(medium: kWaveMedium, simulation_type: SimulationType):
    """
    select the reference sound speed used in the k-space operator
    based on the heterogeneous sound speed map
    Args:
        medium: kWaveMedium object
        simulation_type: Simulation type (fluid, axisymmetric, elastic, elastic with k-space correction)
    Returns:

    """
    if not simulation_type.is_elastic_simulation():
        return get_ordinary_sound_speed_ref(medium)
    elif simulation_type == SimulationType.ELASTIC:  # pragma: no cover
        return get_pstd_elastic_sound_speed_ref(medium)
    elif simulation_type == SimulationType.ELASTIC_WITH_KSPACE_CORRECTION:  # pragma: no cover
        return get_kspace_elastic_sound_speed_ref(medium)
    else:
        raise NotImplementedError("Non-supported simulation type: " + str(simulation_type))


def get_ordinary_sound_speed_ref(medium):
    """
    calculate the reference sound speed for the fluid code, using the
    maximum by default which ensures the model is unconditionally stable
    Args:
        medium:

    Returns:

    """
    c_ref = _get_sound_speed_ref(medium.sound_speed_ref, medium.sound_speed)
    logging.log(logging.INFO, f"  reference sound speed: {c_ref} m/s")
    return c_ref, None, None


def get_pstd_elastic_sound_speed_ref(medium: kWaveMedium):  # pragma: no cover
    """
    in the pstd elastic case, the reference sound speed is only used to
    calculate the PML absorption, so just use the compressional wave speed
    Args:
        medium:

    Returns:

    """
    c_ref = _get_sound_speed_ref(medium.sound_speed_ref, medium.sound_speed_compression)
    logging.log(logging.INFO, f"  reference sound speed: {c_ref} m/s")
    return c_ref, None, None


def get_kspace_elastic_sound_speed_ref(medium: kWaveMedium):  # pragma: no cover
    """
    in the k-space elastic case, there are two reference sound speeds for
    the compressional and shear waves, so compute them seperately
    Args:
        medium:

    Returns:

    """
    c_ref_compression = _get_sound_speed_ref(medium.sound_speed_ref_compression, medium.sound_speed_compression)
    logging.log(logging.INFO, f"  reference sound speed (compression): {c_ref_compression} m/s")

    c_ref_shear = _get_sound_speed_ref(medium.sound_speed_ref_shear, medium.sound_speed_shear)
    logging.log(logging.INFO, f"  reference sound speed (shear): {c_ref_shear} m/s")

    return None, c_ref_compression, c_ref_shear


def _get_sound_speed_ref(reference, speed):
    reductions = {"min": np.min, "max": np.max, "mean": np.mean}

    if reference is not None:
        # if reference is defined, check whether it is a scalar or 'reduction'
        if np.isscalar(reference):
            c_ref = reference
        else:
            c_ref = reductions[reference](speed)
    else:
        c_ref = reductions["max"](speed)

    logging.log(logging.INFO, f"  reference sound speed: {c_ref} m/s")
    return float(c_ref)
