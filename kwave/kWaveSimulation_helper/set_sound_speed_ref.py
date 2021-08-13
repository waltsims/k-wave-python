from kwave import kWaveMedium
import numpy as np


def set_sound_speed_ref(medium: kWaveMedium, elastic_code: bool, kspace_elastic_code: bool):
    """
        select the reference sound speed used in the k-space operator
        based on the heterogeneous sound speed map
    Args:
        medium:
        elastic_code:
        kspace_elastic_code:

    Returns:

    """
    if not elastic_code:
        return get_ordinary_sound_speed_ref(medium)
    elif not kspace_elastic_code:  # pragma: no cover
        return get_pstd_elastic_sound_speed_ref(medium)
    else:  # pragma: no cover
        return get_kspace_elastic_sound_speed_ref(medium)


def get_ordinary_sound_speed_ref(medium):
    """
        calculate the reference sound speed for the fluid code, using the
        maximum by default which ensures the model is unconditionally stable
    Args:
        medium:

    Returns:

    """
    c_ref = _get_sound_speed_ref(medium.sound_speed_ref, medium.sound_speed)
    print('  reference sound speed: ', c_ref, 'm/s')
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
    print('  reference sound speed: ', c_ref, 'm/s')
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
    print('  reference sound speed (compression): ', c_ref_compression, 'm/s')

    c_ref_shear = _get_sound_speed_ref(medium.sound_speed_ref_shear, medium.sound_speed_shear)
    print('  reference sound speed (shear): ', c_ref_shear, 'm/s')

    return None, c_ref_compression, c_ref_shear


def _get_sound_speed_ref(reference, speed):
    reductions = {
        'min': np.min,
        'max': np.max,
        'mean': np.mean
    }

    if reference is not None:
        # if reference is defined, check whether it is a scalar or 'reduction'
        if np.isscalar(reference):
            c_ref = reference
        else:
            c_ref = reductions[reference](speed)
    else:
        c_ref = reductions['max'](speed)

    print('  reference sound speed: ', c_ref, 'm/s')
    return float(c_ref)
