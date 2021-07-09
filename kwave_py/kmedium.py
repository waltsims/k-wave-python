from dataclasses import dataclass
from warnings import warn

import numpy as np
import kwave_py.utils.misc as util
import kwave_py.utils.checkutils


@dataclass
class kWaveMedium(object):
    sound_speed                 : np.array             # [m/s] | required to be defined
    sound_speed_ref             : np.array = None
    density                     : np.array = None      # [kg/m^3]
    alpha_coeff                 : np.array = None      # [dB/(MHz^y cm)]
    alpha_power                 : np.array = None
    alpha_mode                  : np.array = None
    alpha_filter                : np.array = None
    alpha_sign                  : np.array = None
    BonA                        : np.array = None
    absorbing                   : bool     = False
    stokes                      : bool     = False

    def __post_init__(self):
        self.sound_speed                    = np.atleast_1d(self.sound_speed)

    def check_fields(self, kgrid_shape):
        # check the absorption mode input is valid
        if self.alpha_mode is not None:
            assert self.alpha_mode in ['no_absorption', 'no_dispersion', 'stokes'], \
                "medium.alpha_mode must be set to 'no_absorption', 'no_dispersion', or 'stokes'."

        # check the absorption filter input is valid
        if self.alpha_filter is not None and not (self.alpha_filter.shape == kgrid_shape).all():
            raise ValueError('medium.alpha_filter must be the same size as the computational grid.')

        # check the absorption sign input is valid
        if self.alpha_sign is not None and (not kwave_py.utils.checkutils.is_number(self.alpha_sign) or (self.alpha_sign.size != 2)):
            raise ValueError('medium.alpha_sign must be given as a 2 element numerical array controlling absorption and dispersion, respectively.')

        # check alpha_coeff is non-negative and real
        if not np.all(np.isreal(self.alpha_coeff)) or np.any(self.alpha_coeff < 0):
            raise ValueError('medium.alpha_coeff must be non-negative and real.')

    def is_defined(self, *fields):
        results = []
        for f in fields:
            results.append(getattr(self, f) is not None)
        return results

    def ensure_defined(self, *fields):
        for f in fields:
            assert getattr(self, f) is not None, f'The field {f} must be not be None'

    def is_nonlinear(self):
        """
            Check if the medium is nonlinear
        Returns:
            whether the fluid simulation is nonlinear
        """
        return self.BonA is not None

    def set_absorbing(self, is_absorbing, is_stokes=False):
        # only stokes absorption is supported in the axisymmetric code
        self.absorbing, self.stokes = is_absorbing, is_stokes
        if is_absorbing:
            if is_stokes:
                self._check_absorbing_with_stokes()
            else:
                self._check_absorbing_without_stokes()

    def _check_absorbing_without_stokes(self):
        # enforce both absorption parameters
        self.ensure_defined('alpha_coeff', 'alpha_power')

        # check y is a scalar
        assert np.isscalar(self.alpha_power), 'medium.alpha_power must be scalar.'

        # check y is real and within 0 to 3
        assert np.all(np.isreal(self.alpha_coeff)) and 0 < self.alpha_power < 3, \
            'medium.alpha_power must be a real number between 0 and 3.'

        # display warning if y is close to 1 and the dispersion term has not been set to zero
        if self.alpha_mode != 'no_dispersion':
            assert self.alpha_power != 1, \
                """The power law dispersion term in the equation of state is not valid for medium.alpha_power = 1.
                This error can be avoided by choosing a power law exponent close to, but not exactly, 1.
                If modelling acoustic absorption for medium.alpha_power = 1 is important and modelling dispersion is not
                critical, this error can also be avoided by setting medium.alpha_mode to 'no_dispersion'"""

    def _check_absorbing_with_stokes(self):
        # enforce absorption coefficient
        self.ensure_defined('alpha_coeff')

        # give warning if y is specified
        if self.alpha_power is not None and (self.alpha_power.size != 1 or self.alpha_power != 2):
            warn('WARNING: the axisymmetric code and stokes absorption assume alpha_power = 2, user value ignored.')

        # overwrite y value
        self.alpha_power = 2

        # don't allow medium.alpha_mode with the axisymmetric code
        if self.alpha_mode is not None and (self.alpha_mode in ['no_absorption', 'no_dispersion']):
            raise NotImplementedError('Input option medium.alpha_mode is not supported with the axisymmetric code '
                                      'or medium.alpha_mode = ''stokes''.')

        # don't allow alpha_filter with stokes absorption (no variables are applied in k-space)
        assert self.alpha_filter is None, \
            "Input option medium.alpha_filter is not supported with the axisymmetric code " \
            "or medium.alpha_mode = 'stokes'. "

    ##########################################
    # Elastic-code related properties - raise error when accessed
    ##########################################
    _ELASTIC_CODE_ACCESS_ERROR_TEXT_ = 'Elastic simulation and related properties are not supported!'

    @property
    def sound_speed_shear(self):  # pragma: no cover
        raise NotImplementedError(self._ELASTIC_CODE_ACCESS_ERROR_TEXT_)

    @property
    def sound_speed_ref_shear(self):  # pragma: no cover
        raise NotImplementedError(self._ELASTIC_CODE_ACCESS_ERROR_TEXT_)

    @property
    def sound_speed_compression(self):  # pragma: no cover
        raise NotImplementedError(self._ELASTIC_CODE_ACCESS_ERROR_TEXT_)

    @property
    def sound_speed_ref_compression(self):  # pragma: no cover
        raise NotImplementedError(self._ELASTIC_CODE_ACCESS_ERROR_TEXT_)

    @property
    def alpha_coeff_compression(self):  # pragma: no cover
        raise NotImplementedError(self._ELASTIC_CODE_ACCESS_ERROR_TEXT_)

    @property
    def alpha_coeff_shear(self):  # pragma: no cover
        raise NotImplementedError(self._ELASTIC_CODE_ACCESS_ERROR_TEXT_)
