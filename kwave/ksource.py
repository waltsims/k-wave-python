from dataclasses import dataclass
import logging

import numpy as np

from kwave.kgrid import kWaveGrid
from kwave.utils.matrix import num_dim2


@dataclass
class kSource(object):
    _p0 = None
    #: time varying pressure at each of the source positions given by source.p_mask
    p = None
    #: binary matrix specifying the positions of the time varying pressure source distribution
    p_mask = None
    #: optional input to control whether the input pressure is injected as a mass source or enforced
    # as a dirichlet boundary condition; valid inputs are 'additive' (the default) or 'dirichlet'
    p_mode = None
    #: Pressure reference frequency
    p_frequency_ref = None

    #: time varying particle velocity in the x-direction at each of the source positions given by source.u_mask
    ux = None
    #: time varying particle velocity in the y-direction at each of the source positions given by source.u_mask
    uy = None
    #: time varying particle velocity in the z-direction at each of the source positions given by source.u_mask
    uz = None
    #: binary matrix specifying the positions of the time varying particle velocity distribution
    u_mask = None
    #: optional input to control whether the input velocity is applied as a force source or enforced as a dirichlet
    # boundary condition; valid inputs are 'additive' (the default) or 'dirichlet'
    u_mode = None
    #: Velocity reference frequency
    u_frequency_ref = None

    sxx = None  #: Stress source in x -> x direction
    syy = None  #: Stress source in y -> y direction
    szz = None  #: Stress source in z -> z direction
    sxy = None  #: Stress source in x -> y direction
    sxz = None  #: Stress source in x -> z direction
    syz = None  #: Stress source in y -> z direction
    s_mask = None  #: Stress source mask
    s_mode = None  #: Stress source mode

    def is_p0_empty(self) -> bool:
        """
        Check if the `p0` field is set and not empty
        """
        return self.p0 is None or len(self.p0) == 0 or (np.sum(self.p0 != 0) == 0)

    @property
    def p0(self):
        """
        Initial pressure within the acoustic medium
        """
        return self._p0

    @p0.setter
    def p0(self, val):
        # check size and contents
        if len(val) == 0 or np.sum(val != 0) == 0:
            # if the initial pressure is empty, remove field
            self._p0 = None
        else:
            self._p0 = val

    def validate(self, kgrid: kWaveGrid) -> None:
        """
        Validate the object fields for correctness

        Args:
            kgrid: Instance of `~kwave.kgrid.kWaveGrid` class

        Returns:
            None
        """
        if self.p0 is not None:
            if self.p0.shape != kgrid.k.shape:
                # throw an error if p0 is not the correct size
                raise ValueError("source.p0 must be the same size as the computational grid.")

            # if using the elastic code, reformulate source.p0 in terms of the
            # stress source terms using the fact that source.p = [0.5 0.5] /
            # (2*CFL) is the same as source.p0 = 1
            # if self.elastic_code:
            #     raise NotImplementedError

        # check for a time varying pressure source input
        if self.p is not None:
            # force p_mask to be given if p is given
            assert self.p_mask is not None

            # check mask is the correct size
            # noinspection PyTypeChecker
            if (num_dim2(self.p_mask) != kgrid.dim) or (self.p_mask.shape != kgrid.k.shape):
                raise ValueError("source.p_mask must be the same size as the computational grid.")

            # check mask is not empty
            assert np.sum(self.p_mask) != 0, "source.p_mask must be a binary grid with at least one element set to 1."

            # don't allow both source.p0 and source.p in the same simulation
            # USERS: please contact us via http://www.k-wave.org/forum if this
            # is a problem
            assert self.p0 is None, "source.p0 and source.p can't be defined in the same simulation."

            # check the source mode input is valid
            if self.p_mode is not None:
                assert self.p_mode in [
                    "additive",
                    "dirichlet",
                    "additive-no-correction",
                ], "source.p_mode must be set to ''additive'', ''additive-no-correction'', or ''dirichlet''."

            # check if a reference frequency is defined
            if self.p_frequency_ref is not None:
                # check frequency is a scalar, positive number
                assert np.isscalar(self.p_frequency_ref) and self.p_frequency_ref > 0

                # check frequency is within range
                assert self.p_frequency_ref <= kgrid.k_max_all * np.min(
                    self.medium.sound_speed / 2 * np.pi
                ), "source.p_frequency_ref is higher than the maximum frequency supported by the spatial grid."

                # change source mode to no include k-space correction
                self.p_mode = "additive-no-correction"

            if len(self.p[0]) > kgrid.Nt:
                logging.log(logging.WARN, "  source.p has more time points than kgrid.Nt, remaining time points will not be used.")

            # check if the mask is binary or labelled
            p_unique = np.unique(self.p_mask)

            # create a second indexing variable
            if p_unique.size <= 2 and p_unique.sum() == 1:
                # if more than one time series is given, check the number of time
                # series given matches the number of source elements, or the number
                # of labelled sources
                if self.p.shape[0] > 1 and (len(self.p[:, 0]) != self.p_mask.sum()):
                    raise ValueError("The number of time series in source.p " "must match the number of source elements in source.p_mask.")
            else:
                # check the source labels are monotonic, and start from 1
                if (sum(p_unique[1:] - p_unique[:-1]) != len(p_unique) - 1) or (not any(p_unique == 1)):
                    raise ValueError(
                        "If using a labelled source.p_mask, " "the source labels must be monotonically increasing and start from 1."
                    )
                # make sure the correct number of input signals are given
                if np.size(self.p, 1) != (np.size(p_unique) - 1):
                    raise ValueError(
                        "The number of time series in source.p " "must match the number of labelled source elements in source.p_mask."
                    )

        # check for time varying velocity source input and set source flag
        if any([(getattr(self, k) is not None) for k in ["ux", "uy", "uz", "u_mask"]]):
            # force u_mask to be given
            assert self.u_mask is not None

            # check mask is the correct size
            assert (
                num_dim2(self.u_mask) == kgrid.dim and self.u_mask.shape == kgrid.k.shape
            ), "source.u_mask must be the same size as the computational grid."

            # check mask is not empty
            assert np.array(self.u_mask).sum() != 0, "source.u_mask must be a binary grid with at least one element set to 1."

            # check the source mode input is valid
            if self.u_mode is not None:
                assert self.u_mode in [
                    "additive",
                    "dirichlet",
                    "additive-no-correction",
                ], "source.u_mode must be set to ''additive'', ''additive-no-correction'', or ''dirichlet''."

            # check if a reference frequency is defined
            if self.u_frequency_ref is not None:
                # check frequency is a scalar, positive number
                u_frequency_ref = self.u_frequency_ref
                assert np.isscalar(u_frequency_ref) and u_frequency_ref > 0

                # check frequency is within range
                assert self.u_frequency_ref <= (
                    kgrid.k_max_all * np.min(self.medium.sound_speed) / 2 * np.pi
                ), "source.u_frequency_ref is higher than the maximum frequency supported by the spatial grid."

                # change source mode to no include k-space correction
                self.u_mode = "additive-no-correction"

            if self.ux is not None:
                if self.flag_ux > kgrid.Nt:
                    logging.log(logging.WARN, "  source.ux has more time points than kgrid.Nt, " "remaining time points will not be used.")
            if self.uy is not None:
                if self.flag_uy > kgrid.Nt:
                    logging.log(logging.WARN, "  source.uy has more time points than kgrid.Nt, " "remaining time points will not be used.")
            if self.uz is not None:
                if self.flag_uz > kgrid.Nt:
                    logging.log(logging.WARN, "  source.uz has more time points than kgrid.Nt, " "remaining time points will not be used.")

            # check if the mask is binary or labelled
            u_unique = np.unique(self.u_mask)

            # create a second indexing variable
            if u_unique.size <= 2 and u_unique.sum() == 1:
                # if more than one time series is given, check the number of time
                # series given matches the number of source elements
                ux_size = self.ux[:, 0].size
                uy_size = self.uy[:, 0].size if (self.uy is not None) else None
                uz_size = self.uz[:, 0].size if (self.uz is not None) else None
                u_sum = np.sum(self.u_mask)
                if (self.flag_ux and (ux_size > 1)) or (self.flag_uy and (uy_size > 1)) or (self.flag_uz and (uz_size > 1)):
                    if (
                        (self.flag_ux and (ux_size != u_sum))
                        and (self.flag_uy and (uy_size != u_sum))
                        or (self.flag_uz and (uz_size != u_sum))
                    ):
                        raise ValueError(
                            "The number of time series in source.ux (etc) " "must match the number of source elements in source.u_mask."
                        )

                # if more than one time series is given, check the number of time
                # series given matches the number of source elements
                if (self.flag_ux and (ux_size > 1)) or (self.flag_uy and (uy_size > 1)) or (self.flag_uz and (uz_size > 1)):
                    if (
                        (self.flag_ux and (ux_size != u_sum))
                        or (self.flag_uy and (uy_size != u_sum))
                        or (self.flag_uz and (uz_size != u_sum))
                    ):
                        raise ValueError(
                            "The number of time series in source.ux (etc) " "must match the number of source elements in source.u_mask."
                        )
            else:
                raise NotImplementedError

                # check the source labels are monotonic, and start from 1
                # if (sum(u_unique(2:end) - u_unique(1:end-1)) != (numel(u_unique) - 1)) or (~any(u_unique == 1))
                if eng.eval("(sum(u_unique(2:end) - " "u_unique(1:end-1)) ~= " "(numel(u_unique) - 1)) " "|| " "(~any(u_unique == 1))"):
                    raise ValueError(
                        "If using a labelled source.u_mask, " "the source labels must be monotonically increasing and start from 1."
                    )

                # if more than one time series is given, check the number of time
                # series given matches the number of source elements
                # if (flgs.source_ux and (size(source.ux, 1) != (numel(u_unique) - 1))) or
                #   (flgs.source_uy and (size(source.uy, 1) != (numel(u_unique) - 1))) or
                #   (flgs.source_uz and (size(source.uz, 1) != (numel(u_unique) - 1)))
                if eng.eval(
                    "(flgs.source_ux && (size(source.ux, 1) ~= (numel(u_unique) - 1))) "
                    "|| (flgs.source_uy && (size(source.uy, 1) ~= (numel(u_unique) - 1))) "
                    "|| "
                    "(flgs.source_uz && (size(source.uz, 1) ~= (numel(u_unique) - 1)))"
                ):
                    raise ValueError(
                        "The number of time series in source.ux (etc) "
                        "must match the number of labelled source elements in source.u_mask."
                    )

        # check for time varying stress source input and set source flag
        if any([(getattr(self, k) is not None) for k in ["sxx", "syy", "szz", "sxy", "sxz", "syz", "s_mask"]]):
            # force s_mask to be given
            enforce_fields(self, "s_mask")

            # check mask is the correct size
            # if (numDim(source.s_mask) != kgrid.dim) or (all(size(source.s_mask) != size(kgrid.k)))
            if eng.eval("(numDim(source.s_mask) ~= kgrid.dim) || (all(size(source.s_mask) ~= size(kgrid.k)))"):
                raise ValueError("source.s_mask must be the same size as the computational grid.")

            # check mask is not empty
            assert np.array(eng.getfield(source, "s_mask")) != 0, "source.s_mask must be a binary grid with at least one element set to 1."

            # check the source mode input is valid
            if eng.isfield(source, "s_mode"):
                assert eng.getfield(source, "s_mode") in [
                    "additive",
                    "dirichlet",
                ], "source.s_mode must be set to ''additive'' or ''dirichlet''."
            else:
                eng.setfield(source, "s_mode", self.SOURCE_S_MODE_DEF)

            # set source flgs to the length of the sources, this allows the
            # inputs to be defined independently and be of any length
            if self.sxx is not None and self_sxx > k_Nt:
                logging.log(logging.WARN, "  source.sxx has more time points than kgrid.Nt," " remaining time points will not be used.")
            if self.syy is not None and self_syy > k_Nt:
                logging.log(logging.WARN, "  source.syy has more time points than kgrid.Nt," " remaining time points will not be used.")
            if self.szz is not None and self_szz > k_Nt:
                logging.log(logging.WARN, "  source.szz has more time points than kgrid.Nt," " remaining time points will not be used.")
            if self.sxy is not None and self_sxy > k_Nt:
                logging.log(logging.WARN, "  source.sxy has more time points than kgrid.Nt," " remaining time points will not be used.")
            if self.sxz is not None and self_sxz > k_Nt:
                logging.log(logging.WARN, "  source.sxz has more time points than kgrid.Nt," " remaining time points will not be used.")
            if self.syz is not None and self_syz > k_Nt:
                logging.log(logging.WARN, "  source.syz has more time points than kgrid.Nt," " remaining time points will not be used.")

            # create an indexing variable corresponding to the location of all
            # the source elements
            raise NotImplementedError

            # check if the mask is binary or labelled
            "s_unique = unique(source.s_mask);"

            # create a second indexing variable
            if eng.eval("numel(s_unique) <= 2 && sum(s_unique) == 1"):
                s_mask = eng.getfield(source, "s_mask")
                s_mask_sum = np.array(s_mask).sum()

                # if more than one time series is given, check the number of time
                # series given matches the number of source elements
                if (
                    (self.source_sxx and (eng.eval("length(source.sxx(:,1)) > 1))")))
                    or (self.source_syy and (eng.eval("length(source.syy(:,1)) > 1))")))
                    or (self.source_szz and (eng.eval("length(source.szz(:,1)) > 1))")))
                    or (self.source_sxy and (eng.eval("length(source.sxy(:,1)) > 1))")))
                    or (self.source_sxz and (eng.eval("length(source.sxz(:,1)) > 1))")))
                    or (self.source_syz and (eng.eval("length(source.syz(:,1)) > 1))")))
                ):
                    if (
                        (self.source_sxx and (eng.eval("length(source.sxx(:,1))") != s_mask_sum))
                        or (self.source_syy and (eng.eval("length(source.syy(:,1))") != s_mask_sum))
                        or (self.source_szz and (eng.eval("length(source.szz(:,1))") != s_mask_sum))
                        or (self.source_sxy and (eng.eval("length(source.sxy(:,1))") != s_mask_sum))
                        or (self.source_sxz and (eng.eval("length(source.sxz(:,1))") != s_mask_sum))
                        or (self.source_syz and (eng.eval("length(source.syz(:,1))") != s_mask_sum))
                    ):
                        raise ValueError(
                            "The number of time series in source.sxx (etc) " "must match the number of source elements in source.s_mask."
                        )

            else:
                # check the source labels are monotonic, and start from 1
                # if (sum(s_unique(2:end) - s_unique(1:end-1)) != (numel(s_unique) - 1)) or (~any(s_unique == 1))
                if eng.eval("(sum(s_unique(2:end) - s_unique(1:end-1)) ~= " "(numel(s_unique) - 1)) || (~any(s_unique == 1))"):
                    raise ValueError(
                        "If using a labelled source.s_mask, " "the source labels must be monotonically increasing and start from 1."
                    )

                numel_s_unique = eng.eval("numel(s_unique) - 1;")
                # if more than one time series is given, check the number of time
                # series given matches the number of source elements
                if (
                    (self.source_sxx and (eng.eval("size(source.sxx, 1)") != numel_s_unique))
                    or (self.source_syy and (eng.eval("size(source.syy, 1)") != numel_s_unique))
                    or (self.source_szz and (eng.eval("size(source.szz, 1)") != numel_s_unique))
                    or (self.source_sxy and (eng.eval("size(source.sxy, 1)") != numel_s_unique))
                    or (self.source_sxz and (eng.eval("size(source.sxz, 1)") != numel_s_unique))
                    or (self.source_syz and (eng.eval("size(source.syz, 1)") != numel_s_unique))
                ):
                    raise ValueError(
                        "The number of time series in source.sxx (etc) "
                        "must match the number of labelled source elements in source.u_mask."
                    )

    @property
    def flag_ux(self):
        """
        Get the length of the sources in X-direction, this allows the
        inputs to be defined independently and be of any length

        Returns:
            Length of the sources
        """
        return len(self.ux[0]) if self.ux is not None else 0

    @property
    def flag_uy(self):
        """
        Get the length of the sources in X-direction, this allows the
        inputs to be defined independently and be of any length

        Returns:
            Length of the sources
        """
        return len(self.uy[0]) if self.uy is not None else 0

    @property
    def flag_uz(self):
        """
        Get the length of the sources in X-direction, this allows the
        inputs to be defined independently and be of any length

        Returns:
            Length of the sources
        """
        return len(self.uz[0]) if self.uz is not None else 0
