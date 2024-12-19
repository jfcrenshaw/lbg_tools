"""Tools for SED and flux calculations"""

import astropy.units as u
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from scipy.optimize import minimize_scalar

from .bandpass import Bandpass
from .igm import IGM


class SED:
    """LBG SED object"""

    def __init__(
        self,
        z: float,
        M: float,
        beta: float | None = None,
        igm_model: str | None = "inoue",
        wavelen_min: float = 100,
        wavelen_max: float = 12_000,
        N: int = 1_000,
    ) -> None:
        """Create SED

        Parameters
        ----------
        z : float
            Redshift
        M : float
            Absolute AB magnitude at 1500 angstroms
        beta : float or None, optional
            UV slope. If None, best fit model from Bouwens 2014 is used to
            determine beta as a function of M and z.
        igm_model : str or None, optional
            IGM model to use. Can be "inoue", "madau", or None.
            Default is "inoue".
        wavelen_min : float, optional
            Minimum rest-frame wavelength, in angstroms. Default is 100.
        wavelen_max : float, optional
            Maximum rest-frame wavelength, in angstroms. Default is 12,000.
        N : int, optional
            Number of wavelength bins. Default is 1000.
        """
        self.z = z
        self.M = M
        self._beta = beta
        self.igm_model = igm_model
        self.wavelen = np.linspace(wavelen_min, wavelen_max, N)

    @staticmethod
    def _beta_uv_model(M: float, z: float) -> float:
        """Calculate UV slope using model fit to Bouwens 2014 data.

        Parameters
        ----------
        M : float
            Absolute magnitude at 1500 angstroms.
        z : float
            Redshift.

        Returns
        -------
        float
            UV slope
        """
        return -0.167 * (M + 19.5) - 0.063 * z - 1.61

    @property
    def beta(self) -> float:
        """UV slope"""
        if self._beta is None:
            return self._beta_uv_model(self.M, self.z)
        else:
            return self._beta

    @beta.setter
    def beta(self, value: float | None) -> None:
        """Set the UV slope."""
        if value is None:
            self._beta = value
        else:
            self._beta = float(value)

    @property
    def truth(self) -> tuple[np.ndarray, np.ndarray]:
        """True SED."""
        A = 4.83e-8  # erg / s / cm^2 / Angstrom
        flambda = A * 10 ** (-0.4 * self.M) * (self.wavelen / 1_500) ** self.beta
        return self.wavelen.copy(), flambda

    @property
    def observed(self) -> tuple[np.ndarray, np.ndarray]:
        """Observed SED."""
        # Get true values
        wavelen, flambda = self.truth

        # Rescale flux for redshift and luminosity distance
        flambda /= (1 + self.z) * np.square(
            cosmo.luminosity_distance(self.z).to(u.pc).value / 10
        )

        # Redshift the wavelength grid
        wavelen *= 1 + self.z

        # Apply IGM transmission
        if self.igm_model is not None:
            igm = IGM(self.igm_model)
            flambda *= igm.transmission(wavelen, self.z)

        return wavelen, flambda

    def get_band_mag(self, bandpass: Bandpass) -> float:
        """Get observed magnitude in the given bandpass.

        Parameters
        ----------
        bandpass : Bandpass
            Bandpass object in which to calculate observed magnitude.

        Returns
        -------
        float
            Apparent AB magnitude observed in the bandpass
        """
        return bandpass.calc_magnitude(*self.observed)

    def set_band_mag(self, bandpass: Bandpass, m: float) -> None:
        """Set apparent observed magnitude in the bandpass."""
        # Solve for m as a function of the other variables
        res = minimize_scalar(
            lambda M: np.abs(
                SED(
                    z=self.z,
                    M=M,
                    beta=self._beta,
                    igm_model=self.igm_model,
                    wavelen_min=self.wavelen.min(),
                    wavelen_max=self.wavelen.max(),
                    N=self.wavelen.size,
                ).get_band_mag(bandpass)
                - m
            ),
            tol=1e-4,
        )

        # If it failed, raise an error
        # and make sure the user knows this was very unexpected...
        if not res.success:
            raise RuntimeError("Solving for m failed. That is really unexpected!")

        # If we succeeded, save the new absolute magnitude
        self.M = res.x
