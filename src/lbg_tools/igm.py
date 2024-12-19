"""Class for calculation of IGM optical depth and transmission."""

import numpy as np

from ._igm_inoue import igm_tau as inoue_tau
from ._igm_madau import igm_tau as madau_tau


class IGM:
    """Class for calculation of IGM optical depth and transmission.

    Implements models of
    - Madau 1995: https://ui.adsabs.harvard.edu/abs/1995ApJ...441...18M/abstract
    - Inoue 2014: https://ui.adsabs.harvard.edu/abs/2014MNRAS.442.1805I/abstract
    """

    def __init__(self, model: str = "inoue") -> None:
        """Create IGM model.

        Parameters
        ----------
        model : str, optional
            Name of the IGM model to use. Either "inoue" or "madau".
            Default is "inoue".
        """
        if model == "inoue":
            self._tau = inoue_tau
        elif model == "madau":
            self._tau = madau_tau
        else:
            raise ValueError(f"IGM model {model} not implemented.")

    def tau(self, wavelen: float | np.ndarray, z: float) -> np.ndarray:
        """Calculate optical depth of the IGM.

        Parameters
        ----------
        wavelen : float | np.ndarray
            Observed wavelength in Angstroms
        z : float
            Redshift

        Returns
        -------
        np.ndarray
        IGM optical depth
        """
        # First we will calculate Tau on a fine grid, which will allow us to
        # then interpolate linearly. This avoids bugs with the hacky patch
        # that prevents the continuum optical depth from decreasing at
        # very small wavelengths.
        wave_min = 0
        wave_max = max(10_000, np.max(wavelen))
        wave_grid = np.linspace(wave_min, wave_max, 10_000)
        tau_grid = self._tau(wavelen=wave_grid, z=z)

        # Now return interpolation
        return np.interp(wavelen, wave_grid, tau_grid)

    def transmission(self, wavelen: float | np.ndarray, z: float) -> np.ndarray:
        """Calculate transmission of the IGM.

        Parameters
        ----------
        wavelen : float | np.ndarray
            Observed wavelength in Angstroms
        z : float
            Redshift

        Returns
        -------
        np.ndarray
            IGM transmission
        """
        return np.exp(-self.tau(wavelen, z))
