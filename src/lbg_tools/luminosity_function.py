"""Class defining the LBG luminosity function"""

import copy
from typing import Callable, Self

import astropy.units as u
import numpy as np
from astropy.cosmology import Planck18 as cosmo

from .completeness import Completeness


class LuminosityFunction:
    """Double power law luminosity function.

    Evolves smoothly with redshift, using parameterization from
    Finkelstein & Bagley 2022, https://doi.org/10.3847/1538-4357/ac89eb
    """

    def __init__(
        self,
        phi0: float = -1.45,
        phi1: float = -0.31,
        mag0: float = -21.18,
        mag1: float = 0.02,
        alpha0: float = -1.27,
        alpha1: float = -0.11,
        beta0: float = -4.79,
        beta1: float = 0.05,
    ) -> None:
        """Create luminosity function.

        Parameters
        ----------
        phi0 : float, optional
            Constant term in phi* parameterization, by default -1.45
            Phi* sets the overall normalization.
        phi1 : float, optional
            Linear term in phi* parameterization, by default -0.31
            Phi* sets the overall normalization.
        mag0 : float, optional
            Constant term in M* parameterization, by default -21.18
            M* sets the turnover between the two power laws.
        mag1 : float, optional
            Linear term in M* parameterization, by default 0.02
            M* sets the turnover between the two power laws.
        alpha0 : float, optional
            Constant term in alpha parameterization, by default -1.27
            alpha is the faint-end slope.
        alpha1 : float, optional
            Linear term in alpha parameterization, by default -0.11
            alpha is the faint-end slope.
        beta0 : float, optional
            Constant term in beta parameterization, by default -4.79
            beta is the bright-end slope.
        beta1 : float, optional
            Linear term in beta parameterization, by default 0.05
            beta is the bright-end slope.
        """
        # Save params
        self.phi0 = phi0
        self.phi1 = phi1
        self.mag0 = mag0
        self.mag1 = mag1
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.beta0 = beta0
        self.beta1 = beta1

        # Completeness function is just identity
        self.completeness: Completeness | Callable = lambda m, z: 1

    def log_phi_star(self, z: float | np.ndarray) -> float | np.ndarray:
        """Calculate the log normalization.

        Parameters
        ----------
        z : float | np.ndarray
            The redshift

        Returns
        -------
        float | np.ndarray
            The log normalization
        """
        return self.phi0 + self.phi1 * (1 + z)

    def phi_star(self, z: float | np.ndarray) -> float | np.ndarray:
        """Calculate the normalization.

         Parameters
        ----------
        z : float | np.ndarray
            The redshift

        Returns
        -------
        float | np.ndarray
            The normalization
        """
        return np.log(10) / 2.5 * 10 ** self.log_phi_star(z)

    def M_star(self, z: float | np.ndarray) -> float | np.ndarray:
        """Calculate characteristic absolute magnitude.

        Parameters
        ----------
        z : float | np.ndarray
            The redshift

        Returns
        -------
        float | np.ndarray
            The characteristic absolute magnitude
        """
        return self.mag0 + self.mag1 * (1 + z)

    def alpha(self, z: float | np.ndarray) -> float | np.ndarray:
        """Calculate faint-end slope.

        Parameters
        ----------
        z : float | np.ndarray
            The redshift

        Returns
        -------
        float | np.ndarray
            The faint-end slope
        """
        return self.alpha0 + self.alpha1 * (1 + z)

    def beta(self, z: float | np.ndarray) -> float | np.ndarray:
        """Calculate bright-end slope.

        Parameters
        ----------
        z : float | np.ndarray
            The redshift

        Returns
        -------
        float | np.ndarray
            The bright-end slope
        """
        return self.beta0 + self.beta1 * (1 + z)

    def copy(self) -> Self:
        return copy.copy(self)

    def __mul__(self, other: Completeness) -> Self:
        """Define multiplication by completeness object."""
        if not isinstance(other, Completeness):
            raise TypeError("Multiplication is only defined with Completeness objects.")
        copy = self.copy()
        copy.completeness = other
        return copy

    def __rmul__(self, other: Completeness) -> Self:
        """Define multiplication by completeness object."""
        return self.__mul__(other)

    def true(
        self,
        M: float | np.ndarray,
        z: float | np.ndarray,
    ) -> float | np.ndarray:
        """Evaluate true luminosity function.

        Parameters
        ----------
        M : float or np.ndarray
            Absolute magnitude in detection band.
        z : float or np.ndarray
            Redshift

        Returns
        -------
        float or np.ndarray
            True number density in units of mag^-1 Mpc^-3
        """
        # Calculate raw LF
        dM = M - self.M_star(z)
        den1 = 10 ** (0.4 * (self.alpha(z) + 1) * dM)
        den2 = 10 ** (0.4 * (self.beta(z) + 1) * dM)
        lf0 = self.phi_star(z) / (den1 + den2)

        return np.atleast_1d(lf0).squeeze()

    def observed(
        self,
        M: float | np.ndarray,
        z: float | np.ndarray,
    ) -> float | np.ndarray:
        """Evaluate observed luminosity function.

        Parameters
        ----------
        M : float or np.ndarray
            Absolute magnitude in detection band.
        z : float or np.ndarray
            Redshift

        Returns
        -------
        float or np.ndarray
            True number density in units of mag^-1 Mpc^-3
        """
        # Convert absolute magnitude to apparent magnitude
        DL = cosmo.luminosity_distance(z).to(u.pc).value  # Lum. Dist. in pc
        m = M + 5 * np.log10(DL / 10) - 2.5 * np.log10(1 + z)

        # Calculate completeness multipliers
        completeness = self.completeness(m, z)

        return np.squeeze(completeness * self.true(M, z))

    def __call__(
        self,
        M: float | np.ndarray,
        z: float | np.ndarray,
    ) -> float | np.ndarray:
        """Evaluate observed luminosity function.

        Parameters
        ----------
        M : float or np.ndarray
            Absolute magnitude in detection band.
        z : float or np.ndarray
            Redshift

        Returns
        -------
        float or np.ndarray
            Number density in units of mag^-1 Mpc^-3
        """
        return self.observed(M, z)
