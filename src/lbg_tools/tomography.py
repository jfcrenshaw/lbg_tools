"""Class to define tomographic bin."""

import astropy.units as u
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from scipy.integrate import simpson

from .completeness import Completeness
from .luminosity_function import LuminosityFunction


class TomographicBin:
    def __init__(
        self,
        band: str,
        mag_cut: float,
        m5_det: float | None = None,
        lf_params: dict | None = None,
    ) -> None:
        """Create tomographic bin.

        Parameters
        ----------
        band : str
            Name of dropout band
        mag_cut : float
            Magnitude cut in the detection band
        m5 : float or None
            5-sigma depth in the detection band. If None, mag_cut is used.
            The default is None.
        lf_params : dict or None
            Parameters to pass to luminosity function creation.
        """
        # Set m5_det
        m5_det = mag_cut if m5_det is None else m5_det

        # Save params
        self._band = band
        self._mag_cut = mag_cut
        self._m5_det = m5_det

        # Create luminosity function for tomographic bin
        lf_params = {} if lf_params is None else lf_params
        self._lf_params = lf_params
        lf = LuminosityFunction(**lf_params)
        self.completeness = Completeness(band, m5_det)
        self.luminosity_function = lf * self.completeness

    @property
    def band(self) -> str:
        """Name of dropout band"""
        return self._band

    @property
    def mag_cut(self) -> float:
        """Magnitude cut in the detection band"""
        return self._mag_cut

    @property
    def m5_det(self) -> float:
        """5-sigma depth in the detection band"""
        return self._m5_det

    @property
    def nz(self) -> tuple[np.ndarray, np.ndarray]:
        """Projected number density per redshift

        Returns
        -------
        np.ndarray
            Redshift grid
        np.ndarray
            Number density of galaxies in each bin
        """
        # Create grid over apparent magnitude
        m = np.linspace(20, self.mag_cut, 101)

        # Get redshift grid from completeness table
        z = self.completeness.table.index[..., None]

        # Convert apparent to absolute magnitude
        DL = cosmo.luminosity_distance(z).to(u.pc).value  # Lum. Dist. in pc
        M = m - 5 * np.log10(DL / 10) + 2.5 * np.log10(1 + z)

        # Calculate luminosity * completeness
        lfc = self.luminosity_function(M, z)

        # Calculate dV/dz (Mpc^-3 deg^-2)
        A_sky = 41_253  # deg^2
        deg2_per_ster = A_sky / (4 * 3.14159)
        dVdz = cosmo.differential_comoving_volume(z).value / deg2_per_ster

        # Integrate luminosity function to get number density of galaxies
        # in each redshift bin
        nz = simpson(lfc * dVdz, x=M, axis=-1)

        return z.squeeze(), nz

    @property
    def number_density(self) -> float:
        """Number density in deg^2

        Returns
        -------
        float
            Total projected number density of LBGs in units deg^-2
        """
        # Number density in each redshift bin
        z, nz = self.nz

        # Integrate over redshift bins
        n = simpson(nz, x=z)

        return n

    @property
    def pz(self) -> tuple[np.ndarray, np.ndarray]:
        """Redshift distribution.

        Returns
        -------
        np.ndarray
            Redshift grid
        np.ndarray
            Normalized redshift distribution
        """
        # Number density in each redshift bin
        z, nz = self.nz

        # Integrate over redshift bins
        n = simpson(nz, x=z)

        # Normalize redshift distribution
        pz = nz / n[:, None]

        return z, pz.squeeze()

    @property
    def mag_bias(self) -> float:
        """Magnification bias alpha coefficient.

        Defined as 2.5 * d/dm(log number_density) at mag_cut
        """
        # Create deeper bin
        # This is equivalent to making all the galaxies brighter by dm
        dm = 0.01
        magnified_bin = TomographicBin(
            band=self.band,
            mag_cut=self.mag_cut + dm,
            m5_det=self.m5_det + dm,
            lf_params=self._lf_params,
        )

        # Calculate log10 number density for original and deeper bin
        n0 = np.log10(self.number_density)
        n1 = np.log10(magnified_bin.number_density)

        # Calculate alpha
        alpha = 2.5 * (n1 - n0) / dm

        return alpha
