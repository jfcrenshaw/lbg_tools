"""Class to define tomographic bin."""

import astropy.units as u
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from scipy.integrate import simpson

from .completeness import Completeness
from .luminosity_function import LuminosityFunction


class TomographicBin:
    """Tomographic sample of LBGs."""

    def __init__(
        self,
        band: str,
        mag_cut: float,
        m5_det: float | None = None,
        dz: float = 0,
        f_interlopers: float = 0,
        lf_params: dict | None = None,
    ) -> None:
        """Create tomographic bin.

        Parameters
        ----------
        band : str
            Name of dropout band
        mag_cut : float
            Magnitude cut in the detection band
        m5_dat : float or None, optional
            5-sigma depth in the detection band. If None, mag_cut is used.
            The default is None.
        dz : float, optional
            Amount by which to shift the distribution of true LBGs (i.e.
            interlopers are not shifted). This corresponds to the DES delta z
            nuisance parameters. (the default is zero)
        f_interlopers : float, optional
            Fraction of low-redshift interlopers. Same p(z) shape is used
            for interlopers, but shifted to the redshift corresponding to
            Lyman-/Balmer-break confusion. (the default is zero)
        lf_params : dict or None, optional
            Parameters to pass to luminosity function creation.
            The default is None (i.e. default Luminosity Function used)
        """
        # Set m5_det
        m5_det = mag_cut if m5_det is None else m5_det

        # Save params
        self._band = band
        self._mag_cut = mag_cut
        self._m5_det = m5_det
        self._dz = dz
        self._f_interlopers = f_interlopers

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
    def dz(self) -> float:
        """Shift in true LBG redshift distribution"""
        return self._dz

    @property
    def f_interlopers(self) -> float:
        """Interloper fraction"""
        return self._f_interlopers

    def _get_z_grids(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the two redshift grids.

        Returns
        -------
        np.ndarray
            Interloper redshift grid
        np.ndarray
            True LBG redshift grid
        """
        # Get grid from completeness table
        z_lbg = self.completeness.table.index.to_numpy()

        # Create interloper grid
        lambda_L = 1216  # angstroms
        lambda_B = 4000  # angstroms
        z_interlopers = lambda_L / lambda_B * (1 + z_lbg) - 1

        # Cut off negative values
        mask = z_interlopers > 0
        z_interlopers = z_interlopers[mask]

        return z_interlopers, z_lbg

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

        # Get redshift grids
        z_interlopers, z_lbg = self._get_z_grids()

        # Expand dimension on LBG redshifts for calculations below
        z_lbg = z_lbg[..., None]

        # Convert apparent to absolute magnitude
        DL = cosmo.luminosity_distance(z_lbg).to(u.pc).value  # Lum. Dist. in pc
        M = m - 5 * np.log10(DL / 10) + 2.5 * np.log10(1 + z_lbg)

        # Calculate luminosity * completeness
        lfc = self.luminosity_function(M, z_lbg)

        # Calculate dV/dz (Mpc^-3 deg^-2)
        A_sky = 41_253  # deg^2
        deg2_per_ster = A_sky / (4 * 3.14159)
        dVdz = cosmo.differential_comoving_volume(z_lbg).value / deg2_per_ster

        # Integrate luminosity function to get number density of galaxies
        # in each redshift bin
        nz_lbg = simpson(lfc * dVdz, x=M, axis=-1)

        # Re-collapse redshift grid
        z_lbg = z_lbg.squeeze()

        # Generate interloper distribution
        if self.f_interlopers > 0:
            # Rescale to appropriate interloper fraction
            nz_interlopers = nz_lbg[-z_interlopers.size :].copy()
            nz_interlopers /= simpson(nz_interlopers, x=z_interlopers)
            N_lbg = simpson(nz_lbg, x=z_lbg)
            N_interlopers = N_lbg * self.f_interlopers / (1 - self.f_interlopers)
            nz_interlopers *= N_interlopers
        else:
            z_interlopers = np.array([])
            nz_interlopers = np.array([])

        # Shift the true LBG distribution
        z_lbg += self.dz

        # Combine true and interloper distributions
        z = np.concatenate((z_interlopers, z_lbg))
        nz = np.concatenate((nz_interlopers, nz_lbg))

        return z, nz

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
        n = simpson(nz, x=z, axis=-1)

        return n

    @property
    def pz(self) -> tuple[np.ndarray, np.ndarray]:
        """Redshift distribution

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
        n = np.atleast_1d(simpson(nz, x=z, axis=-1))

        # Normalize redshift distribution
        pz = nz / n[:, None]

        return z, pz.squeeze()

    @property
    def g_bias(self) -> tuple[np.ndarray, np.ndarray]:
        """Linear galaxy bias"""
        # Get redshift distributions
        z_interlopers, z_lbg = self._get_z_grids()

        # Calculate the galaxy bias
        # TODO: better interloper bias
        b_interlopers = 0.28 * (1 + z_interlopers) ** 1.6
        b_lbg = 0.28 * (1 + z_lbg) ** 1.6

        if self.f_interlopers > 0:
            z = np.concatenate((z_interlopers, z_lbg))
            b = np.concatenate((b_interlopers, b_lbg))
        else:
            z = z_lbg
            b = b_lbg

        return z, b

    @property
    def mag_bias(self) -> float:
        """Magnification bias alpha coefficient

        Defined as 2.5 * d/dm(log number_density) at mag_cut
        """
        # Create deeper bin
        # This is equivalent to making all the galaxies brighter by dm
        dm = 0.01
        magnified_bin = TomographicBin(
            band=self.band,
            mag_cut=self.mag_cut + dm,
            m5_det=self.m5_det + dm,
            dz=self.dz,
            f_interlopers=self.f_interlopers,
            lf_params=self._lf_params,
        )

        # Calculate log10 number density for original and deeper bin
        n0 = np.log10(self.number_density)
        n1 = np.log10(magnified_bin.number_density)

        # Calculate alpha
        alpha = 2.5 * (n1 - n0) / dm

        return alpha
