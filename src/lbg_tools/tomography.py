"""Class to define tomographic bin."""

import numpy as np
from astropy.cosmology import Cosmology, Planck18
from scipy.integrate import simpson

from .completeness import Completeness
from .cosmo_utils import check_cosmology, diff_comoving_volume, luminosity_distance
from .luminosity_function import LuminosityFunction

# Protected import for optional dependency
try:
    import pyccl as ccl
except ImportError:  # pragma: no cover
    ccl = None


class TomographicBin:
    """Tomographic sample of LBGs."""

    def __init__(
        self,
        band: str,
        mag_cut: float,
        m5_det: float | None = None,
        dz: float = 0.0,
        stretch: float = 1.0,
        f_interlopers: float = 0.0,
        dz_interlopers: float = 0.0,
        stretch_interlopers: float = 1.0,
        lf_params: dict | None = None,
        completeness_params: dict | None = None,
        cosmology: "Cosmology | ccl.Cosmology" = Planck18,
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
        stretch : float, optional
            Stretch factor for the width of the true LBG redshift distribution.
            (the default is 1.0)
        f_interlopers : float, optional
            Fraction of low-redshift interlopers. Same p(z) shape is used
            for interlopers, but shifted to the redshift corresponding to
            Lyman-/Balmer-break confusion. (the default is zero)
        dz_interlopers : float, optional
            Amount by which to shift the distribution of interlopers (i.e.
            true LBGs are not shifted). This corresponds to the DES delta z
            nuisance parameters. (the default is zero)
        stretch_interlopers : float, optional
            Stretch factor for the width of the interloper redshift distribution.
            (the default is 1.0)
        lf_params : dict or None, optional
            Parameters to pass to luminosity function creation.
            The default is None (i.e. default Luminosity Function used).
            Note if this dictionary contains a cosmology, it will be overridden
            by the cosmology parameter below.
        completeness_params : dict or None, optional
            Additional parameters to pass to the Completeness constructor.
            Default is None.
        cosmology : Cosmology or pyccl.Cosmology, optional
            Astropy or pyccl Cosmology object to use. Default is astropy's Planck18.
            Note if you want to use pyccl, you must install it yourself.
        """
        # Set m5_det
        m5_det = mag_cut if m5_det is None else m5_det

        # Save params
        self._band = band
        self._mag_cut = mag_cut
        self._m5_det = m5_det
        self._dz = dz
        self._stretch = stretch
        self._f_interlopers = f_interlopers
        self._dz_interlopers = dz_interlopers
        self._stretch_interlopers = stretch_interlopers

        # Check and save cosmology
        check_cosmology(cosmology)
        self.cosmology = cosmology

        # Create luminosity function for tomographic bin
        lf_params = {} if lf_params is None else lf_params
        lf_params["cosmology"] = self.cosmology  # Override cosmology
        self._lf_params = lf_params
        lf = LuminosityFunction(**lf_params)

        # Create completeness function for tomographic bin
        completeness_params = {} if completeness_params is None else completeness_params
        self._completeness_params = completeness_params
        self.completeness = Completeness(band, m5_det, **completeness_params)
        self.luminosity_function = lf * self.completeness

        # Calculate n(z)
        self._calc_nz()

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
    def stretch(self) -> float:
        """Stretch factor for true LBG redshift distribution"""
        return self._stretch

    @property
    def f_interlopers(self) -> float:
        """Interloper fraction"""
        return self._f_interlopers

    @property
    def dz_interlopers(self) -> float:
        """Shift in interloper redshift distribution"""
        return self._dz_interlopers

    @property
    def stretch_interlopers(self) -> float:
        """Stretch factor for interloper redshift distribution"""
        return self._stretch_interlopers

    def _calc_nz(self) -> None:
        """Perform n(z) calculation to set everything up.

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

        # Create grid over apparent magnitude
        m = np.linspace(20, self.mag_cut, 101)

        # Expand dimension on LBG redshifts for calculations below
        z_lbg = z_lbg[..., None]

        # Convert apparent to absolute magnitude
        dL = luminosity_distance(self.cosmology, z_lbg)
        M = m - 5 * np.log10(dL / 10) + 2.5 * np.log10(1 + z_lbg)

        # Calculate luminosity * completeness
        lfc = self.luminosity_function(M, z_lbg)

        # Calculate dV/dz (Mpc^3 deg^-2)
        dVdz = diff_comoving_volume(self.cosmology, z_lbg)

        # Integrate luminosity function to get number density of galaxies
        # in each redshift bin
        nz_lbg = simpson(lfc * dVdz, x=M, axis=-1)

        # Re-collapse redshift grid
        z_lbg = z_lbg.squeeze()

        # Generate interloper distribution
        nz_interlopers = nz_lbg[-z_interlopers.size :].copy()
        nz_interlopers /= simpson(nz_interlopers, x=z_interlopers)
        N_lbg = simpson(nz_lbg, x=z_lbg)
        N_interlopers = N_lbg * self.f_interlopers / (1 - self.f_interlopers)
        nz_interlopers *= N_interlopers

        # Shift distributions
        z_lbg += self.dz
        z_interlopers += self.dz_interlopers

        # Stretch distributions
        z_lbg = self.stretch * (z_lbg - z_lbg.mean()) + z_lbg.mean()
        z_interlopers = (
            self.stretch_interlopers * (z_interlopers - z_interlopers.mean())
            + z_interlopers.mean()
        )

        # Remove negative redshifts
        mask = z_interlopers > 0
        z_interlopers = z_interlopers[mask]
        nz_interlopers = nz_interlopers[mask]

        # Re-normalize distributions
        nz_lbg *= N_lbg / simpson(nz_lbg, x=z_lbg)
        if self.f_interlopers > 0:
            nz_interlopers *= N_interlopers / simpson(nz_interlopers, x=z_interlopers)

        # Combine true and interloper distributions
        z = np.concatenate((z_interlopers, z_lbg))
        nz = np.concatenate((nz_interlopers, nz_lbg))

        # Save values to be reused
        self._z_interlopers = z_interlopers
        self._z_lbg = z_lbg.squeeze()
        self._nz_interlopers = nz_interlopers
        self._nz_lbg = nz_lbg
        self._z = z
        self._nz = nz
        self._density = N_lbg + N_interlopers

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
        return self._z, self._nz

    @property
    def number_density(self) -> float:
        """Number density in deg^2

        Returns
        -------
        float
            Total projected number density of LBGs in units deg^-2
        """
        return self._density

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
        z, nz = self.nz
        n = np.atleast_1d(self.number_density)
        pz = nz / n[:, None]

        return z, pz.squeeze()

    @property
    def g_bias(self) -> tuple[np.ndarray, np.ndarray]:
        """Linear galaxy bias"""
        # Get redshift distributions
        z_interlopers, z_lbg = self._z_interlopers, self._z_lbg

        # Calculate the galaxy bias
        # TODO: better interloper bias
        b_interlopers = 0.28 * (1 + z_interlopers) ** 1.6
        b_lbg = 0.28 * (1 + z_lbg) ** 1.6

        z = np.concatenate((z_interlopers, z_lbg))
        b = np.concatenate((b_interlopers, b_lbg))

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
            stretch=self.stretch,
            f_interlopers=self.f_interlopers,
            dz_interlopers=self.dz_interlopers,
            stretch_interlopers=self.stretch_interlopers,
            lf_params=self._lf_params,
            completeness_params=self._completeness_params,
            cosmology=self.cosmology,
        )

        # Calculate log10 number density for original and deeper bin
        n0 = np.log10(self.number_density)
        n1 = np.log10(magnified_bin.number_density)

        # Calculate alpha
        alpha = 2.5 * (n1 - n0) / dm

        return alpha
