"""Class to define tomographic bin."""

import warnings

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


def _truncate_at_zero(
    z: np.ndarray,
    nz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict a redshift distribution to positive redshifts.

    Galaxies cannot sit at negative redshift, so a distribution that has been
    shifted or stretched below zero has to be cut off there. Discarding the
    sample points that have gone negative would make the distribution a
    discontinuous function of the shift, because each point carries a finite
    part of the integral: crossing zero would remove it all at once. Instead the
    distribution is cut where it belongs, at zero, with the value there
    interpolated from its neighbours.

    Parameters
    ----------
    z : np.ndarray
        Redshift grid, increasing.
    nz : np.ndarray
        Number density per unit redshift on that grid.

    Returns
    -------
    np.ndarray
        The grid, starting at zero if it previously started below it.
    np.ndarray
        The number density on that grid.
    """
    if z.size == 0 or z[0] >= 0:
        return z, nz

    positive = z > 0
    if not positive.any():
        return z[:0], nz[:0]

    return (
        np.concatenate(([0.0], z[positive])),
        np.concatenate(([np.interp(0.0, z, nz)], nz[positive])),
    )


def _stretch_about_mean(
    z: np.ndarray,
    nz: np.ndarray,
    stretch: float,
) -> np.ndarray:
    """Scale the width of a distribution without moving its mean.

    The pivot is the number-weighted mean of the distribution, not the midpoint
    of the grid it happens to be sampled on. Pivoting on the grid instead
    couples the width to the mean: for an LSST-like u-dropout sample the two
    differ by 0.047 in redshift, so a ten per cent stretch also shifted the mean
    by 0.005 -- larger than the shift uncertainty such a sample is usually
    assigned, which makes the width and shift parameters degenerate for a reason
    that is arithmetic rather than physical.

    Parameters
    ----------
    z : np.ndarray
        Redshift grid, increasing.
    nz : np.ndarray
        Number density per unit redshift on that grid, used as the weight.
    stretch : float
        Factor by which to scale the width.

    Returns
    -------
    np.ndarray
        The stretched grid. The distribution keeps its mean and its integral,
        and its standard deviation is scaled by exactly ``stretch``.
    """
    # An unpopulated grid has no weighted mean to pivot on, which is the case
    # for the interlopers when there are none; the midpoint will do there, since
    # nothing is weighted by it either way.
    weight = simpson(nz, x=z)
    pivot = simpson(z * nz, x=z) / weight if weight > 0 else z.mean()

    return stretch * (z - pivot) + pivot


def lbg_bias(mag_cut: float, z: float | np.ndarray) -> np.ndarray:
    """Linear bias of a Lyman-break sample limited at an apparent magnitude.

    Equation 2.7 of Wilson & White 2019 (arXiv:1904.13378), a low-order
    polynomial in (1 + z) fit to a compilation of Lyman-break galaxy bias
    measurements. The first term is stable clustering, b D+ = constant; the
    second captures the bias rising more steeply at high redshift, which the
    authors attribute to the apparent magnitude limit. The same relation is used
    by Sailer et al. 2021 (arXiv:2106.09713) and Ebina & White 2024
    (arXiv:2401.13166, their equation 2.9).

    Parameters
    ----------
    mag_cut : float
        Apparent magnitude limit of the sample, in the detection band.
    z : float or np.ndarray
        Redshift(s) at which to evaluate the bias.

    Returns
    -------
    np.ndarray
        Linear galaxy bias.

    Warns
    -----
    UserWarning
        If ``mag_cut`` falls outside 24 < m < 25.5, the range of depths Wilson &
        White show, or if the relation returns a non-positive bias. It is an
        interpolation of a data compilation with no physical floor, so it goes
        negative for faint limits at low redshift: at m = 26 it crosses zero at
        z = 2.3.
    """
    if not 24.0 <= mag_cut <= 25.5:
        warnings.warn(
            f"mag_cut={mag_cut} is outside 24 < m < 25.5, the range of depths "
            "Wilson & White 2019 fit their bias relation over. The relation is "
            "an interpolation, so treat the result as an extrapolation.",
            stacklevel=2,
        )

    A = -0.98 * (mag_cut - 25) + 0.11
    B = 0.12 * (mag_cut - 25) + 0.17
    b = A * (1 + np.asarray(z, dtype=float)) + B * (1 + np.asarray(z, dtype=float)) ** 2

    if np.any(b <= 0):
        warnings.warn(
            f"The Wilson & White 2019 bias relation is non-positive somewhere on "
            f"this redshift grid for mag_cut={mag_cut}. It is a fit to a data "
            "compilation rather than a physical model, and it has no positive "
            "floor. Check that the redshift range is the 2 < z < 5 the relation "
            "was calibrated for.",
            stacklevel=2,
        )

    return b


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
        b_interlopers: float = 1.5,
        dm_mag_bias: float = 0.01,
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
        b_interlopers : float, optional
            Linear galaxy bias of the interlopers, taken as constant across their
            redshift range. It is a free input rather than a formula because the
            interloper population is a mix of galaxy types that the Lyman-break
            bias relation does not describe. The default, 1.5, is typical of the
            low-redshift lens samples interlopers resemble.
        dm_mag_bias : float, optional
            Magnitude step used to differentiate the number counts when
            evaluating :attr:`mag_bias`. The default is 0.01.
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

        # A bin describes one selection. The redshift grid comes from the
        # completeness table and the magnitude grid from the cut, so an array of
        # cuts or depths would have to carry an extra axis through the
        # luminosity function, the completeness and the integrals below. Rather
        # than half-support that, say so: building one bin per depth is cheap,
        # because the expensive part is the completeness table and that is
        # shared between bins of the same band.
        for name, arg in (("mag_cut", mag_cut), ("m5_det", m5_det)):
            if np.ndim(arg) != 0:
                raise TypeError(
                    f"{name} must be a scalar, not an array. To scan a grid, "
                    "build one TomographicBin per grid point: the completeness "
                    "table is loaded once per band and shared."
                )

        # Save params
        self._band = band
        self._mag_cut = mag_cut
        self._m5_det = m5_det
        self._dz = dz
        self._stretch = stretch
        self._f_interlopers = f_interlopers
        self._dz_interlopers = dz_interlopers
        self._stretch_interlopers = stretch_interlopers
        self._b_interlopers = b_interlopers
        self._dm_mag_bias = dm_mag_bias

        # Filled in on first access; see the mag_bias property
        self._mag_bias: float | None = None

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

    @property
    def b_interlopers(self) -> float:
        """Linear galaxy bias of the interlopers"""
        return self._b_interlopers

    @property
    def dm_mag_bias(self) -> float:
        """Magnitude step used to differentiate the counts for mag_bias"""
        return self._dm_mag_bias

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
        # np.array rather than to_numpy, because pandas >= 3 returns a
        # read-only view of the index, and the grid is shifted in place below
        z_lbg = np.array(self.completeness.table.index)

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

        # Stretch distributions about their own means, so that stretch changes
        # the width and dz changes the mean, independently
        z_lbg = _stretch_about_mean(z_lbg, nz_lbg, self.stretch)
        z_interlopers = _stretch_about_mean(
            z_interlopers, nz_interlopers, self.stretch_interlopers
        )

        # Cut the interloper distribution off at zero redshift
        z_interlopers, nz_interlopers = _truncate_at_zero(z_interlopers, nz_interlopers)

        # Re-normalize distributions. The number of galaxies that pass the
        # selection is an observable: dz and stretch express uncertainty in
        # where those galaxies sit in redshift, not in how many there are. So
        # truncating at zero redistributes the interlopers rather than
        # discarding them, and the totals below are the pre-truncation ones.
        nz_lbg *= N_lbg / simpson(nz_lbg, x=z_lbg)
        if self.f_interlopers > 0:
            if z_interlopers.size < 2:
                raise ValueError(
                    f"dz_interlopers={self.dz_interlopers} and "
                    f"stretch_interlopers={self.stretch_interlopers} push the "
                    "whole interloper distribution below zero redshift, leaving "
                    "no support to hold the interlopers the selection admits. "
                    "The model cannot represent that; use a smaller shift."
                )
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
    def z_lbg(self) -> np.ndarray:
        """Redshift grid of the true Lyman-break galaxies

        The grid returned by :attr:`nz` is this grid concatenated onto
        :attr:`z_interlopers`, with a gap between the two populations.

        Returns
        -------
        np.ndarray
            Redshift grid of the Lyman-break half of the sample
        """
        return self._z_lbg

    @property
    def nz_lbg(self) -> np.ndarray:
        """Projected number density per redshift of the true Lyman-break galaxies

        Returns
        -------
        np.ndarray
            Number density on :attr:`z_lbg`, in deg^-2 per unit redshift
        """
        return self._nz_lbg

    @property
    def z_interlopers(self) -> np.ndarray:
        """Redshift grid of the low-redshift interlopers

        Returns
        -------
        np.ndarray
            Redshift grid of the interloper half of the sample
        """
        return self._z_interlopers

    @property
    def nz_interlopers(self) -> np.ndarray:
        """Projected number density per redshift of the low-redshift interlopers

        Returns
        -------
        np.ndarray
            Number density on :attr:`z_interlopers`, in deg^-2 per unit redshift
        """
        return self._nz_interlopers

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

        return z, nz / self.number_density

    @property
    def g_bias(self) -> tuple[np.ndarray, np.ndarray]:
        """Linear galaxy bias

        The Lyman-break half uses :func:`lbg_bias`, which depends on both the
        apparent magnitude limit and redshift. The interloper half is the constant
        :attr:`b_interlopers`, because the Lyman-break relation is calibrated for
        2 < z < 5 and returns negative bias at interloper redshifts.

        Returns
        -------
        np.ndarray
            Redshift grid
        np.ndarray
            Linear galaxy bias on that grid
        """
        # Get redshift distributions
        z_interlopers, z_lbg = self.z_interlopers, self.z_lbg

        # Calculate the galaxy bias
        b_interlopers = np.full_like(z_interlopers, self.b_interlopers)
        b_lbg = lbg_bias(self.mag_cut, z_lbg)

        z = np.concatenate((z_interlopers, z_lbg))
        b = np.concatenate((b_interlopers, b_lbg))

        return z, b

    def _shifted(self, dm: float) -> "TomographicBin":
        """Copy this bin with every galaxy brightened by ``dm``.

        Brightening the galaxies moves the cut and the depth together: the cut
        is fixed in apparent magnitude, so relative to brighter galaxies it sits
        ``dm`` fainter, and their signal-to-noise improves by the same amount.

        Parameters
        ----------
        dm : float
            Amount by which to brighten the galaxies. Negative dims them.

        Returns
        -------
        TomographicBin
            An otherwise identical bin.
        """
        return TomographicBin(
            band=self.band,
            mag_cut=self.mag_cut + dm,
            m5_det=self.m5_det + dm,
            dz=self.dz,
            stretch=self.stretch,
            f_interlopers=self.f_interlopers,
            dz_interlopers=self.dz_interlopers,
            stretch_interlopers=self.stretch_interlopers,
            b_interlopers=self.b_interlopers,
            dm_mag_bias=self.dm_mag_bias,
            lf_params=self._lf_params,
            completeness_params=self._completeness_params,
            cosmology=self.cosmology,
        )

    @property
    def mag_bias(self) -> float:
        """Magnification bias alpha coefficient

        Defined as 2.5 * d/dm(log number_density) at mag_cut, evaluated with a
        central difference over +/- :attr:`dm_mag_bias`. Central rather than
        one-sided because the one-sided version returns the slope half a step
        away from the cut rather than at it, and the extra accuracy is free
        after the first access: each side costs a second bin, so the result is
        computed once and kept.

        Returns
        -------
        float
            The magnification bias coefficient alpha.
        """
        if self._mag_bias is not None:
            return self._mag_bias

        dm = self.dm_mag_bias

        # Number counts with the galaxies brightened and dimmed by dm
        n_up = np.log10(self._shifted(dm).number_density)
        n_down = np.log10(self._shifted(-dm).number_density)

        # Calculate alpha
        self._mag_bias = float(2.5 * (n_up - n_down) / (2 * dm))

        return self._mag_bias
