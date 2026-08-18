"""Test Tomographic Bin class"""

import numpy as np
import pytest

from lbg_tools import TomographicBin, library


def test_cant_set_properties() -> None:
    """Make sure we can't set properties after creation"""
    # Create tomographic bin object
    tbin = TomographicBin(library.get_bands()[0], 26)

    # Check that changing properties throws errors
    with pytest.raises(AttributeError):
        tbin.band = "fake"  # type: ignore
    with pytest.raises(AttributeError):
        tbin.mag_cut = -99  # type: ignore
    with pytest.raises(AttributeError):
        tbin.m5_det = -99  # type: ignore
    with pytest.raises(AttributeError):
        tbin.dz = -99  # type: ignore
    with pytest.raises(AttributeError):
        tbin.f_interlopers = -99  # type: ignore


def test_properties() -> None:
    """Test that bin properties run successfully"""
    tbin = TomographicBin(library.get_bands()[0], 26)
    tbin.nz
    tbin.number_density
    tbin.pz
    tbin.g_bias
    tbin.mag_bias


def test_min_redshift_zero() -> None:
    """Make sure negative redshifts aren't returned."""
    for band in library.get_bands():
        tbin = TomographicBin(band, 26, f_interlopers=0.2)
        z, _ = tbin.pz
        assert z.min() >= 0


def test_reasonable_mag_bias() -> None:
    """Test that the mag bias is reasonable."""
    u0 = TomographicBin("u", 24.5, 24.5, f_interlopers=0)
    assert np.isclose(u0.mag_bias, 2.7, atol=0.1)

    u0 = TomographicBin("u", 24.5, 24.5, f_interlopers=0.1)
    assert np.isclose(u0.mag_bias, 2.7, atol=0.1)


def test_dz() -> None:
    """Test that setting dz works."""
    u0 = TomographicBin("u", 24.5, 24.5, f_interlopers=0)
    u1 = TomographicBin("u", 24.5, 24.5, dz=0.3, f_interlopers=0)

    # Compare mean of redshift distributions
    z0, pz0 = u0.pz
    z1, pz1 = u1.pz
    mean0 = np.trapezoid(z0 * pz0, z0) / np.trapezoid(pz0, z0)
    mean1 = np.trapezoid(z1 * pz1, z1) / np.trapezoid(pz1, z1)
    assert mean1 > mean0


def test_stretch() -> None:
    """Test that stretching the p(z) works."""
    u0 = TomographicBin("u", 24.5, 24.5, f_interlopers=0)
    u1 = TomographicBin("u", 24.5, 24.5, stretch=2.0, f_interlopers=0)

    # Compare mean of redshift distributions
    z0, pz0 = u0.pz
    z1, pz1 = u1.pz
    mean0 = np.trapezoid(z0 * pz0, z0) / np.trapezoid(pz0, z0)
    mean1 = np.trapezoid(z1 * pz1, z1) / np.trapezoid(pz1, z1)
    assert mean1 > mean0

    # Check that number density is the same
    assert np.isclose(u1.number_density, u0.number_density)

    # Check that variance is greater
    var0 = np.trapezoid((z0 - mean0) ** 2 * pz0, z0) / np.trapezoid(pz0, z0)
    var1 = np.trapezoid((z1 - mean1) ** 2 * pz1, z1) / np.trapezoid(pz1, z1)
    assert var1 > var0


def test_interlopers_stay_continuous_across_zero_redshift() -> None:
    """Test that shifting interlopers below z=0 does not jump discontinuously."""
    # The u-dropout interloper grid starts just above zero, so a small negative
    # shift pushes its first point below zero. Cutting the distribution off at
    # zero has to leave it a continuous function of the shift; discarding the
    # sample point instead would remove a finite part of the integral at once.
    grid = np.linspace(0, 1, 2001)

    def mean_redshift(dz_interlopers: float) -> float:
        """Mean interloper redshift, measured on a fixed grid."""
        tbin = TomographicBin(
            "u", 24.5, 24.5, f_interlopers=0.2, dz_interlopers=dz_interlopers
        )
        z, nz = tbin.nz
        n_interlopers = tbin._z_interlopers.size
        resampled = np.interp(
            grid, z[:n_interlopers], nz[:n_interlopers], left=0, right=0
        )
        return np.trapezoid(grid * resampled, grid) / np.trapezoid(resampled, grid)

    # Take the same size of step across the crossing and well away from it. A
    # continuous distribution changes by a comparable amount either way; one that
    # loses a sample point at the crossing does not. Comparing the two needs no
    # tolerance to be chosen by hand.
    crossing = TomographicBin("u", 24.5, 24.5, f_interlopers=0.2)._z_interlopers[0]
    step = 1e-6
    across = abs(mean_redshift(-crossing + step) - mean_redshift(-crossing - step))
    away = abs(mean_redshift(0.01 + step) - mean_redshift(0.01 - step))
    assert across < 10 * max(away, 1e-12)

    # A shift larger than the grid spacing does legitimately drop points, but the
    # distribution stays physical and keeps moving in the same direction
    means = []
    for dz in (-0.10, -0.05, -0.01, 0.0, 0.01, 0.05, 0.10):
        tbin = TomographicBin("u", 24.5, 24.5, f_interlopers=0.2, dz_interlopers=dz)
        assert tbin._z_interlopers.min() >= 0
        means.append(mean_redshift(dz))
    assert np.all(np.diff(means) > 0)


def test_lbg_bias_matches_wilson_white() -> None:
    """Check lbg_bias against Wilson & White 2019 equation 2.7 evaluated by hand."""
    from lbg_tools import lbg_bias

    # m = 25 is where the fit's linear coefficients take their intercepts,
    # A = 0.11 and B = 0.17, so the relation is easy to check by hand there.
    assert np.isclose(lbg_bias(25.0, 3.0), 0.11 * 4 + 0.17 * 16)

    # Brighter samples sit in more massive halos, so they are more biased.
    assert lbg_bias(24.5, 3.0) > lbg_bias(25.0, 3.0) > lbg_bias(25.4, 3.0)

    # Bias rises with redshift at fixed depth across the dropout range.
    z = np.linspace(2, 5, 20)
    assert np.all(np.diff(lbg_bias(25.0, z)) > 0)


def test_lbg_bias_warns_outside_fit_range() -> None:
    """The relation is an interpolation; extrapolating it should say so."""
    from lbg_tools import lbg_bias

    with pytest.warns(UserWarning, match="outside 24"):
        lbg_bias(26.5, 3.0)

    # Faint limits drive the relation negative, which is unphysical rather than
    # merely uncertain, so that gets its own warning.
    with pytest.warns(UserWarning, match="non-positive"):
        lbg_bias(27.0, 1.0)


def test_public_population_split() -> None:
    """The two populations are exposed separately and concatenate back to nz."""
    tbin = TomographicBin("u", 24.5, 24.5, f_interlopers=0.2)

    z, nz = tbin.nz
    assert np.allclose(z, np.concatenate((tbin.z_interlopers, tbin.z_lbg)))
    assert np.allclose(nz, np.concatenate((tbin.nz_interlopers, tbin.nz_lbg)))

    # The populations are disjoint, with a gap between them.
    assert tbin.z_interlopers.max() < tbin.z_lbg.min()

    # With no interlopers there is nothing in the interloper half.
    tbin = TomographicBin("u", 24.5, 24.5, f_interlopers=0)
    assert np.allclose(tbin.nz_interlopers, 0)


def test_interloper_bias_is_configurable() -> None:
    """The interloper bias is an input, constant across their redshift range."""
    tbin = TomographicBin("u", 24.5, 24.5, f_interlopers=0.2)
    assert tbin.b_interlopers == 1.5

    tbin = TomographicBin("u", 24.5, 24.5, f_interlopers=0.2, b_interlopers=1.9)
    assert tbin.b_interlopers == 1.9

    _, b = tbin.g_bias
    n_interlopers = tbin.z_interlopers.size
    assert np.allclose(b[:n_interlopers], 1.9)

    # The Lyman-break half is unaffected and still rises with redshift.
    assert np.all(np.diff(b[n_interlopers:]) > 0)


def test_interlopers_shifted_entirely_below_zero() -> None:
    """A shift large enough to push every interloper below zero leaves none."""
    tbin = TomographicBin("u", 24.5, 24.5, f_interlopers=0.2, dz_interlopers=-2)

    assert tbin.z_interlopers.size == 0
    assert tbin.nz_interlopers.size == 0

    # What survives is the Lyman-break half alone, still a valid distribution.
    z, nz = tbin.nz
    assert np.allclose(z, tbin.z_lbg)
    assert np.all(np.isfinite(nz))
    assert z.min() >= 0

    # The bias still evaluates on whatever grid is left.
    _, b = tbin.g_bias
    assert b.size == z.size
