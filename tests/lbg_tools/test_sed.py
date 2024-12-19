"""Test the SED class"""

import numpy as np

from lbg_tools import SED, Bandpass, library


def test_beta() -> None:
    """Test beta values match expectations."""
    # auto-calculation
    sed = SED(0, -19.5)
    assert np.isclose(sed.beta, -1.61)

    # override
    sed = SED(0, -19.5, beta=-4.9)
    assert np.isclose(sed.beta, -4.9)

    # override ex post facto
    sed.beta = 2
    assert np.isclose(sed.beta, 2)

    # back to auto-calculation
    sed.beta = None  # type: ignore
    assert np.isclose(sed.beta, -1.61)


def test_observed_is_fainter() -> None:
    """Test that the observed SED is fainter than the truth."""
    sed = SED(4, -22)

    # Grab true and observed values
    wt, ft = sed.truth
    wo, fo = sed.observed

    # Resample truth onto observed
    ft = np.interp(wo, wt, ft, left=np.nan)
    ft = ft[np.isfinite(ft)]

    # Check observed is fainter
    assert np.all(ft > fo[: ft.size])


def test_band_mag_reasonable() -> None:
    """Check that band mag is reasonable.

    We will do this by looking at z~0 band mag and comparing to M
    """
    # Closest to 1500 angstroms at z~0
    bandpass = Bandpass("u")

    # Redshift ~ 10 pc
    z = 2.257e-9

    # Create spectrum flat in flambda
    sed = SED(z, -22, beta=-2)

    # Make sure band mag is very close
    m = sed.get_band_mag(bandpass)
    assert np.isclose(m, -22, atol=1e-2)


def test_round_trip() -> None:
    """Check that set_band_mag returns input absolute mag."""
    # Loop over a bunch of different settings
    bands = library.get_bands(completeness=False, bandpass=True)
    zs = [3.1, 4.4, 5.6, 6.2, 5.1, 2.8]
    Ms = [-23.2, -19.2, -24.0, -20.7, -21.1, -25.0]
    igm_models = 2 * ["inoue", "madau", None]
    for band, z, M, igm_model in zip(bands, zs, Ms, igm_models):
        # Create the bandpass
        bandpass = Bandpass(band)

        # Create an SED
        sed = SED(z, M, igm_model=igm_model)

        # Calculate the bandpass mag
        m = sed.get_band_mag(bandpass)

        # Set absolute mag to match observed mag
        sed.set_band_mag(bandpass, m)

        # Make sure result matches input
        assert np.isclose(sed.M, M, atol=1e-2)
