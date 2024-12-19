"""Test the bandpass class"""

import numpy as np
import pytest

from lbg_tools import SED, Bandpass, library


def test_load_missing_band() -> None:
    """Loading a missing band should fail."""
    with pytest.raises(ValueError):
        Bandpass("fake")


def test_cant_change_band() -> None:
    """Changing band name after creation should fail."""
    # Create object
    bandpass = Bandpass(library.get_bands()[0])

    # Check that changing name fails
    with pytest.raises(AttributeError):
        bandpass.band = "fake"  # type: ignore


def test_mean_wavelengths() -> None:
    """Check that mean wavelength ascends for ugrizy"""
    wavelens = [Bandpass(band).mean_wavelength for band in "ugrizy"]
    assert np.all(np.diff(wavelens) > 0)


def test_effective_wavelength() -> None:
    """Check that bluer spectra make effective wavelength shorter."""
    bandpass = Bandpass(library.get_bands()[0])

    sed1 = SED(4, -22, -1)  # redder
    sed2 = SED(4, -22, -2)  # bluer

    eff_wavelen_1 = bandpass.calc_eff_wavelength(*sed1.truth)  # longer
    eff_wavelen_2 = bandpass.calc_eff_wavelength(*sed2.truth)  # shorter

    assert eff_wavelen_1 > eff_wavelen_2
