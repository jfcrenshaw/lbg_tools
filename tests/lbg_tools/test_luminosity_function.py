"""Test the luminosity function class."""

import numpy as np

from lbg_tools import Completeness, LuminosityFunction, library


def test_broadcasting() -> None:
    """Test that shapes broadcast as expected."""
    lf = LuminosityFunction()
    M = np.linspace(-24, -14, 3)
    z = np.linspace(3, 6, 5)
    assert lf(M[:, None], z[None, :]).shape == (M.size, z.size)  # type: ignore


def test_completeness() -> None:
    """Basic test of multiplying by completeness function."""
    # Original completeness function is the same true vs obs
    lf = LuminosityFunction()
    M = np.linspace(-24, -14, 3)
    z = 4
    true = lf.true(M, z)
    obs0 = lf.observed(M, z)
    obs1 = lf(M, z)  # Call method is same as observed method
    assert np.allclose(true, obs0)
    assert np.allclose(obs0, obs1)

    # Apply completeness and check that values are smaller
    lf *= Completeness(library.get_bands()[0], 26.0)
    true = lf.true(M, z)
    obs = lf(M, z)
    assert not np.allclose(true, obs)
    assert np.all(obs <= true)
