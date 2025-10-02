"""Utils for cosmology objects"""

from typing import Any

import numpy as np
from astropy import units as u
from astropy.cosmology import Cosmology

# Protected import for optional dependency
try:
    import pyccl as ccl
except ImportError:
    ccl = None


def check_cosmology(cosmology: Any) -> None:
    """Check if cosmology is valid.

    Parameters
    ----------
    cosmology : Any
        Cosmology object to check. Will raise an error unless the object is of
        type astropy.cosmology.Cosmology or pyccl.Cosmology. Note if you want
        to use pyccl, you must install it yourself.
    """
    if not (
        isinstance(cosmology, Cosmology)
        or (ccl is not None and isinstance(cosmology, ccl.Cosmology))
    ):
        raise TypeError("Cosmology must be an astropy or pyccl Cosmology object.")


def luminosity_distance(
    cosmology: "Cosmology | ccl.Cosmology",
    z: float | np.ndarray,
) -> float | np.ndarray:
    """Calculate luminosity distance in parsecs.

    Parameters
    ----------
    cosmology : Cosmology or pyccl.CosmologyCosmology or pyccl.Cosmology
        Cosmology to use. Must be an instance of astropy.cosmology.Cosmology
        or pyccl.Cosmology.
    z : float or Any
        Redshift or array of redshifts.

    Returns
    -------
    float or Any
        Luminosity distance in parsecs.
    """
    check_cosmology(cosmology)
    if ccl is not None and isinstance(cosmology, ccl.Cosmology):
        z = np.atleast_1d(z)
        dL = 1e6 * ccl.luminosity_distance(cosmology, 1 / (1 + z.flatten()))
        dL = dL.reshape(z.shape)
    else:
        dL = cosmology.luminosity_distance(z).to(u.pc).value

    return dL


def diff_comoving_volume(
    cosmology: "Cosmology | ccl.Cosmology",
    z: float | np.ndarray,
) -> float | np.ndarray:
    """Calculate the differential comoving volume as a function of redshift

    Parameters
    ----------
    cosmology : Cosmology or pyccl.CosmologyCosmology or pyccl.Cosmology
        Cosmology to use. Must be an instance of astropy.cosmology.Cosmology
        or pyccl.Cosmology.
    z : float or Any
        Redshift or array of redshifts.

    Returns
    -------
    float or Any
        Differential comoving volume in Mpc^3 deg^-2
    """
    check_cosmology(cosmology)

    # Some constants for converting steradian -> deg^2
    if ccl is not None and isinstance(cosmology, ccl.Cosmology):
        z = np.atleast_1d(z)
        dVda = ccl.comoving_volume_element(cosmology, 1 / (1 + z.flatten()))
        dVdz = dVda.reshape(z.shape) / (1 + z) ** 2
    else:
        dVdz = cosmology.differential_comoving_volume(z).value

    # Convert sr^{-1} to deg^{-2}
    A_sky = 41_253  # deg^2
    deg2_per_ster = A_sky / (4 * np.pi)
    dVdz /= deg2_per_ster

    return dVdz
