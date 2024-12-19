"""Implementation of the IGM model from Madau 1995

https://ui.adsabs.harvard.edu/abs/1995ApJ...441...18M/abstract
"""

import numpy as np

# Wavelengths and coefficients for equation 12, 15
# values of higher order terms retrieved from FSPS:
# https://github.com/cconroy20/fsps/blob/master/src/igm_absorb.f90#L63-L64
_tls_wavelen = np.array(
    [
        1215.67,
        1025.72,
        972.537,
        949.743,
        937.803,
        930.748,
        926.226,
        923.150,
        920.963,
        919.352,
        918.129,
        917.181,
        916.429,
        915.824,
        915.329,
        914.919,
        914.576,
    ]
)
_tls_coeff = np.array(
    [
        0.0036,
        0.0017,
        0.0011846,
        0.0009410,
        0.0007960,
        0.0006967,
        0.0006236,
        0.0005665,
        0.0005200,
        0.0004817,
        0.0004487,
        0.0004200,
        0.0003947,
        0.000372,
        0.000352,
        0.0003334,
        0.00031644,
    ]
)


def tls(wavelen: np.ndarray, z: float) -> np.ndarray:
    """Calculate optical depth contribution from Lyman-series

    Parameters
    ----------
    wavelen : np.ndarray
        Observed wavelength in Angstroms
    z : float
        Redshift

    Returns
    -------
    np.ndarray
        Optical depth contribution
    """
    # Evaluate power-law terms at every wavelength
    w = wavelen[:, None] / _tls_wavelen[None, :]
    tau = _tls_coeff * w**3.46

    # Mask values outside appropriate wavelength ranges
    mask = (w >= 1) & (w <= 1 + z)
    tau *= mask

    # Sum over every line in the series
    tau = tau.sum(axis=-1)

    return tau


def tlc(wavelen: np.ndarray, z: float) -> np.ndarray:
    """Calculate optical depth contribution from Lyman-continuum

    Using approximation of Eq. 16 from Madau 1995. See footnote 3.

    Parameters
    ----------
    wavelen : np.ndarray
        Observed wavelength in Angstroms
    z : float
        Redshift

    Returns
    -------
    np.ndarray
        Optical depth contribution
    """
    xc = wavelen / 911.75
    xm = 1 + z

    with np.errstate(divide="ignore", invalid="ignore"):
        tau = (
            0.25 * xc**3 * (xm**0.46 - xc**0.46)
            + 9.4 * xc**1.5 * (xm**0.18 - xc**0.18)
            - 0.7 * xc**3 * (xc ** (-1.32) - xm ** (-1.32))
            - 0.023 * (xm**1.68 - xc**1.68)
        )

    mask = xc <= 1 + z
    tau *= mask

    # Fix for low-wavelength continuum from FSPS
    # i.e., continuum fitting function decreases at low-wavelengths, which isn't
    # expected. As continuum tau starts to decrease towards lower wavelength,
    # we simply set values to the max value
    idx = np.nanargmax(tau)
    tau[:idx] = tau[idx]

    return tau


def tmet(wavelen: float | np.ndarray, z: float) -> np.ndarray:
    """Calculate optical depth contribution from metals

    Per the Madau model, this uses the Lyman-alpha wavelength,
    i.e. _tls_wavelen[0].

    Note this has essentially zero effect, but is included for completeness.

    Parameters
    ----------
    wavelen : np.ndarray
        Observed wavelength in Angstroms
    z : float
        Redshift

    Returns
    -------
    np.ndarray
        Optical depth contribution
    """
    # Evaluate power-law terms at every wavelength
    w = wavelen / _tls_wavelen[0]
    tau = 0.0017 * w**1.68

    # Mask values outside appropriate wavelength range
    mask = (w >= 1) & (w <= 1 + z)
    tau *= mask

    return tau


def igm_tau(wavelen: float | np.ndarray, z: float) -> np.ndarray:
    """Calculate optical depth of the IGM using the Madau model.

    Parameters
    ----------
    wavelen : float | np.ndarray
        Observed wavelength in Angstroms
    z : float
        Redshift

    Returns
    -------
    np.ndarray
       IGM optical depth
    """
    # Make sure wavelength is an array
    wavelen = np.atleast_1d(wavelen).astype(float)

    # Optical depth of IGM due to Lyman transitions
    return tls(wavelen, z) + tlc(wavelen, z) + tmet(wavelen, z)
