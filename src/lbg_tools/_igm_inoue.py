"""Implementation of the IGM model from Inoue 2014 (arXiv:1402.0677)"""

import numpy as np

# Table 2 from arXiv:1402.0677
_table = r"""
   2 (Ly$\alpha$) & 1215.67 & 1.690e-02 & 2.354e-03 & 1.026e-04 & 1.617e-04 & 5.390e-05 \\
   3 (Ly$\beta$) & 1025.72 & 4.692e-03 & 6.536e-04 & 2.849e-05 & 1.545e-04 & 5.151e-05 \\
   4 (Ly$\gamma$) &  972.537 & 2.239e-03 & 3.119e-04 & 1.360e-05 & 1.498e-04 & 4.992e-05 \\
   5 &  949.743 & 1.319e-03 & 1.837e-04 & 8.010e-06 & 1.460e-04 & 4.868e-05 \\
   6 &  937.803 & 8.707e-04 & 1.213e-04 & 5.287e-06 & 1.429e-04 & 4.763e-05 \\
   7 &  930.748 & 6.178e-04 & 8.606e-05 & 3.752e-06 & 1.402e-04 & 4.672e-05 \\
   8 &  926.226 & 4.609e-04 & 6.421e-05 & 2.799e-06 & 1.377e-04 & 4.590e-05 \\
   9 &  923.150 & 3.569e-04 & 4.971e-05 & 2.167e-06 & 1.355e-04 & 4.516e-05 \\
   10 &  920.963 & 2.843e-04 & 3.960e-05 & 1.726e-06 & 1.335e-04 & 4.448e-05 \\
   11 &  919.352 & 2.318e-04 & 3.229e-05 & 1.407e-06 & 1.316e-04 & 4.385e-05 \\
   12 &  918.129 & 1.923e-04 & 2.679e-05 & 1.168e-06 & 1.298e-04 & 4.326e-05 \\
   13 &  917.181 & 1.622e-04 & 2.259e-05 & 9.847e-07 & 1.281e-04 & 4.271e-05 \\
   14 &  916.429 & 1.385e-04 & 1.929e-05 & 8.410e-07 & 1.265e-04 & 4.218e-05 \\
   15 &  915.824 & 1.196e-04 & 1.666e-05 & 7.263e-07 & 1.250e-04 & 4.168e-05 \\
   16 &  915.329 & 1.043e-04 & 1.453e-05 & 6.334e-07 & 1.236e-04 & 4.120e-05 \\
   17 &  914.919 & 9.174e-05 & 1.278e-05 & 5.571e-07 & 1.222e-04 & 4.075e-05 \\
   18 &  914.576 & 8.128e-05 & 1.132e-05 & 4.936e-07 & 1.209e-04 & 4.031e-05 \\
   19 &  914.286 & 7.251e-05 & 1.010e-05 & 4.403e-07 & 1.197e-04 & 3.989e-05 \\
   20 &  914.039 & 6.505e-05 & 9.062e-06 & 3.950e-07 & 1.185e-04 & 3.949e-05 \\
   21 &  913.826 & 5.868e-05 & 8.174e-06 & 3.563e-07 & 1.173e-04 & 3.910e-05 \\
   22 &  913.641 & 5.319e-05 & 7.409e-06 & 3.230e-07 & 1.162e-04 & 3.872e-05 \\
   23 &  913.480 & 4.843e-05 & 6.746e-06 & 2.941e-07 & 1.151e-04 & 3.836e-05 \\
   24 &  913.339 & 4.427e-05 & 6.167e-06 & 2.689e-07 & 1.140e-04 & 3.800e-05 \\
   25 &  913.215 & 4.063e-05 & 5.660e-06 & 2.467e-07 & 1.130e-04 & 3.766e-05 \\
   26 &  913.104 & 3.738e-05 & 5.207e-06 & 2.270e-07 & 1.120e-04 & 3.732e-05 \\
   27 &  913.006 & 3.454e-05 & 4.811e-06 & 2.097e-07 & 1.110e-04 & 3.700e-05 \\
   28 &  912.918 & 3.199e-05 & 4.456e-06 & 1.943e-07 & 1.101e-04 & 3.668e-05 \\
   29 &  912.839 & 2.971e-05 & 4.139e-06 & 1.804e-07 & 1.091e-04 & 3.637e-05 \\
   30 &  912.768 & 2.766e-05 & 3.853e-06 & 1.680e-07 & 1.082e-04 & 3.607e-05 \\
   31 &  912.703 & 2.582e-05 & 3.596e-06 & 1.568e-07 & 1.073e-04 & 3.578e-05 \\
   32 &  912.645 & 2.415e-05 & 3.364e-06 & 1.466e-07 & 1.065e-04 & 3.549e-05 \\
   33 &  912.592 & 2.263e-05 & 3.153e-06 & 1.375e-07 & 1.056e-04 & 3.521e-05 \\
   34 &  912.543 & 2.126e-05 & 2.961e-06 & 1.291e-07 & 1.048e-04 & 3.493e-05 \\
   35 &  912.499 & 2.000e-05 & 2.785e-06 & 1.214e-07 & 1.040e-04 & 3.466e-05 \\
   36 &  912.458 & 1.885e-05 & 2.625e-06 & 1.145e-07 & 1.032e-04 & 3.440e-05 \\
   37 &  912.420 & 1.779e-05 & 2.479e-06 & 1.080e-07 & 1.024e-04 & 3.414e-05 \\
   38 &  912.385 & 1.682e-05 & 2.343e-06 & 1.022e-07 & 1.017e-04 & 3.389e-05 \\
   39 &  912.353 & 1.593e-05 & 2.219e-06 & 9.673e-08 & 1.009e-04 & 3.364e-05 \\
   40 &  912.324 & 1.510e-05 & 2.103e-06 & 9.169e-08 & 1.002e-04 & 3.339e-05 \\
"""
_lambda_L = 911.8

_tls_params_list = []
for row in _table.split(" \\")[:-1]:
    _tls_params_list.append([float(p) for p in row.split(" & ")[1:]])
_tls_params = np.array(_tls_params_list)


def tls_laf(wavelen: np.ndarray, z: float) -> np.ndarray:
    """Calculate optical depth contribution from Lyman-series: Lya Forest

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
    w = wavelen[:, None] / _tls_params[None, :, 0]
    vals = _tls_params[:, 1:4] * np.power(w[..., None], [1.2, 3.7, 5.5])

    # Zero points outside of appropriate wavelength ranges
    mask = (w > 1) & (w < 1 + z)
    vals *= mask[..., None]

    # Apply mask so only appropriate coefficient contributes to each
    mask = w < 2.2
    vals[..., 0] *= mask

    mask = (w >= 2.2) & (w < 5.7)
    vals[..., 1] *= mask

    mask = w >= 5.7
    vals[..., 2] *= mask

    # Sum over last axis. After masks, there is only non-zero element in each of
    # these sums, so the sum is really just to pick up that non-zero element
    vals = vals.sum(axis=-1)

    # Finally sum over the contributions from every Lyman transition
    # Note this could be combined with previous sum, but I separated them to
    # make the logic a little easier to parse.
    tau = vals.sum(axis=-1)

    return tau


def tls_dla(wavelen: np.ndarray, z: float) -> np.ndarray:
    """Calculate optical depth contribution from Lyman-series: Damped Lya Systems

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
    w = wavelen[:, None] / _tls_params[None, :, 0]
    vals = _tls_params[:, 4:] * np.power(w[..., None], [2.0, 3.0])

    # Zero points outside of appropriate wavelength ranges
    mask = (w > 1) & (w < 1 + z)
    vals *= mask[..., None]

    # Apply mask so only appropriate coefficient contributes to each
    mask = w < 3.0
    vals[..., 0] *= mask

    mask = w >= 3.0
    vals[..., 1] *= mask

    # Sum over last axis. After masks, there is only non-zero element in each of
    # these sums, so the sum is really just to pick up that non-zero element
    vals = vals.sum(axis=-1)

    # Finally sum over the contributions from every Lyman transition
    # Note this could be combined with previous sum, but I separated them to
    # make the logic a little easier to parse.
    tau = vals.sum(axis=-1)

    return tau


def tlc_laf(wavelen: np.ndarray, z: float) -> np.ndarray:
    """Calculate optical depth contribution from Lyman-continuum: Lya Forest

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
    with np.errstate(divide="ignore"):
        w = wavelen / _lambda_L

        tau = np.zeros_like(w)

        if z < 1.2:
            mask = (w >= 1) & (z < 1.2) & (w < 1 + z)
            vals = 0.325 * (w**1.2 - (1 + z) ** (-0.9) * w**2.1)
            tau[mask] = vals[mask]

        if (z >= 1.2) & (z < 4.7):
            mask = (w >= 1) & (z >= 1.2) & (z < 4.7) & (w < 2.2)
            vals = (
                2.55e-2 * (1 + z) ** 1.6 * w**2.1 + 0.325 * w**1.2 - 0.25 * w**2.1
            )
            tau[mask] = vals[mask]

            mask = (w >= 1) & (z >= 1.2) & (z < 4.7) & (w >= 2.2) & (w < 1 + z)
            vals = 2.55e-2 * ((1 + z) ** 1.6 * w**2.1 - w**3.7)
            tau[mask] = vals[mask]

        else:
            mask = (w >= 1) & (z >= 4.7) & (w < 2.2)
            vals = (
                5.22e-4 * (1 + z) ** 3.4 * w**2.1
                + 0.325 * w**1.2
                - 3.14e-2 * w**2.1
            )
            tau[mask] = vals[mask]

            mask = (w >= 1) & (z >= 4.7) & (w >= 2.2) & (w < 5.7)
            vals = (
                5.22e-4 * (1 + z) ** 3.4 * w**2.1
                + 0.218 * w**2.1
                - 2.55e-2 * w**3.7
            )
            tau[mask] = vals[mask]

            mask = (w >= 1) & (z >= 4.7) & (w >= 5.7) & (w < 1 + z)
            vals = 5.22e-4 * ((1 + z) ** 3.4 * w**2.1 - w**5.5)
            tau[mask] = vals[mask]

    # Fix for low-wavelength continuum from FSPS
    # i.e., continuum fitting function decreases at low-wavelengths, which isn't
    # expected. As continuum tau starts to decrease towards lower wavelength,
    # we simply set values to the max value
    idx = np.nanargmax(tau)
    tau[:idx] = tau[idx]

    return tau


def tlc_dla(wavelen: np.ndarray, z: float) -> np.ndarray:
    """Calculate optical depth contribution from Lyman-continuum: Damped Lya Systems

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
    with np.errstate(divide="ignore"):
        w = wavelen / _lambda_L

        tau = np.zeros_like(w)

        if z < 2.0:
            mask = (w >= 1) & (z < 2) & (w < 1 + z)
            vals = (
                0.211 * (1 + z) ** 2
                - 7.66e-2 * (1 + z) ** 2.3 * w ** (-0.3)
                - 0.135 * w**2
            )
            tau[mask] = vals[mask]

        if z >= 2.0:
            mask = (w >= 1) & (z >= 2) & (w < 3)
            vals = (
                0.634
                + 4.7e-2 * (1 + z) ** 3
                - 1.78e-2 * (1 + z) ** 3.3 * w ** (-0.3)
                - 0.135 * w**2
                - 0.291 * w ** (-0.3)
            )
            tau[mask] = vals[mask]

            mask = (w >= 1) & (z >= 2) & (w >= 3) & (w < 1 + z)
            vals = (
                4.7e-2 * (1 + z) ** 3
                - 1.78e-2 * (1 + z) ** 3.3 * w ** (-0.3)
                - 2.92e-2 * w**3
            )
            tau[mask] = vals[mask]

    # Fix for low-wavelength continuum from FSPS
    # i.e., continuum fitting function decreases at low-wavelengths, which isn't
    # expected. As continuum tau starts to decrease towards lower wavelength,
    # we simply set values to the max value
    idx = np.nanargmax(tau)
    tau[:idx] = tau[idx]

    return tau


def igm_tau(wavelen: float | np.ndarray, z: float) -> np.ndarray:
    """Calculate optical depth of the IGM using the Inoue model.

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
    return (
        tls_laf(wavelen, z)
        + tls_dla(wavelen, z)
        + tlc_laf(wavelen, z)
        + tlc_dla(wavelen, z)
    )
